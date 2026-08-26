# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
)
from lmcache.logging import init_logger
from lmcache.v1.cold_start_perf import cold_start_perf_log
from lmcache.v1.cache_engine import LayerwiseStoreResult
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.forward_context import is_forward_context_available
from vllm_ascend.live_source_handoff import (
    LIVE_SOURCE_EVENT_HANDOFF_KEY,
)
import torch

# First Party
from lmcache_ascend.v1.content_diagnostics import (
    log_npu_content_diagnostic_event,
    npu_content_diagnostics_enabled,
)
from lmcache_ascend.v1.remote_fill_producer import parse_remote_fill_handoff

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request

logger = init_logger(__name__)

_MOONCAKE_PREFERRED_SEGMENT_CONFIG = "lmcache.mooncake_preferred_segment"
_REMOTE_FILL_ROUTING_KEYS = (
    "do_remote_decode",
    "do_remote_prefill",
    "last_token_id",
    "live_split_capabilities",
    "live_split_transfer_id",
    "num_prompt_blocks",
    "remote_block_ids",
    "remote_dcp_size",
    "remote_dp_rank",
    "remote_engine_id",
    "remote_host",
    "remote_multi_nodes_meta_mapping",
    "remote_pcp_size",
    "remote_port",
    "remote_ptp_size",
    "remote_request_id",
)
_REMOTE_FILL_PUBLIC_HANDOFF_KEYS = (
    "transfer_id",
    "request_attempt",
    "source_engine_id",
    "destination_engine_id",
    "destination_engine_epoch",
    "destination_dp_rank",
    "shared_cache_generation",
    "destination_tp_size",
    "destination_dp_size",
    "global_te_push",
    "token_hash_algorithm",
    "python_hash_seed",
)


def _remote_fill_response_params(
    params: Mapping[str, Any],
    terminal: Mapping[str, str | int],
    base: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the pointer-free producer response returned through vLLM."""

    result = dict(base or {})
    for key in _REMOTE_FILL_ROUTING_KEYS:
        if key in params:
            result[key] = params[key]
    raw_handoff = params.get("lmcache.remote_fill")
    public_handoff: dict[str, Any] = {}
    if isinstance(raw_handoff, Mapping):
        for key in _REMOTE_FILL_PUBLIC_HANDOFF_KEYS:
            if key in raw_handoff:
                public_handoff[key] = raw_handoff[key]
    public_handoff["terminal"] = dict(terminal)
    result["lmcache.remote_fill"] = public_handoff
    return result


def _remote_fill_handoff_qualified(request_configs: Any) -> bool:
    """Return whether this request may activate direct remote fill."""
    try:
        handoff = parse_remote_fill_handoff(request_configs)
    except ValueError:
        return False
    return bool(handoff is not None and handoff.global_te_push)


def _remote_fill_request_qualified(request: Any) -> bool:
    """Cache handoff qualification on one worker request metadata object."""

    cached = getattr(request, "_lmcache_remote_fill_qualified", None)
    if cached is None:
        cached = _remote_fill_handoff_qualified(request.request_configs)
        request._lmcache_remote_fill_qualified = cached
    return bool(cached)


def _prepare_remote_fill_persistent_placement(
    request_configs: Any,
) -> bool:
    """Select the validated prefiller-local Mooncake default for persistence."""

    if not _remote_fill_handoff_qualified(request_configs):
        return False
    assert isinstance(request_configs, dict)
    request_configs.pop(_MOONCAKE_PREFERRED_SEGMENT_CONFIG, None)
    return True


def _validate_remote_fill_sleep_mode(config: Any, vllm_config: Any) -> None:
    """Reject an allocator lifecycle that cannot yet drain armed writes."""

    if (
        getattr(config, "enable_remote_lmcache_store", False)
        and bool(
            getattr(
                getattr(vllm_config, "model_config", None),
                "enable_sleep_mode",
                False,
            )
        )
    ):
        raise ValueError(
            "direct remote LMCache store is incompatible with vLLM sleep mode "
            "until armed transfers can be drained before allocator changes"
        )


@dataclass
class LiveSourceWorkerMetadata(KVConnectorWorkerMetadata):
    descriptors: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    remote_fill_results: dict[str, dict[str, str | int]] = field(
        default_factory=dict
    )

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "LiveSourceWorkerMetadata":
        if not isinstance(other, LiveSourceWorkerMetadata):
            raise TypeError("Cannot aggregate incompatible LMCache worker metadata")
        merged = {req_id: list(items) for req_id, items in self.descriptors.items()}
        for req_id, items in other.descriptors.items():
            merged.setdefault(req_id, []).extend(items)
        results = dict(self.remote_fill_results)
        for req_id, result in other.remote_fill_results.items():
            existing = results.get(req_id)
            if existing is not None and existing != result:
                raise ValueError(
                    "Tensor-parallel ranks reported different remote-fill outcomes"
                )
            results[req_id] = dict(result)
        return LiveSourceWorkerMetadata(merged, results)


@dataclass(slots=True)
class _LiveSourceReadyFence:
    event: Any
    event_source: str
    ready_at_finalize: Optional[bool]


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        # LMCache-NPU initializes this field only for worker connectors;
        # EngineCore also constructs this implementation for the scheduler.
        self.use_layerwise = bool(
            getattr(
                self,
                "use_layerwise",
                getattr(self.config, "use_layerwise", False),
            )
        )
        self.store_async = self.config.store_async
        get_extra = getattr(self.config, "get_extra_config_value", None)
        _validate_remote_fill_sleep_mode(self.config, vllm_config)
        self._remote_store_requested = bool(
            getattr(self.config, "enable_remote_lmcache_store", False)
            and getattr(self.config, "pd_role", None) != "receiver"
        )
        self._direct_store_requested = bool(
            self._remote_store_requested
            or (
                get_extra("mooncake_direct_npu_prefill_store", False)
                if callable(get_extra)
                else False
            )
        )
        # The provider must not self-activate the hybrid wire format.  A new
        # LMCache provider can be paired with an older vLLM/Mooncake consumer;
        # in that case an unnegotiated latent extension could discard the
        # otherwise valid group-1 source descriptor. AscendMultiConnector enables this
        # only after both sides of the in-process transport negotiate support.
        self._live_latent_split_requested = False
        self._direct_store_observed_layers: set[str] = set()
        self._direct_store_step_supported: Optional[bool] = None
        self._completed_layerwise_stores: dict[
            tuple[str, int], LayerwiseStoreResult
        ] = {}
        self._scheduler_live_sources: dict[str, list[dict[str, Any]]] = {}
        self._scheduler_remote_fill_results: dict[
            str, dict[str, str | int]
        ] = {}
        self._unfenced_live_stores: dict[str, ReqMeta] = {}
        self._latest_live_source_ready_event: Any = None
        self._latest_live_source_ready_event_source = "missing"
        self._latest_direct_source_ready_events: dict[str, Any] = {}
        self._live_source_ready_fences: dict[str, _LiveSourceReadyFence] = {}
        self._finalized_live_source_submissions: set[str] = set()

        if self._direct_store_requested:
            extra = self.config.extra_config or {}
            valid = (
                self.store_async
                and self.config.store_async_max_queue_size == 2
                and str(self.config.remote_url).startswith("mooncakestore://")
                and self.use_layerwise
                and (
                    getattr(self, "_remote_store_requested", False)
                    or self.config.enable_shared_cpu_cache
                )
                and extra.get("mooncake_page_first_multi_buffer", False)
                and extra.get("mooncake_layer_merged_page_objects", False)
                and extra.get("save_only_first_rank", False)
                and extra.get("use_ascend_direct", False)
                and not extra.get("save_chunk_meta", False)
            )
            if not valid:
                raise ValueError(
                    "mooncake_direct_npu_prefill_store requires store_async=true, "
                    "store_async_max_queue_size=2, layerwise merged Mooncake pages, "
                    "save_only_first_rank, use_ascend_direct, and "
                    "save_chunk_meta=false"
                )
        if (
            role != KVConnectorRole.SCHEDULER
            and self.kv_role != "kv_consumer"
            and self.use_layerwise
            and self.store_async
            and not self._direct_store_requested
        ):
            raise ValueError(
                "Layerwise storing is not supported with async store"
            )
        logger.debug("store_async: %s", self.store_async)

    def configure_live_latent_source(self, enabled: bool) -> None:
        """Apply the two-sided hybrid-transport capability decision."""
        configure = getattr(super(), "configure_live_latent_source", None)
        if callable(configure):
            configure(enabled)
            return
        # Preserve startup compatibility with an older LMCache-NPU base.  It
        # has no hybrid transport hook, so fail closed without affecting the
        # established group-1 path.
        self._live_latent_split_requested = False

    def _producer_fence_handoff_targets(
        self,
        requests: Iterable[ReqMeta],
    ) -> tuple[tuple[str, int], ...]:
        mode = getattr(
            self.config, "remote_fill_submission_mode", "final_deferred"
        )
        return tuple(
            sorted({
                (request.req_id, len(request.token_ids))
                for request in requests
                if (
                    request.live_source_requested and request.is_last_prefill
                )
                or (
                    getattr(self, "_remote_store_requested", False)
                    and _remote_fill_request_qualified(request)
                    and (mode == "per_chunk" or request.is_last_prefill)
                )
            })
        )

    def start_load_kv(
        self,
        forward_context: ForwardContext,
        **kwargs: Any,
    ) -> None:
        """Start loads and arm a real producer-event handoff when required."""

        super().start_load_kv(forward_context, **kwargs)
        if forward_context.attn_metadata is None or not self.config.dsa_two_groups:
            return
        requests = self._direct_prefill_requests() or []
        targets = self._producer_fence_handoff_targets(requests)
        if not targets or not self._latent_layer_names:
            return
        existing = forward_context.additional_kwargs.setdefault(
            LIVE_SOURCE_EVENT_HANDOFF_KEY, targets
        )
        if existing != targets:
            forward_context.additional_kwargs.pop(
                LIVE_SOURCE_EVENT_HANDOFF_KEY, None
            )
            logger.warning(
                "Conflicting live-source event handoff; using persistent fallback"
            )

    def capture_live_source_event_handoff(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        """Retain the published event across deferred MTP finalization."""

        arm = forward_context.additional_kwargs.pop(
            LIVE_SOURCE_EVENT_HANDOFF_KEY, None
        )
        if arm is None:
            return False
        requests = self._direct_prefill_requests() or ()
        targets = self._producer_fence_handoff_targets(requests)
        if arm != targets:
            return False
        attn_metadata = forward_context.attn_metadata
        # A single event cannot fence multiple DBO microbatch streams.
        if not isinstance(attn_metadata, Mapping):
            return False

        for producer_layer_name in reversed(self._latent_layer_names):
            source_ready_event = self._source_ready_event(
                producer_layer_name, attn_metadata
            )
            if source_ready_event is not None:
                break
        else:
            return False

        metadata = self._parent._get_connector_metadata()
        if getattr(metadata, "_live_source_event_handoff", None) is not None:
            delattr(metadata, "_live_source_event_handoff")
            logger.warning(
                "Duplicate live-source event handoff; using persistent fallback"
            )
            return False
        metadata._live_source_event_handoff = (  # type: ignore[attr-defined]
            targets,
            source_ready_event,
        )
        return True

    def _direct_prefill_requests(self) -> Optional[list[ReqMeta]]:
        if (
            not getattr(self, "_direct_store_requested", False)
            or self.kv_role == "kv_consumer"
            or self.lmcache_engine is None
            or self._parent._connector_metadata is None
        ):
            return None
        metadata = self._parent._get_connector_metadata()
        requests = list(getattr(metadata, "requests", ()))
        if not requests or any(
            request.is_sparse_decode
            or request.is_decode_window_save
            or not request.slot_mapping
            or (
                self.config.dsa_two_groups
                and (
                    request.save_spec is None
                    or request.save_spec.can_save_indexer
                )
                and not request.indexer_slot_mapping
            )
            or (
                self.kv_role != "kv_producer"
                and not request.live_source_requested
                and (
                    request.save_spec is None
                    or not request.save_spec.can_save
                )
            )
            for request in requests
        ):
            return None
        return requests

    def _consume_completed_layerwise_store(
        self,
        request: ReqMeta,
        kv_group: int,
        completed: bool,
        result: Optional[LayerwiseStoreResult],
    ) -> None:
        super()._consume_completed_layerwise_store(
            request, kv_group, completed, result
        )
        if (
            self._direct_store_requested
            and completed
            and result is not None
            and result.committed_end > 0
        ):
            self._completed_layerwise_stores[(request.req_id, kv_group)] = result

    def _abort_save_step(self, requests: Iterable[ReqMeta]) -> None:
        requests = tuple(requests)
        req_ids = {request.req_id for request in requests}
        try:
            super()._abort_save_step(requests)
        finally:
            if is_forward_context_available():
                get_forward_context().additional_kwargs.pop(
                    LIVE_SOURCE_EVENT_HANDOFF_KEY, None
                )
            metadata = getattr(self._parent, "_connector_metadata", None)
            if metadata is not None and hasattr(
                metadata, "_live_source_event_handoff"
            ):
                delattr(metadata, "_live_source_event_handoff")
            self._direct_store_step_supported = None
            self._direct_store_observed_layers.clear()
            self._latest_live_source_ready_event = None
            self._latest_live_source_ready_event_source = "missing"
            getattr(self, "_latest_direct_source_ready_events", {}).clear()
            for key in list(self._completed_layerwise_stores):
                if key[0] in req_ids:
                    self._completed_layerwise_stores.pop(key, None)
            if self.lmcache_engine is not None:
                try:
                    self.lmcache_engine.wait_for_direct_stores(req_ids)
                except Exception:
                    logger.exception(
                        "Failed to drain direct stores while aborting %s",
                        sorted(req_ids),
                    )
                for req_id in req_ids:
                    self._unfenced_live_stores.pop(req_id, None)
                    fences = getattr(self, "_live_source_ready_fences", None)
                    if fences is not None:
                        fences.pop(req_id, None)
                    finalized = getattr(
                        self, "_finalized_live_source_submissions", None
                    )
                    if finalized is not None:
                        finalized.discard(req_id)
                self.lmcache_engine.drop_direct_store_states(req_ids)

    @staticmethod
    def _source_ready_event(layer_name: str, attn_metadata: Any) -> Any:
        metadata = attn_metadata
        if isinstance(attn_metadata, Mapping):
            metadata = attn_metadata.get(layer_name)
            if metadata is None and layer_name.endswith(".indexer.k_cache"):
                # SFA publishes one event after writing both the latent and
                # unbundled indexer caches, but ForwardContext stores it under
                # the latent attention-layer name.  Resolve the synthetic
                # indexer cache callback back to that unique sibling layer.
                sibling_metadata = [
                    value
                    for name, value in attn_metadata.items()
                    if isinstance(name, str)
                    and name.rsplit(".", 1)[0] + ".indexer.k_cache"
                    == layer_name
                ]
                if len(sibling_metadata) == 1:
                    metadata = sibling_metadata[0]
        return getattr(metadata, "reshape_cache_event", None)

    @staticmethod
    def _query_source_ready_event(event: Any) -> Optional[bool]:
        query = getattr(event, "query", None)
        if not callable(query):
            return None
        try:
            return bool(query())
        except Exception:
            logger.warning("Failed to query live-source NPU event", exc_info=True)
            return None

    def _fence_live_source_descriptors(self) -> None:
        fences = getattr(self, "_live_source_ready_fences", None)
        if not fences:
            return
        assert self.lmcache_engine is not None

        by_event: dict[int, tuple[Any, list[tuple[str, _LiveSourceReadyFence]]]] = {}
        for req_id, fence in fences.items():
            entry = by_event.setdefault(id(fence.event), (fence.event, []))
            entry[1].append((req_id, fence))

        completed: list[str] = []
        for event, requests in by_event.values():
            ready_before_publish = (
                self._query_source_ready_event(event)
                if npu_content_diagnostics_enabled()
                else None
            )
            started = time.perf_counter()
            # This is an event-local producer fence.  It does not drain
            # unrelated NPU streams or invoke torch.npu.synchronize().
            event.synchronize()
            wait_ms = (time.perf_counter() - started) * 1000
            req_ids = [req_id for req_id, _ in requests]
            finalize_readiness = getattr(
                self.lmcache_engine, "finalize_live_source_readiness", None
            )
            if callable(finalize_readiness):
                finalize_readiness(req_ids)
            for req_id, fence in requests:
                cold_start_perf_log(
                    logger,
                    "live_source_ready_fence",
                    req_id=req_id,
                    event_source=fence.event_source,
                    ready_at_finalize=fence.ready_at_finalize,
                    ready_before_publish=ready_before_publish,
                    wait_ms=round(wait_ms, 3),
                    fence_scope="producer_event",
                )
                log_npu_content_diagnostic_event(
                    "group1_source_ready_fence",
                    req_id=req_id,
                    event_source=fence.event_source,
                    ready_at_finalize=fence.ready_at_finalize,
                    ready_before_publish=ready_before_publish,
                    ready_after_fence=True,
                    wait_ms=round(wait_ms, 3),
                    query_precedes_device_readback=True,
                    fence_scope="producer_event",
                )
                completed.append(req_id)
        for req_id in completed:
            fences.pop(req_id, None)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Use one direct page submission after all required layers are ready."""
        requests = self._direct_prefill_requests()
        if requests is None:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return
        if self._remote_store_requested:
            for request in requests:
                if _remote_fill_request_qualified(request):
                    request.request_configs.pop(
                        _MOONCAKE_PREFERRED_SEGMENT_CONFIG, None
                    )
        expected = set(self._latent_layer_names)
        if self.config.dsa_two_groups:
            expected.update(self._indexer_layer_names)
        if not expected:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return
        if self._direct_store_step_supported is None:
            self._direct_store_step_supported = self._preflight_direct_store(
                requests
            )
        if not self._direct_store_step_supported:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return
        if layer_name in expected:
            direct_ready_events = getattr(
                self, "_latest_direct_source_ready_events", None
            )
            if direct_ready_events is None:
                direct_ready_events = {}
                self._latest_direct_source_ready_events = direct_ready_events
            if layer_name in self._direct_store_observed_layers:
                self._direct_store_observed_layers.clear()
                self._latest_live_source_ready_event = None
                self._latest_live_source_ready_event_source = "missing"
                direct_ready_events.clear()
            self._direct_store_observed_layers.add(layer_name)
            source_ready_event = LMCacheAscendConnectorV1Impl._source_ready_event(
                layer_name, attn_metadata
            )
            if source_ready_event is not None:
                direct_ready_events[layer_name] = source_ready_event
                self._latest_live_source_ready_event = source_ready_event
                self._latest_live_source_ready_event_source = (
                    "attn_metadata.reshape_cache_event"
                )
        if not expected.issubset(self._direct_store_observed_layers):
            return

        retain_remote_fill_fence = bool(
            getattr(self, "_remote_store_requested", False)
            and any(_remote_fill_request_qualified(request) for request in requests)
        )
        complete_events = getattr(
            self, "_latest_direct_source_ready_events", {}
        )
        source_ready_events: tuple[Any, ...] = ()
        if expected.issubset(complete_events):
            unique_events: dict[int, Any] = {}
            for producer_event in complete_events.values():
                unique_events.setdefault(id(producer_event), producer_event)
            source_ready_events = tuple(unique_events.values())
        self._submit_direct_prefill_requests(
            requests,
            source_ready_event=self._latest_live_source_ready_event,
            source_ready_event_source=(
                self._latest_live_source_ready_event_source
            ),
            source_ready_events=source_ready_events,
        )
        if retain_remote_fill_fence:
            return
        self._direct_store_observed_layers.clear()
        self._latest_live_source_ready_event = None
        self._latest_live_source_ready_event_source = "missing"
        getattr(self, "_latest_direct_source_ready_events", {}).clear()

    def _direct_group_caches(self) -> dict[int, list]:
        self._refresh_kvcaches_list()
        groups = {0: self._kvcaches_for_group(0)}
        if self.config.dsa_two_groups:
            groups[1] = self._kvcaches_for_group(1)
        return {group: caches for group, caches in groups.items() if caches}

    def _direct_selected_groups(
        self, request: ReqMeta, group_caches: dict[int, list]
    ) -> dict[int, list]:
        selected = dict(group_caches)
        if (
            getattr(self, "_remote_store_requested", False)
            and _remote_fill_request_qualified(request)
        ):
            # Persistent SaveSpec flags describe which immutable pages still
            # need Mooncake puts.  The direct destination has an independent
            # LocalCPU prefix, so it needs both authoritative source groups;
            # the engine's persistent existence probe still suppresses
            # duplicate Mooncake writes.
            return selected
        save_spec = request.save_spec
        if save_spec is not None and self.config.dsa_two_groups:
            if not save_spec.can_save_latent:
                selected.pop(0, None)
            if not save_spec.can_save_indexer:
                selected.pop(1, None)
        return selected

    def _direct_request_inputs(
        self, request: ReqMeta, group_caches: dict[int, list]
    ) -> tuple[dict[int, list], dict[int, torch.Tensor], int]:
        mapping_base = int(getattr(request, "save_slot_mapping_base", 0) or 0)
        selected = self._direct_selected_groups(request, group_caches)
        slot_mappings = {
            group: self._windowed_sparse_save_mapping(request, group, mapping_base)
            for group in selected
        }
        if any(mapping is None for mapping in slot_mappings.values()):
            mapping_base = 0
            slot_mappings = {
                group: (
                    request.indexer_slot_mapping[0]
                    if group
                    else request.slot_mapping[0]
                )
                for group in selected
            }
        return selected, slot_mappings, mapping_base

    def _preflight_direct_store(self, requests: list[ReqMeta]) -> bool:
        assert self.lmcache_engine is not None
        group_caches = self._direct_group_caches()
        selected_groups = {
            group
            for request in requests
            for group in self._direct_selected_groups(request, group_caches)
        }
        selected = {
            group: group_caches[group] for group in sorted(selected_groups)
        }
        return not selected or self.lmcache_engine.direct_prefill_plan_supported(
            selected
        )

    def _submit_direct_prefill_requests(
        self,
        requests: list[ReqMeta],
        adopted_requests: Optional[set[str]] = None,
        *,
        finish_batch: bool = False,
        source_ready_event: Any = None,
        source_ready_event_source: str = "missing",
        source_ready_events: tuple[Any, ...] = (),
    ) -> None:
        """Submit direct pages; also used by the wait-for-save fallback."""
        assert self.lmcache_engine is not None
        live_source_ready_fences = getattr(
            self, "_live_source_ready_fences", {}
        )
        finalized_live_sources = getattr(
            self, "_finalized_live_source_submissions", set()
        )
        requests = [
            request
            for request in requests
            if request.req_id not in self._unfenced_live_stores
            and request.req_id not in live_source_ready_fences
            and request.req_id not in finalized_live_sources
        ]
        if not requests:
            return
        if not finish_batch and getattr(
            self, "_remote_store_requested", False
        ):
            # RemoteFill accepts readiness only from the exact post-forward
            # request/frontier handoff. Callback events remain retained until
            # _finish_save_batch submits this chunk.
            requests = [
                request
                for request in requests
                if not _remote_fill_request_qualified(request)
            ]
            if not requests:
                return
        group_caches = self._direct_group_caches()
        for request in requests:
            live_source = bool(getattr(request, "live_source_requested", False))
            selected, slot_mappings, mapping_base = self._direct_request_inputs(
                request, group_caches
            )
            live_selected = group_caches if live_source else selected
            if not selected and not live_selected:
                continue
            load_spec = getattr(request, "load_spec", None)
            committed_prefix = getattr(load_spec, "dsa_committed_end", None)
            if committed_prefix is None:
                committed_prefix = getattr(load_spec, "lmcache_cached_tokens", 0)
            verified_prefix_end = min(
                mapping_base,
                int(committed_prefix or 0),
            )
            live_token_ids = (
                getattr(request, "live_source_token_ids", None)
                if live_source
                and getattr(request, "live_source_token_ids", None)
                else request.token_ids
            )
            live_latent_mapping = getattr(
                request, "live_source_slot_mapping", None
            )
            live_indexer_mapping = getattr(
                request, "live_source_indexer_slot_mapping", None
            )
            live_slot_mappings = slot_mappings
            if live_source:
                live_slot_mappings = (
                    {
                        group: (
                            live_indexer_mapping[0]
                            if group
                            else live_latent_mapping[0]
                        )
                        for group in live_selected
                    }
                    if live_latent_mapping
                    and (
                        not self.config.dsa_two_groups
                        or live_indexer_mapping
                    )
                    else slot_mappings
                )
            parallel = self._vllm_config.parallel_config
            topology_supported = (
                int(getattr(parallel, "pipeline_parallel_size", 1)) == 1
                and int(getattr(parallel, "prefill_context_parallel_size", 1)) == 1
                and int(getattr(parallel, "decode_context_parallel_size", 1)) == 1
            )
            remote_fill_request = bool(
                self._remote_store_requested
                and _remote_fill_request_qualified(request)
            )
            if remote_fill_request:
                # Mandatory persistence is independent of decoder placement.
                # Removing an old decoder hint makes Mooncake use the startup-
                # validated prefiller-local default. Direct LocalCPU fill also
                # supersedes the legacy live NPU source for this request.
                request.request_configs.pop(
                    _MOONCAKE_PREFERRED_SEGMENT_CONFIG, None
                )
                live_source = False
                self.lmcache_engine.discard_live_source_descriptor(request.req_id)
            if live_source and not topology_supported:
                logger.warning(
                    "Live split disabled for %s: PP/PCP/DCP must all equal 1",
                    request.req_id,
                )
                live_source = False
            if (
                live_source
                and request.is_last_prefill
                and source_ready_event is None
            ):
                # The descriptor exposes model-cache addresses to a remote
                # reader.  A current-stream event is not a valid substitute
                # for the attention producer's reshape_cache_event: Group 1
                # may have been restored or scattered on another stream.
                # Fail closed to the persistent path instead of publishing
                # bytes whose producer ordering is unknown.
                self.lmcache_engine.discard_live_source_descriptor(
                    request.req_id
                )
                logger.warning(
                    "Live split disabled for %s: final Group-1 source has "
                    "no producer NPU event",
                    request.req_id,
                )
                cold_start_perf_log(
                    logger,
                    "live_source_missing_producer_event",
                    req_id=request.req_id,
                    event_source=source_ready_event_source,
                    action="persistent_only",
                )
                log_npu_content_diagnostic_event(
                    "group1_source_missing_producer_event",
                    req_id=request.req_id,
                    event_source=source_ready_event_source,
                    action="persistent_only",
                    fallback_event_created=False,
                )
                live_source = False
            if live_source:
                live_groups = (
                    (0, 1)
                    if getattr(self, "_live_latent_split_requested", False)
                    and get_tensor_model_parallel_rank() == 0
                    else (1,)
                )
                self.lmcache_engine.begin_live_source_descriptor(
                    request.req_id, live_groups
                )
                self.lmcache_engine.capture_live_source_step(
                    request.req_id,
                    live_token_ids,
                    live_selected,
                    live_slot_mappings,
                    request.request_configs,
                    0,
                    request.is_last_prefill,
                )
            else:
                live_groups = ()
            finalized_live = False
            if live_source and request.is_last_prefill:
                ready_at_finalize = (
                    LMCacheAscendConnectorV1Impl._query_source_ready_event(
                        source_ready_event
                    )
                    if source_ready_event is not None
                    and npu_content_diagnostics_enabled()
                    else None
                )
                finalized_live = self.lmcache_engine.finalize_live_source_descriptor(
                    request.req_id,
                    len(live_token_ids),
                    get_tensor_model_parallel_rank(),
                    int(getattr(parallel, "data_parallel_index", 0) or 0),
                )
                if finalized_live:
                    finalized = getattr(
                        self, "_finalized_live_source_submissions", None
                    )
                    if finalized is None:
                        finalized = set()
                        self._finalized_live_source_submissions = finalized
                    finalized.add(request.req_id)
                if finalized_live and source_ready_event is not None:
                    fences = getattr(self, "_live_source_ready_fences", None)
                    if fences is None:
                        fences = {}
                        self._live_source_ready_fences = fences
                    fences[request.req_id] = _LiveSourceReadyFence(
                        event=source_ready_event,
                        event_source=source_ready_event_source,
                        ready_at_finalize=ready_at_finalize,
                    )
                elif finalized_live:
                    raise RuntimeError(
                        "Final live Group-1 descriptor has no producer NPU event"
                    )
            direct_store = self.lmcache_engine.direct_prefill_store_enabled()
            preferred_segment = (
                request.request_configs.get(_MOONCAKE_PREFERRED_SEGMENT_CONFIG)
                if finalized_live
                and 0 in selected
                and isinstance(request.request_configs, dict)
                else None
            )
            # A decoder-preferred group-0 object is the decoder's only source
            # when the live transport carries group 1 alone.  Fence that final
            # persistent write before the live descriptor can leave the worker;
            # otherwise the decoder can race the asynchronous Mooncake Put.
            # Preserve the existing unfenced path for unhinted requests and for
            # true group-0 live transfer, where persistence is not on TTFT's
            # critical correctness path.
            fence_preferred_group0 = bool(
                preferred_segment and 0 not in live_groups
            )
            if direct_store and selected:
                direct_final = request.is_last_prefill and (
                    not finalized_live or fence_preferred_group0
                )
                if remote_fill_request:
                    # save_kv_layer can observe the last scheduler window before
                    # the request-matched forward-context handoff is adopted.
                    # Only _finish_save_batch owns that final causal boundary.
                    direct_final = direct_final and finish_batch
                self.lmcache_engine.store_direct_prefill(
                    request.req_id,
                    request.token_ids,
                    selected,
                    slot_mappings,
                    request.request_configs,
                    final=direct_final,
                    slot_mapping_base=mapping_base,
                    verified_prefix_end=verified_prefix_end,
                    # ReqMeta.token_ids is the scheduler-accepted store range
                    # after chunked-prefill and final-token policy.
                    accepted_store_end=len(request.token_ids),
                    source_ready_event=source_ready_event,
                    source_ready_event_source=source_ready_event_source,
                    source_ready_events=source_ready_events,
                )
            if (
                finalized_live
                and direct_store
                and selected
                and not fence_preferred_group0
            ):
                self._unfenced_live_stores[request.req_id] = request

    def build_connector_worker_meta(self) -> Optional[KVConnectorWorkerMetadata]:
        if self.lmcache_engine is None:
            return None
        self._fence_live_source_descriptors()
        descriptors = self.lmcache_engine.drain_live_source_descriptors()
        remote_fill_results = (
            self.lmcache_engine.drain_remote_fill_terminal_results()
        )
        for req_id, descriptor in descriptors.items():
            cold_start_perf_log(
                logger,
                "live_source_worker_emit",
                req_id=req_id,
                tp_rank=descriptor.get("tp_rank"),
                dp_rank=descriptor.get("dp_rank"),
                segments=len(descriptor.get("segments", ())),
                compact_layers=len(
                    descriptor.get("compact_layout", {}).get("layers", ())
                ),
                compact_runs=len(
                    descriptor.get("compact_layout", {}).get("runs", ())
                ),
                latent_layers=len(
                    descriptor.get("latent_layout", {}).get("layers", ())
                ),
                latent_pages=len(
                    descriptor.get("latent_layout", {}).get("pages", ())
                ),
                group_byte_totals=descriptor.get("group_byte_totals"),
            )
        return (
            LiveSourceWorkerMetadata(
                {req_id: [descriptor] for req_id, descriptor in descriptors.items()},
                remote_fill_results,
            )
            if descriptors or remote_fill_results
            else None
        )

    def update_connector_output(self, connector_output: Any) -> None:
        super().update_connector_output(connector_output)

    def update_connector_worker_metadata(
        self, metadata: KVConnectorWorkerMetadata, active_req_ids: set[str]
    ) -> None:
        if isinstance(metadata, LiveSourceWorkerMetadata):
            for req_id, descriptors in metadata.descriptors.items():
                active = req_id in active_req_ids
                tracked = req_id in self._unfinished_requests
                cold_start_perf_log(
                    logger,
                    "live_source_scheduler_ingest",
                    req_id=req_id,
                    active=active,
                    tracked=tracked,
                    descriptor_count=len(descriptors),
                    ranks=[
                        [item.get("tp_rank"), item.get("dp_rank")]
                        for item in descriptors
                    ],
                )
                # The final prefiller token may set the request status to
                # finished before this worker metadata is consumed.  The
                # scheduler still calls request_finished() later in the same
                # output-processing pass, and _unfinished_requests remains the
                # authoritative LMCache lifetime fence until then. Dropping a
                # descriptor solely because active is false loses the live
                # split offer; supported topologies then fall back to the
                # ordinary persistent-transfer path.
                if not tracked:
                    continue
                self._scheduler_live_sources[req_id] = list(descriptors)
            for req_id, result in metadata.remote_fill_results.items():
                if req_id in self._unfinished_requests:
                    results = getattr(
                        self, "_scheduler_remote_fill_results", None
                    )
                    if results is None:
                        results = {}
                        self._scheduler_remote_fill_results = results
                    results[req_id] = dict(result)

    def update_state_after_alloc(
        self, request: "Request", num_external_tokens: int, blocks: Any = None
    ) -> None:
        if request.request_id not in self._unfinished_requests:
            self._scheduler_live_sources.pop(request.request_id, None)
            getattr(self, "_scheduler_remote_fill_results", {}).pop(
                request.request_id, None
            )
        super().update_state_after_alloc(request, num_external_tokens, blocks)

    def _effective_skip_leading_tokens(
        self,
        _request: ReqMeta,
        save_spec: Any,
    ) -> int:
        # Ascend chunked prefill must not use producer transfer progress here.
        return save_spec.skip_leading_tokens

    def _prepare_direct_store_inputs(
        self,
        _request: ReqMeta,
        slot_mapping: torch.Tensor,
        save_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self.lmcache_engine is not None
        ordering_event = save_context.get("ordering_event")
        if ordering_event is None:
            ordering_event = torch.npu.Event()
            ordering_event.record()
            save_context["ordering_event"] = ordering_event

        if slot_mapping.device.type == "npu":
            slot_mapping_npu = slot_mapping.to(dtype=torch.long)
        else:
            slot_mapping = slot_mapping.pin_memory()
            with torch.npu.stream(
                self.lmcache_engine.gpu_connector.store_stream
            ):
                slot_mapping_npu = slot_mapping.to(
                    device="npu",
                    dtype=torch.long,
                    non_blocking=True,
                )
        return slot_mapping, {
            "ordering_event": ordering_event,
            "slot_mapping_npu": slot_mapping_npu,
        }

    def _finish_save_batch(self, _save_context: dict[str, Any]) -> None:
        # Preserve the final attention producer dependency before resetting
        # per-step bookkeeping.  Prefix-hit/cold-resume steps can reach this
        # deferred path when not every registered cache callback fires.  The
        # previous reset silently discarded reshape_cache_event and caused a
        # live descriptor to be fenced by an unrelated current-stream event.
        source_ready_event = getattr(
            self, "_latest_live_source_ready_event", None
        )
        source_ready_event_source = getattr(
            self, "_latest_live_source_ready_event_source", "missing"
        )
        direct_ready_events = dict(
            getattr(self, "_latest_direct_source_ready_events", {})
        )
        observed_direct_layers = set(self._direct_store_observed_layers)
        expected_direct_layers = set(self._latent_layer_names)
        if self.config.dsa_two_groups:
            expected_direct_layers.update(self._indexer_layer_names)
        source_ready_events: tuple[Any, ...] = ()
        if expected_direct_layers and expected_direct_layers.issubset(
            direct_ready_events
        ):
            unique_events: dict[int, Any] = {}
            for producer_event in direct_ready_events.values():
                unique_events.setdefault(id(producer_event), producer_event)
            source_ready_events = tuple(unique_events.values())
        self._direct_store_step_supported = None
        self._direct_store_observed_layers.clear()
        self._latest_live_source_ready_event = None
        self._latest_live_source_ready_event_source = "missing"
        getattr(self, "_latest_direct_source_ready_events", {}).clear()
        if self.kv_role != "kv_consumer" and self.lmcache_engine is not None:
            try:
                self.lmcache_engine.wait_for_pending_sync_stores()
            finally:
                completed = self._completed_layerwise_stores
                self._completed_layerwise_stores = {}
            requests = self._direct_prefill_requests() or []
            metadata = self._parent._get_connector_metadata()
            handoff = getattr(metadata, "_live_source_event_handoff", None)
            if hasattr(metadata, "_live_source_event_handoff"):
                delattr(metadata, "_live_source_event_handoff")
            expected_targets = self._producer_fence_handoff_targets(requests)
            expected_ids = {req_id for req_id, _ in expected_targets}
            handoff_status = "absent"
            if (
                expected_targets
                and isinstance(handoff, tuple)
                and len(handoff) == 2
                and handoff[0] == expected_targets
                and handoff[1] is not None
            ):
                _, source_ready_event = handoff
                source_ready_event_source = (
                    "forward_context.sfa_reshape_cache_event"
                )
                # The request/frontier-matched event causally joins preceding
                # cache writes on this forward stream, including callbacks.
                source_ready_events = (source_ready_event,)
                handoff_status = "adopted"
            elif not expected_targets:
                handoff_status = "not_armed"
            elif handoff is not None and not (
                isinstance(handoff, tuple) and len(handoff) == 2
            ):
                handoff_status = "malformed"
            elif isinstance(handoff, tuple) and handoff[0] != expected_targets:
                handoff_status = "target_mismatch"
            elif isinstance(handoff, tuple) and handoff[1] is None:
                handoff_status = "missing_event"
            for request in requests:
                remote_fill_eligible = bool(
                    getattr(request, "_lmcache_remote_fill_qualified", False)
                )
                if request.req_id not in expected_ids and not remote_fill_eligible:
                    continue
                cold_start_perf_log(
                    logger,
                    "remote_fill_producer_fence_decision",
                    req_id=request.req_id,
                    handoff_status=handoff_status,
                    expected_layer_count=len(expected_direct_layers),
                    observed_layer_count=len(
                        expected_direct_layers & observed_direct_layers
                    ),
                    event_layer_count=len(
                        expected_direct_layers & direct_ready_events.keys()
                    ),
                    callback_fence_complete=bool(
                        expected_direct_layers
                        and expected_direct_layers.issubset(direct_ready_events)
                    ),
                    complete_fence_count=len(source_ready_events),
                    source_event_present=source_ready_event is not None,
                    event_source=source_ready_event_source,
                    accepted_store_end=len(request.token_ids),
                    submission_mode=getattr(
                        self.config,
                        "remote_fill_submission_mode",
                        "final_deferred",
                    ),
                    remote_fill_eligible=remote_fill_eligible,
                )
            request_ids = {request.req_id for request in requests}
            adopted_requests = set()
            for (req_id, _), result in completed.items():
                if req_id in request_ids:
                    self.lmcache_engine.adopt_completed_layerwise_store(result)
                    adopted_requests.add(req_id)
            if requests:
                # Reuse completed layerwise progress before fencing a window
                # whose callbacks did not cover every registered cache.
                self._submit_direct_prefill_requests(
                    requests,
                    adopted_requests,
                    finish_batch=True,
                    source_ready_event=source_ready_event,
                    source_ready_event_source=source_ready_event_source,
                    source_ready_events=source_ready_events,
                )
            final_ids = {
                request.req_id for request in requests if request.is_last_prefill
            }
            if final_ids:
                fenced_ids = final_ids - self._unfenced_live_stores.keys()
                if fenced_ids:
                    self.lmcache_engine.wait_for_direct_stores(fenced_ids)
                for request in requests:
                    if (
                        request.req_id not in final_ids
                        or request.req_id in self._unfenced_live_stores
                    ):
                        continue
                    for group, committed_end in (
                        self.lmcache_engine.direct_store_committed_ends(
                            request.req_id
                        ).items()
                    ):
                        self._record_prefill_save_group_completed(
                            request,
                            group,
                            LayerwiseStoreResult(
                                request_id=request.req_id,
                                kv_group=group,
                                committed_end=committed_end,
                            ),
                        )
                    self._mark_prefill_committed(request)

    def _handle_save_request_error(
        self,
        request: ReqMeta,
        _error: Exception,
    ) -> bool:
        logger.exception(
            "wait_for_save failed for request %s; skipping save",
            request.req_id,
        )
        return True

    def _finalize_worker_requests_after_store(
        self, finished_req_ids: set[str]
    ) -> set[str]:
        """Release worker state once its store pipeline no longer owns it."""
        if self.lmcache_engine is None:
            return super()._finalize_worker_requests_after_store(
                finished_req_ids
            )
        live_finished = finished_req_ids & self._unfenced_live_stores.keys()
        failed_sending: set[str] = set()
        if live_finished:
            requests = [self._unfenced_live_stores[req_id] for req_id in live_finished]
            try:
                for request in requests:
                    try:
                        selected, slot_mappings, mapping_base = (
                            self._direct_request_inputs(
                                request, self._direct_group_caches()
                            )
                        )
                        self.lmcache_engine.store_direct_prefill(
                            request.req_id,
                            request.token_ids,
                            selected,
                            slot_mappings,
                            request.request_configs,
                            final=True,
                            slot_mapping_base=mapping_base,
                            accepted_store_end=len(request.token_ids),
                        )
                    except Exception as error:
                        self._handle_save_request_error(request, error)
                        try:
                            self.lmcache_engine.wait_for_pending_stores(
                                (request.req_id,)
                            )
                        except Exception:
                            logger.exception(
                                "Failed to drain rejected persistent store for %s",
                                request.req_id,
                            )
                            continue
                        self.lmcache_engine.drop_direct_store_states((request.req_id,))
                        failed_sending.add(request.req_id)
                        continue
                    for group, committed_end in (
                        self.lmcache_engine.direct_store_committed_ends(
                            request.req_id
                        ).items()
                    ):
                        self._record_prefill_save_group_completed(
                            request,
                            group,
                            LayerwiseStoreResult(
                                request_id=request.req_id,
                                kv_group=group,
                                committed_end=committed_end,
                            ),
                        )
                    self._mark_prefill_committed(request)
            finally:
                for req_id in live_finished:
                    self._unfenced_live_stores.pop(req_id, None)
        finished_sending = set(
            self.lmcache_engine.get_finished_stores(finished_req_ids) or ()
        )
        finished_sending.update(failed_sending)
        releasable_req_ids = (
            finished_sending if self.store_async else finished_req_ids
        )
        if releasable_req_ids:
            self._release_finished_worker_requests(releasable_req_ids)
            self.lmcache_engine.drop_direct_store_states(releasable_req_ids)
            finalized = getattr(
                self, "_finalized_live_source_submissions", None
            )
            if finalized is not None:
                finalized.difference_update(releasable_req_ids)
        for req_id in finished_req_ids - releasable_req_ids:
            # Retrieval pins are request-owned, not async-store-owned. Storage
            # submissions retain their own sources until their futures finish.
            self._drop_worker_retrieve_state(req_id)
        return finished_sending

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.lmcache_engine is None:
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )

        # Lookup pins are request-scoped; release via _drop_worker_retrieve_state.
        for req_id in preempted_req_ids:
            self._drop_worker_retrieve_state(req_id)

        if not self.store_async or self.kv_role == "kv_consumer":
            return

        waited_req_ids = self.lmcache_engine.wait_for_pending_stores(
            preempted_req_ids
        )
        self.lmcache_engine.wait_for_direct_stores(preempted_req_ids)
        for req_id in preempted_req_ids:
            getattr(self, "_scheduler_remote_fill_results", {}).pop(
                req_id, None
            )
            self._unfenced_live_stores.pop(req_id, None)
            fences = getattr(self, "_live_source_ready_fences", None)
            if fences is not None:
                fences.pop(req_id, None)
            finalized = getattr(
                self, "_finalized_live_source_submissions", None
            )
            if finalized is not None:
                finalized.discard(req_id)
        self.lmcache_engine.drop_direct_store_states(preempted_req_ids)
        if waited_req_ids:
            logger.info(
                "Handled preemptions after draining async stores: req_ids=%s",
                sorted(waited_req_ids),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        _, return_params = super().request_finished(request, block_ids)
        descriptors = self._scheduler_live_sources.pop(request.request_id, None)
        remote_fill_result = getattr(
            self, "_scheduler_remote_fill_results", {}
        ).pop(request.request_id, None)
        params = getattr(request, "kv_transfer_params", None)
        if remote_fill_result is not None and isinstance(params, dict):
            handoff = params.get("lmcache.remote_fill")
            if (
                isinstance(handoff, dict)
                and handoff.get("transfer_id")
                == remote_fill_result.get("transfer_id")
            ):
                handoff["terminal"] = dict(remote_fill_result)
                return_params = _remote_fill_response_params(
                    params, remote_fill_result, return_params
                )
                cold_start_perf_log(
                    logger,
                    "remote_fill_scheduler_attach",
                    req_id=request.request_id,
                    outcome=remote_fill_result.get("outcome"),
                    persistent_common_end=remote_fill_result.get(
                        "persistent_common_end"
                    ),
                    required_store_end=remote_fill_result.get(
                        "required_store_end"
                    ),
                )
            else:
                logger.warning(
                    "Remote-fill result identity mismatch for request %s; "
                    "leaving direct destination unpublished",
                    request.request_id,
                )
        if descriptors:
            parallel = self._vllm_config.parallel_config
            tp_size = int(parallel.tensor_parallel_size)
            dp_rank = int(
                getattr(parallel, "data_parallel_index", None) or 0
            )
            expected = {
                (tp_rank, dp_rank) for tp_rank in range(tp_size)
            }
            actual = {
                (int(item["tp_rank"]), int(item["dp_rank"]))
                for item in descriptors
            }
            if actual != expected or len(descriptors) != len(expected):
                logger.warning(
                    "Incomplete live source ranks for %s: expected=%s actual=%s; "
                    "using persistent fallback",
                    request.request_id,
                    sorted(expected),
                    sorted(actual),
                )
                descriptors = None
        if (
            descriptors
            and isinstance(params, dict)
            and params.get("request_live_split", False)
        ):
            # MultiConnector children share this request. Publish the source
            # here so the following Mooncake child can canonicalize it even
            # when the generic vLLM MultiConnector is selected.
            params["ascend_live_split_source_v1"] = {
                "descriptors": descriptors
            }
            cold_start_perf_log(
                logger,
                "live_source_attach",
                req_id=request.request_id,
                attached=True,
                descriptor_count=len(descriptors),
            )
        elif isinstance(params, dict) and params.get("request_live_split", False):
            cold_start_perf_log(
                logger,
                "live_source_attach",
                req_id=request.request_id,
                attached=False,
                descriptor_count=len(descriptors or ()),
                reason=(
                    "missing_or_incomplete_descriptor"
                    if not descriptors
                    else "request_not_eligible"
                ),
            )
        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
