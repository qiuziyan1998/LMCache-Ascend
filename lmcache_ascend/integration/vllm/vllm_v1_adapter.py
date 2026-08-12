# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
)
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LayerwiseStoreResult
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LiveSourceWorkerMetadata(KVConnectorWorkerMetadata):
    descriptors: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "LiveSourceWorkerMetadata":
        if not isinstance(other, LiveSourceWorkerMetadata):
            raise TypeError("Cannot aggregate incompatible LMCache worker metadata")
        merged = {req_id: list(items) for req_id, items in self.descriptors.items()}
        for req_id, items in other.descriptors.items():
            merged.setdefault(req_id, []).extend(items)
        return LiveSourceWorkerMetadata(merged)


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
        self._direct_store_requested = bool(
            get_extra("mooncake_direct_npu_prefill_store", False)
            if callable(get_extra)
            else False
        )
        self._direct_store_observed_layers: set[str] = set()
        self._direct_store_step_supported: Optional[bool] = None
        self._completed_layerwise_stores: dict[
            tuple[str, int], LayerwiseStoreResult
        ] = {}
        self._scheduler_live_sources: dict[str, list[dict[str, Any]]] = {}
        if self._direct_store_requested:
            extra = self.config.extra_config or {}
            valid = (
                self.store_async
                and self.config.store_async_max_queue_size == 2
                and str(self.config.remote_url).startswith("mooncakestore://")
                and self.use_layerwise
                and self.config.enable_shared_cpu_cache
                and extra.get("mooncake_page_first_multi_buffer", False)
                and extra.get("mooncake_layer_merged_page_objects", False)
                and extra.get("save_only_first_rank", False)
                and extra.get("use_ascend_direct", False)
                and not extra.get("save_chunk_meta", False)
            )
            if not valid:
                raise ValueError(
                    "mooncake_direct_npu_prefill_store requires store_async=true, "
                    "store_async_max_queue_size=2, shared layerwise merged Mooncake "
                    "pages, save_only_first_rank, use_ascend_direct, and "
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
            self._direct_store_step_supported = None
            self._direct_store_observed_layers.clear()
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
                self.lmcache_engine.drop_direct_store_states(req_ids)

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
            if layer_name in self._direct_store_observed_layers:
                self._direct_store_observed_layers.clear()
            self._direct_store_observed_layers.add(layer_name)
        if not expected.issubset(self._direct_store_observed_layers):
            return

        self._submit_direct_prefill_requests(requests)
        self._direct_store_observed_layers.clear()

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
    ) -> None:
        """Submit direct pages; also used by the wait-for-save fallback."""
        assert self.lmcache_engine is not None
        group_caches = self._direct_group_caches()
        for request in requests:
            selected, slot_mappings, mapping_base = self._direct_request_inputs(
                request, group_caches
            )
            if not selected:
                continue
            load_spec = getattr(request, "load_spec", None)
            committed_prefix = getattr(load_spec, "dsa_committed_end", None)
            if committed_prefix is None:
                committed_prefix = getattr(load_spec, "lmcache_cached_tokens", 0)
            verified_prefix_end = min(
                mapping_base,
                int(committed_prefix or 0),
            )
            live_source = bool(getattr(request, "live_source_requested", False))
            if live_source:
                self.lmcache_engine.begin_live_source_descriptor(request.req_id)
            direct_store = self.lmcache_engine.direct_prefill_store_enabled()
            if direct_store:
                if (
                    live_source
                    and adopted_requests is not None
                    and request.req_id in adopted_requests
                ):
                    # The persistent write reused below has no missing direct
                    # pages from which to build a live source descriptor.
                    self.lmcache_engine.capture_live_source_step(
                        request.req_id,
                        request.token_ids,
                        selected,
                        slot_mappings,
                        request.request_configs,
                        mapping_base,
                        request.is_last_prefill,
                    )
                self.lmcache_engine.store_direct_prefill(
                    request.req_id,
                    request.token_ids,
                    selected,
                    slot_mappings,
                    request.request_configs,
                    final=request.is_last_prefill,
                    slot_mapping_base=mapping_base,
                    verified_prefix_end=verified_prefix_end,
                )
            elif live_source:
                self.lmcache_engine.capture_live_source_step(
                    request.req_id,
                    request.token_ids,
                    selected,
                    slot_mappings,
                    request.request_configs,
                    mapping_base,
                    request.is_last_prefill,
                )
            if live_source and request.is_last_prefill:
                parallel = self._vllm_config.parallel_config
                self.lmcache_engine.finalize_live_source_descriptor(
                    request.req_id,
                    len(request.token_ids),
                    get_tensor_model_parallel_rank(),
                    int(getattr(parallel, "data_parallel_rank_local", 0) or 0),
                )

    def build_connector_worker_meta(self) -> Optional[KVConnectorWorkerMetadata]:
        if self.lmcache_engine is None:
            return None
        descriptors = self.lmcache_engine.drain_live_source_descriptors()
        return (
            LiveSourceWorkerMetadata(
                {req_id: [descriptor] for req_id, descriptor in descriptors.items()}
            )
            if descriptors
            else None
        )

    def update_connector_output(self, connector_output: Any) -> None:
        super().update_connector_output(connector_output)
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(metadata, LiveSourceWorkerMetadata):
            for req_id, descriptors in metadata.descriptors.items():
                by_rank = {
                    (int(item["tp_rank"]), int(item["dp_rank"])): item
                    for item in self._scheduler_live_sources.get(req_id, ())
                }
                by_rank.update(
                    {
                        (int(item["tp_rank"]), int(item["dp_rank"])): item
                        for item in descriptors
                    }
                )
                self._scheduler_live_sources[req_id] = list(by_rank.values())

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
        self._direct_store_step_supported = None
        self._direct_store_observed_layers.clear()
        if self.kv_role != "kv_consumer" and self.lmcache_engine is not None:
            try:
                self.lmcache_engine.wait_for_pending_sync_stores()
            finally:
                completed = self._completed_layerwise_stores
                self._completed_layerwise_stores = {}
            requests = self._direct_prefill_requests() or []
            request_ids = {request.req_id for request in requests}
            adopted_requests = set()
            for (req_id, _), result in completed.items():
                if req_id in request_ids:
                    self.lmcache_engine.adopt_completed_layerwise_store(result)
                    adopted_requests.add(req_id)
            if requests:
                # Reuse completed layerwise progress before fencing a window
                # whose callbacks did not cover every registered cache.
                self._submit_direct_prefill_requests(requests, adopted_requests)
            final_ids = {
                request.req_id for request in requests if request.is_last_prefill
            }
            if final_ids:
                self.lmcache_engine.wait_for_direct_stores(final_ids)
                for request in requests:
                    if request.req_id not in final_ids:
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
        finished_sending = set(
            self.lmcache_engine.get_finished_stores(finished_req_ids) or ()
        )
        releasable_req_ids = (
            finished_sending if self.store_async else finished_req_ids
        )
        if releasable_req_ids:
            self._release_finished_worker_requests(releasable_req_ids)
            self.lmcache_engine.drop_direct_store_states(releasable_req_ids)
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
        params = getattr(request, "kv_transfer_params", None)
        if descriptors:
            parallel = self._vllm_config.parallel_config
            tp_size = int(parallel.tensor_parallel_size)
            dp_rank = int(
                getattr(parallel, "data_parallel_rank_local", None) or 0
            )
            expected = {
                (tp_rank, dp_rank) for tp_rank in range(tp_size)
            }
            actual = {
                (int(item["tp_rank"]), int(item["dp_rank"]))
                for item in descriptors
            }
            if actual != expected:
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
            return_params = return_params or {}
            return_params["ascend_live_split_source_v1"] = {
                "descriptors": descriptors
            }
        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
