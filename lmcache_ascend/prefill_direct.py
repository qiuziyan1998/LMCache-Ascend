# SPDX-License-Identifier: Apache-2.0
"""Sender-only direct prefix restoration, selected once at construction.

Disabled P and every D instance keep the baseline classes and serving methods.
The specialized adapter falls back to baseline for mixed/sparse batches.
"""

# Standard
from typing import Any, Generator, Optional, TYPE_CHECKING, Union
from collections import defaultdict
import traceback

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cache_engine import _RemoteFillChunkLookupPlan, _RemoteFillLookupPlan
from lmcache.v1.memory_management import MemoryObj
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata, ReqMeta
from lmcache.v1.mooncake_layout import mooncake_valid_tokens
from lmcache.v1.serving_perf import (
    serving_perf_enabled,
    serving_perf_now,
    serving_perf_log,
)
from lmcache.v1.storage_backend.remote_backend import RemoteExternalPageReader
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.remote_fill_producer import RemoteFillFatalError

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


class PrefillDirectConnector(LMCacheAscendConnectorV1Impl):
    """Use direct G0 and shared-CPU G1 only for ordinary sender prefix loads."""

    def register_kv_caches(self, kv_caches: dict) -> None:
        super().register_kv_caches(kv_caches)
        self.lmcache_engine.preflight_prefill_group0_direct_hbm(
            self._kvcaches_for_group(0)
        )

    def _start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load ordinary P prefixes; preserve baseline behavior for other batches."""
        metadata = self._parent._get_connector_metadata()
        if (
            not self.use_layerwise
            or self.enable_blending
            or getattr(metadata, "dsa_cold_compact_load_pending", False)
        ):
            return super()._start_load_kv(forward_context, **kwargs)
        self.current_layer = 0
        self._wait_for_save_done = False
        attn_metadata = forward_context.attn_metadata
        assert isinstance(metadata, LMCacheConnectorMetadata)
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return
        if len(self.kv_caches) == 0:
            raise RuntimeError(
                "prefill_group0_direct_hbm requires register_kv_caches preflight"
            )
        active_req_ids: set[str] = set()
        resumed_req_ids: set[str] = set()
        loadable_requests: list[tuple[int, ReqMeta]] = []
        vllm_hit_tokens = 0
        prompt_tokens = 0
        has_load_spec = False
        for idx, request in enumerate(metadata.requests):
            # Decide before pruning state, priming loaders or submitting work.
            if request.is_sparse_decode or request.sparse_warm_ref:
                return super()._start_load_kv(forward_context, **kwargs)
            if not self._is_decode_window_save_request(request):
                active_req_ids.add(request.req_id)
            if request.resumed_from_preemption:
                resumed_req_ids.add(request.req_id)
            load_spec = request.load_spec
            if load_spec is None:
                continue
            if getattr(load_spec, "dsa_cold_compact_load", False):
                continue
            if not load_spec.can_load and getattr(
                load_spec, "dsa_cold_compact_resume", False
            ):
                raise RuntimeError(
                    "Cold compact resume requires a prepared worker load: "
                    f"req_id={request.req_id}"
                )
            has_load_spec = True
            vllm_hit_tokens += load_spec.vllm_cached_tokens
            prompt_tokens += request.retrieve_token_count()
            if not load_spec.can_load:
                continue
            loadable_requests.append((idx, request))
        staged_load_count = len(loadable_requests)
        self._prune_worker_retrieve_state(active_req_ids, resumed_req_ids)
        assert len(self.kv_caches) > 0
        if not self._kvcaches_list:
            self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_list
        assert self.lmcache_engine is not None
        self._drain_layerwise_retrievers()
        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        if (
            staged_load_count
            and gpu_connector is not None
            and hasattr(gpu_connector, "set_layerwise_staging_concurrency")
        ):
            gpu_connector.set_layerwise_staging_concurrency(
                max(2, staged_load_count + 1)
            )
        if has_load_spec:
            self._stats_monitor.update_interval_vllm_hit_tokens(vllm_hit_tokens)
            self._stats_monitor.update_interval_prompt_tokens(prompt_tokens)
        for load_idx, (idx, request) in enumerate(loadable_requests):
            request_perf_started = serving_perf_now() if serving_perf_enabled() else 0.0
            tokens = request.token_ids
            assert request.load_spec is not None
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            assert request.slot_mapping
            slot_mapping = request.slot_mapping[0]
            assert len(tokens) == len(slot_mapping)
            retrieve_tokens = self._load_tokens_for_retrieve(
                tokens, lmcache_cached_tokens, is_sparse_decode=False
            )
            recalc_last_applied = self._full_hit_recalc_last_token(
                request.load_spec,
                request.retrieve_token_count(),
                is_sparse_decode=False,
            )
            if recalc_last_applied:
                retrieve_tokens, slot_mapping = self._trim_prefill_for_recalc_last(
                    request, retrieve_tokens, slot_mapping
                )
            token_count = len(retrieve_tokens)
            token_mask = self._load_token_mask_for_retrieve(
                request, token_count, self._lmcache_chunk_size
            )
            indexer_token_mask = token_mask
            if token_count > len(slot_mapping):
                logger.warning(
                    "Request %s: retrieve_len=%d exceeds slot_mapping len=%d "
                    "(KV scatter will be incomplete -> garbage). "
                    "Often chunked-prefill metadata out of sync with lookup_hit.",
                    request.req_id,
                    token_count,
                    len(slot_mapping),
                )
            sync = load_idx == len(loadable_requests) - 1
            if request_perf_started:
                self._cold_perf_dense_load_completed.pop(request.req_id, None)
                self._cold_perf_dense_load_started[request.req_id] = (
                    request_perf_started,
                    token_count,
                )
            retrieve_slot_mapping = slot_mapping
            if lmcache_cached_tokens < len(slot_mapping):
                retrieve_slot_mapping = slot_mapping[:lmcache_cached_tokens]
            self._drop_worker_retrieve_state(request.req_id, release_lookup_pins=False)
            dsa_two_groups = self._is_dsa_two_groups()
            dense_preflight_state = {}
            layerwise_retriever = self.lmcache_engine.retrieve_prefill_group0_direct(
                retrieve_tokens,
                token_mask,
                kvcaches=kvcaches,
                slot_mapping=retrieve_slot_mapping,
                req_id=request.req_id,
                request_configs=request.request_configs,
                shared_cpu_request_preflight_state=dense_preflight_state,
            )
            self.layerwise_retrievers.append((layerwise_retriever, None))
            self._layerwise_requests.append(request)
            self._layerwise_retriever_is_sparse.append(False)
            self._layerwise_sparse_shared_ordered.append(False)
            indexer_retriever = None
            indexer_kvcaches = []
            idx_slot = None
            if dsa_two_groups:
                indexer_kvcaches = self._kvcaches_for_group(1)
                if not indexer_kvcaches:
                    raise RuntimeError(
                        "Dense prefix retrieval with dsa_two_groups=true "
                        "requires DSA index kvcaches for kv_group=1."
                    )
            if dsa_two_groups and indexer_kvcaches:
                indexer_layer_name = (
                    self._indexer_layer_names[0] if self._indexer_layer_names else None
                )
                if request.indexer_slot_mapping:
                    idx_slot = request.indexer_slot_mapping[0].to(
                        device=self.device, dtype=torch.long
                    )
                    if lmcache_cached_tokens < len(idx_slot):
                        idx_slot = idx_slot[:lmcache_cached_tokens]
                    if len(idx_slot) < lmcache_cached_tokens:
                        idx_slot = None
                if idx_slot is None:
                    idx_slot = self._indexer_retrieve_slot_mapping(
                        attn_metadata, lmcache_cached_tokens, indexer_layer_name
                    )
                if idx_slot is None:
                    raise RuntimeError(
                        "Dense prefix retrieval with dsa_two_groups=true "
                        "could not resolve the Group-1 index slot mapping. "
                        "Refusing to mix loaded Group-0 latent KV with stale "
                        "or uninitialized index rows."
                    )
                indexer_retriever = self.lmcache_engine.retrieve_layer(
                    retrieve_tokens,
                    indexer_token_mask,
                    kvcaches=indexer_kvcaches,
                    slot_mapping=idx_slot,
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    sync=sync,
                    kv_group=1,
                    req_id=request.req_id,
                    request_configs=request.request_configs,
                    shared_cpu_request_ordinal=idx,
                    shared_cpu_request_preflight_state=dense_preflight_state,
                    _retain_shared_dense_cache=False,
                    _retain_dense_sources_until_save=True,
                )
                self.layerwise_retrievers[-1] = (layerwise_retriever, indexer_retriever)
            self._prime_dense_prefix_retrievers(layerwise_retriever, indexer_retriever)
            continue


class PrefillDirectLMCacheEngine(AscendLMCacheEngine):
    """Keep sender index sources leased without a latent CPU sparse seed."""

    prefill_direct_active = True

    def post_init(self, **kwargs: Any) -> None:
        super().post_init(**kwargs)
        if not self._is_passive():
            return
        perf = serving_perf_enabled()
        started = serving_perf_now() if perf else None
        try:
            self._group1_external_page_reader = RemoteExternalPageReader(
                self.config, self.metadata
            )
        except BaseException:
            self._rollback_group1_direct_hbm_startup()
            raise
        if perf:
            serving_perf_log(
                logger,
                "group0_external_reader_init_complete",
                started=started,
                rank=self.metadata.worker_id,
            )

    def preflight_prefill_group0_direct_hbm(self, kvcaches: list) -> None:
        """Validate opted-in P destinations and readers uniformly across TP."""
        local_error: Optional[BaseException] = None
        try:
            if (
                self.config.pd_role != "sender"
                or self.metadata.first_rank != 0
                or kvcaches[0][0].device.type != "npu"
                or not callable(
                    getattr(self.gpu_connector, "record_dense_load_readiness", None)
                )
                or not callable(
                    getattr(
                        self.gpu_connector, "synchronize_dense_load_readiness", None
                    )
                )
            ):
                raise RuntimeError(
                    "Prefill Group-0 direct-HBM capabilities are unavailable"
                )
            check = getattr(
                self.gpu_connector,
                "direct_page_load_supported",
                None,
            )
            if not callable(check) or not check(kvcaches, 0):
                raise RuntimeError("Group-0 direct-HBM layout is unsupported")
            if not callable(self._group1_external_page_load()):
                raise RuntimeError("External page loader is unavailable")
        except BaseException as error:
            local_error = error
        ready = local_error is None
        if self.metadata.world_size > 1:
            try:
                collective = getattr(self, "collective_all_true_fn", None)
                if not callable(collective):
                    raise RuntimeError("Group-0 direct-HBM startup lacks TP consensus")
                ready = bool(collective(ready))
            except BaseException as error:
                local_error = local_error or error
                ready = False
        if not ready:
            detail = (
                f": {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else " on another TP rank"
            )
            failure = RuntimeError("Group-0 direct-HBM preflight failed" + detail)
            try:
                self._rollback_group1_direct_hbm_startup()
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"{failure}; startup rollback also failed"
                ) from rollback_error
            raise failure

    def retrieve_prefill_group0_direct(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor],
        *,
        kvcaches: list,
        slot_mapping: torch.Tensor,
        req_id: str,
        request_configs: Optional[dict],
        shared_cpu_request_preflight_state: dict[str, Any],
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Restore a sender's dense G0 prefix without LocalCPU payload pages.

        Tokens/mask and CPU slots follow the ordinary dense-prefix contract.
        The preflight dictionary shares exact hashes with the G1 retriever.
        Yields L+1 placeholders and the final CPU mask for L cache layers.
        Raises before the first yield if any TP rank fails; unknown DMA keeps
        native destination ownership latched until the existing restart path.
        """
        perf_enabled = serving_perf_enabled()
        started = serving_perf_now() if perf_enabled else 0.0
        local_error: Optional[BaseException] = None
        metrics = None
        try:
            if not self.is_healthy():
                raise RuntimeError("Prefill Group-0 direct-HBM load is unavailable")
            ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
            plans = list(
                self._dense_retrieve_token_results(
                    tokens,
                    mask,
                    request_configs,
                    0,
                    {
                        "shared_cpu_request_preflight_state": (
                            shared_cpu_request_preflight_state
                        )
                    },
                )
            )
            previous_end = plans[0][0] if plans else len(tokens)
            for start, end, key in plans:
                if (
                    start != previous_end
                    or not 0 <= start < end <= len(tokens)
                    or key.kv_group != 0
                    or mooncake_valid_tokens(key, self.config.chunk_size) != end - start
                ):
                    raise ValueError("Prefill Group-0 page ranges are invalid")
                ret_mask[start:end] = True
                previous_end = end
            expected = (
                mask.to(device="cpu", dtype=torch.bool) if mask is not None else None
            )
            if previous_end != len(tokens) or (
                not torch.equal(ret_mask, expected)
                if expected is not None
                else not bool(torch.all(ret_mask))
            ):
                raise ValueError(
                    "Prefill Group-0 page plan does not cover the load mask"
                )
            if plans and not self._is_passive():
                proof = self._remote_fill_retrieve_plan(req_id, plans, 0)
                if proof is None or any(
                    item != ("RemoteBackend", True) for item in proof
                ):
                    raise ValueError(
                        "Prefill Group-0 load lacks its persistent prefix proof"
                    )
            plan_ms = (serving_perf_now() - started) * 1000 if perf_enabled else 0.0
            if plans:
                metrics = self._load_prefill_group0_page_plan(
                    plans,
                    slot_mapping,
                    kvcaches,
                    req_id,
                    perf_enabled=perf_enabled,
                )
        except BaseException as error:
            local_error = error
        decision_started = serving_perf_now() if perf_enabled else 0.0
        try:
            ready = local_error is None
            if self.metadata.world_size > 1:
                ready = bool(self.collective_all_true_fn(ready))
            if not ready:
                if local_error is not None:
                    raise local_error
                raise RuntimeError(
                    "Prefill Group-0 direct load failed on another TP rank"
                )
        finally:
            # A propagated exception must not retain itself through this frame
            # when cyclic GC is disabled.
            local_error = None
        if perf_enabled and (serving_perf_now() - started) * 1000 >= 100.0:
            layout_ms, destination_ms, read_ms, predecessor_ms, sizes = (
                metrics if metrics is not None else (0.0, 0.0, 0.0, 0.0, [])
            )
            serving_perf_log(
                logger,
                "prefill_group0_direct_load_slow",
                started=started,
                req_id=req_id,
                rank=self.metadata.worker_id,
                pages=len(plans),
                bytes=sum(map(sum, sizes)),
                plan_ms=round(plan_ms + layout_ms + destination_ms, 3),
                predecessor_wait_ms=round(predecessor_ms, 3),
                read_call_ms=round(read_ms, 3),
                tp_decision_ms=round((serving_perf_now() - decision_started) * 1000, 3),
            )
        for _ in range(self.num_layers + 1):
            yield None
        yield ret_mask

    def _lookup_persistent_direct_hbm_prefix(
        self,
        chunks: list[tuple[int, int, CacheEngineKey]],
        group_keys: dict[int, list[LayerCacheEngineKey]],
        *,
        search_range: list[str],
        lookup_id: Optional[str],
        pin: bool,
    ) -> int:
        """Prove a Mooncake pair prefix, then overlay role-appropriate CPU hits."""

        assert self.storage_manager is not None
        if "RemoteBackend" not in search_range:
            return 0

        local_mapping: dict[str, list[CacheEngineKey]] = defaultdict(list)
        pins_registered = False
        try:
            pair_count, persistent_mapping = (
                self.storage_manager.batched_contains_two_group_layer_pages(
                    group_keys[0],
                    group_keys[1],
                    ["RemoteBackend"],
                    False,
                )
            )
            if not 0 <= pair_count <= len(chunks):
                raise RuntimeError(
                    "Persistent two-group lookup returned an invalid prefix"
                )
            persistent_keys = persistent_mapping.get("RemoteBackend", [])
            if (
                set(persistent_mapping) - {"RemoteBackend"}
                or len(persistent_keys) != 2 * pair_count
            ):
                raise RuntimeError(
                    "Persistent two-group lookup returned invalid backend mapping"
                )
            if pair_count == 0:
                return 0

            local_counts = [0, 0]
            if "LocalCPUBackend" in search_range:
                # P loads both groups through shared CPU pages. D must keep
                # Group 1 on the persistent direct-to-HBM path.
                local_groups = (1,) if pin else ()
                for group in local_groups:
                    local_count, group_mapping = (
                        self.storage_manager.batched_contains_layer_pages(
                            group_keys[group][:pair_count],
                            ["LocalCPUBackend"],
                            pin,
                        )
                    )
                    # Retain each returned pin before validation so a later
                    # group failure also rolls back earlier successful pins.
                    for location, keys in group_mapping.items():
                        local_mapping[location].extend(keys)
                    local_keys = group_mapping.get("LocalCPUBackend", [])
                    if (
                        not 0 <= local_count <= pair_count
                        or set(group_mapping) - {"LocalCPUBackend"}
                        or len(local_keys) != local_count
                    ):
                        raise RuntimeError(
                            f"Group-{group} LocalCPU overlay returned invalid "
                            "prefix mapping"
                        )
                    local_counts[group] = local_count

            plan = tuple(
                _RemoteFillChunkLookupPlan(
                    start=start,
                    end=end,
                    chunk_hash=key.chunk_hash,
                    locations_by_group=(
                        "LocalCPUBackend"
                        if index < local_counts[0]
                        else "RemoteBackend",
                        "LocalCPUBackend"
                        if index < local_counts[1]
                        else "RemoteBackend",
                    ),
                    page_by_group=(True, True),
                )
                for index, (start, end, key) in enumerate(chunks[:pair_count])
            )
            if pin:
                assert lookup_id is not None
                for location, keys in local_mapping.items():
                    self.lookup_pins[lookup_id][location].extend(keys)
                pins_registered = True
                self._remote_fill_lookup_plans[lookup_id] = _RemoteFillLookupPlan(plan)
            return plan[-1].end
        except Exception:
            if pin:
                if pins_registered:
                    # The request registry now owns the pins; remove it before
                    # the caller's error cleanup can attempt another release.
                    assert lookup_id is not None
                    self._release_lookup_pins(lookup_id)
                else:
                    for location, keys in local_mapping.items():
                        if keys:
                            self.storage_manager.batched_unpin(keys, [location])
            raise

    def _release_retained_dense_retrieve_objs(
        self,
        memory_objs: list[MemoryObj],
        *,
        unpin: bool,
        kwargs: dict[str, Any],
    ) -> None:
        """Fence failed dense loads before releasing retained sources."""
        if memory_objs and (
            kwargs.get("_retain_shared_dense_cache")
            or kwargs.get("_retain_dense_sources_until_save")
        ):
            synchronize = getattr(
                self.gpu_connector, "synchronize_dense_load_stream", None
            )
            if not callable(synchronize):
                raise RuntimeError("Dense source cleanup requires load-stream sync")
            synchronize()
        self._release_shared_retrieve_objs(memory_objs, unpin=unpin)

    def _adopt_dense_shared_retrieve_cache(
        self,
        *,
        req_id: str,
        kv_group: int,
        memory_objs: list[list[MemoryObj]],
        kwargs: dict[str, Any],
        **metadata: Any,
    ) -> bool:
        if req_id and kwargs.get("_retain_dense_sources_until_save"):
            self.register_shared_cpu_sparse_request(
                req_id, owned_groups={kv_group: memory_objs}
            )
            return True
        return super()._adopt_dense_shared_retrieve_cache(
            req_id=req_id,
            kv_group=kv_group,
            memory_objs=memory_objs,
            kwargs=kwargs,
            **metadata,
        )

    def _load_prefill_group0_page_plan(
        self,
        plans: list[tuple[int, int, CacheEngineKey]],
        slot_mapping: torch.Tensor,
        kvcaches: list,
        req_id: str,
        *,
        perf_enabled: bool,
    ) -> Optional[tuple[float, float, float, float, list[list[int]]]]:
        """Read P Group 0 after its compute predecessor reaches readiness."""
        kv_group = 0
        phase_started = serving_perf_now() if perf_enabled else 0.0
        self._ensure_layerwise_connector_layout(kvcaches=kvcaches, kv_group=kv_group)
        layout_ms = (serving_perf_now() - phase_started) * 1000 if perf_enabled else 0.0
        planner = getattr(self.gpu_connector, "plan_direct_page_destinations", None)
        if not callable(planner):
            raise RuntimeError(f"Group-{kv_group} direct-HBM planner is unavailable")
        starts = [start for start, _, _ in plans]
        ends = [end for _, end, _ in plans]
        keys = [key for _, _, key in plans]
        phase_started = serving_perf_now() if perf_enabled else 0.0
        planned = planner(kvcaches, slot_mapping, starts, ends, kv_group)
        destination_ms = (
            (serving_perf_now() - phase_started) * 1000 if perf_enabled else 0.0
        )
        if planned is None:
            rejection = getattr(self.gpu_connector, "direct_page_plan_rejection", None)
            reason = rejection(kv_group) if callable(rejection) else None
            raise RuntimeError(
                f"Group-{kv_group} direct-HBM destination plan failed: "
                f"{reason or 'unsupported_layout'}"
            )
        ptrs, sizes, owners = planned
        if len(ptrs) != len(keys) or len(sizes) != len(keys):
            raise RuntimeError(f"Group-{kv_group} direct-HBM page count is invalid")
        load_pages = self._group1_external_page_load()
        phase_started = serving_perf_now() if perf_enabled else 0.0
        # External DMA is not ordered by an event recorded after the read.
        ready = self.gpu_connector.record_dense_load_readiness(
            stream=torch.npu.current_stream()
        )
        self.gpu_connector.synchronize_dense_load_readiness(ready)
        predecessor_ms = (
            (serving_perf_now() - phase_started) * 1000 if perf_enabled else 0.0
        )
        try:
            phase_started = serving_perf_now() if perf_enabled else 0.0
            load_pages(keys, ptrs, sizes, owners, req_id)
            if perf_enabled:
                native_ms = (serving_perf_now() - phase_started) * 1000
                return layout_ms, destination_ms, native_ms, predecessor_ms, sizes
            return None
        except NativeExternalPageTransferUnknownError as error:
            self._remote_fill_require_paired_restart((req_id,))
            raise RemoteFillFatalError(
                "prefiller Group-0 external-page DMA state is unknown"
            ) from error

        except Exception as error:
            # Only completed inner frames are cleared. Unknown native DMA above
            # retains its explicit Future/owners and requires paired restart.
            traceback.clear_frames(error.__traceback__)
            raise
