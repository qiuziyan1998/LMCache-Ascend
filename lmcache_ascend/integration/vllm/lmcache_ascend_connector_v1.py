# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, TYPE_CHECKING, Optional

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.kv_cache_interface import KVCacheConfig

# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    supports_dsa_index_lmcache = True

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        impl = self._lmcache_engine
        if impl._layerwise_prefill_dma:
            from vllm.v1.core.dsa_shared_pool import (
                DSASharedBlockLayout, layerwise_prefill_bundle_multiplier,
            )
            from lmcache_ascend.v1.npu_connector.layerwise_dma import (
                build_group_cycles,
                cache_page_size_bytes,
            )

            latent = impl._kvcaches_for_group(0)[0]
            indexer = impl._kvcaches_for_group(1)[0]
            # Match the scheduler's layout construction, including its slab split.
            layout = DSASharedBlockLayout(
                latent_page_size_bytes=cache_page_size_bytes(latent),
                indexer_page_size_bytes=cache_page_size_bytes(indexer),
                capacity_bundles=1,
                bundle_multiplier=layerwise_prefill_bundle_multiplier(),
            )
            impl.lmcache_engine.gpu_connector.prefill_dma_cycles = build_group_cycles(
                latent, indexer,
                impl._lmcache_chunk_size,
                layout.bundle_multiplier,
                (layout.k_nope_dim, layout.k_pe_dim),
            )

    @property
    def uses_layerwise_model_callbacks(self) -> bool:
        """Whether model-layer Python callbacks are part of this execution."""
        return bool(getattr(self._lmcache_engine, "use_layerwise", False))

    @property
    def supports_staged_sfa_sparse_load(self) -> bool:
        """Advertise the exact staged-SFA selective-load contract."""
        engine = self._lmcache_engine
        config = getattr(engine, "config", None)
        return bool(
            getattr(engine, "use_layerwise", False)
            and getattr(engine, "kv_role", None) in ("kv_both", "kv_consumer")
            and getattr(config, "dsa_two_groups", False)
            and getattr(config, "enable_sparse_attention", False)
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        parallel = getattr(vllm_config, "parallel_config", None)
        if transfer is not None and parallel is not None:
            extra = dict(getattr(transfer, "kv_connector_extra_config", None) or {})
            dp_rank = getattr(parallel, "data_parallel_index", None)
            if dp_rank is None:
                dp_rank = getattr(parallel, "data_parallel_rank_local", 0)
            extra["lmcache_remote_fill_destination_dp_rank"] = int(dp_rank or 0)
            extra["lmcache_remote_fill_destination_dp_size"] = int(
                getattr(parallel, "data_parallel_size", 1) or 1
            )
            transfer.kv_connector_extra_config = extra
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

    def capture_live_source_event_handoff(self, forward_context: Any) -> bool:
        """Forward an armed post-forward producer event to the implementation."""

        return bool(
            self._lmcache_engine.capture_live_source_event_handoff(
                forward_context
            )
        )

    def seal_sparse_destination_layout(self) -> None:
        """Forward the final staged-capture storage contract when supported."""
        seal = getattr(self._lmcache_engine, "seal_sparse_destination_layout", None)
        if callable(seal):
            seal()

    def get_remote_fill_placement_info(
        self,
    ) -> dict[str, int | str | bool] | None:
        """Return the decoder's pointer-free remote-fill placement."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        discover = getattr(engine, "get_remote_fill_placement_info", None)
        return discover() if callable(discover) else None

    def get_remote_fill_metrics(self) -> dict[str, int] | None:
        """Return fixed-cardinality decoder protocol metrics when active."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        snapshot = getattr(engine, "get_remote_fill_metrics", None)
        return snapshot() if callable(snapshot) else None

    def remote_fill_requires_paired_restart(self) -> bool:
        """Expose an armed-transfer fatal latch to the worker supervisor."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        check = getattr(engine, "remote_fill_requires_paired_restart", None)
        return bool(check()) if callable(check) else False
