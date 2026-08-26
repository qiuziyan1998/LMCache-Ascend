# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
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
        super().__init__(vllm_config=vllm_config, role=role)

    def capture_live_source_event_handoff(self, forward_context: Any) -> bool:
        """Forward an armed post-forward producer event to the implementation."""

        return bool(
            self._lmcache_engine.capture_live_source_event_handoff(
                forward_context
            )
        )

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
