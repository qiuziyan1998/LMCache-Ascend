# SPDX-License-Identifier: Apache-2.0
# Third Party
from typing import Any, Optional

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
            and getattr(engine, "kv_role", None)
            in ("kv_both", "kv_consumer")
            and getattr(config, "dsa_two_groups", False)
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ) -> None:
        # DSA replay P1: forward the third constructor argument to the parent
        # Dynamic connector, which already implements the v0.23 3-arg
        # compatibility layer (passes kv_cache_config to KVConnectorBase_V1
        # when provided). The 2-arg override here previously shadowed the
        # parent's signature and tripped the v0.23 factory's
        # supports_kw(kv_cache_config) hard check.
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
