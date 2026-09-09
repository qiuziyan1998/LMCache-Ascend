# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
import torch
import torch.distributed as dist

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
from lmcache.v1.gpu_connector.sparse import PreparedSparseGraphStep

# First Party
from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        self._sparse_graph_transfers: dict[
            tuple[Any, str], SparseGraphTransfer
        ] = {}
        self._sparse_graph_layer_names: dict[Any, list[str]] = {}
        self._sparse_graph_binding_signatures: dict[
            Any, tuple[tuple[str, ...], tuple[int, ...]]
        ] = {}
        self._active_sparse_graph_key: Any = None
        self._active_sparse_graph_lease: PreparedSparseGraphStep | None = None
        self._pending_sparse_graph_leases: list[
            tuple[Any, PreparedSparseGraphStep]
        ] = []
        self._last_sparse_graph_event: Any = None
        self._sparse_graph_readiness = torch.empty(
            1,
            dtype=torch.int32,
            device="cpu",
        )

    @property
    def supports_sparse_decode_graph_load(self) -> bool:
        """Whether this worker can capture selective latent loads in a graph."""
        return self.supports_staged_sfa_sparse_load

    def _release_completed_sparse_graph_leases(self) -> None:
        pending: list[tuple[Any, PreparedSparseGraphStep]] = []
        for event, lease in self._pending_sparse_graph_leases:
            query = getattr(event, "query", None)
            if callable(query) and query():
                lease.release()
            else:
                pending.append((event, lease))
        self._pending_sparse_graph_leases = pending

    def _collect_retired_sparse_graph_bindings(self) -> None:
        retired = self._lmcache_engine.take_retired_sparse_graph_steps()
        if not retired:
            return
        event = self._last_sparse_graph_event
        self._last_sparse_graph_event = None
        query = getattr(event, "query", None)
        if event is None or (callable(query) and query()):
            for lease in retired:
                lease.release()
            return
        self._pending_sparse_graph_leases.extend(
            (event, lease) for lease in retired
        )

    def _retire_sparse_graph_binding(self) -> None:
        self._lmcache_engine.retire_sparse_graph_binding()
        self._collect_retired_sparse_graph_bindings()

    def _synchronize_sparse_graph_readiness(self, ready: bool) -> bool:
        """Reach one graph/eager verdict across TP and internal-DP peers."""
        flag = self._sparse_graph_readiness
        flag.fill_(int(ready))
        world_group = get_world_group()
        if world_group.world_size > 1:
            dist.all_reduce(
                flag,
                op=dist.ReduceOp.MIN,
                group=world_group.cpu_group,
            )
        return bool(flag.item())

    def begin_sparse_decode_graph_step(self, forward_context: Any) -> bool:
        """Bind warm sources once before replay; return false for eager fallback."""
        self._release_completed_sparse_graph_leases()
        graph_key = getattr(forward_context, "staged_sfa_graph_key", None)
        if (
            graph_key is None
            or getattr(forward_context, "staged_sfa_graph_dummy_run", False)
            or not self.supports_sparse_decode_graph_load
        ):
            self._retire_sparse_graph_binding()
            return False
        target_layer_names = tuple(
            self._sparse_graph_layer_names.get(graph_key, ())
        )
        lease = None
        if target_layer_names:
            first_metadata = forward_context.attn_metadata[
                target_layer_names[0]
            ]
            request_ids = getattr(
                first_metadata, "decode_request_ids_compact", None
            )
            if request_ids:
                lease = self._lmcache_engine.prepare_sparse_graph_step(
                    tuple(request_ids),
                    target_layer_names,
                    int(graph_key.request_capacity),
                )
                self._collect_retired_sparse_graph_bindings()
        if not self._synchronize_sparse_graph_readiness(lease is not None):
            if lease is not None:
                self._lmcache_engine.finish_sparse_graph_step()
            self._retire_sparse_graph_binding()
            return False
        assert lease is not None
        try:
            binding_signature = (
                lease.request_ids,
                tuple(source.binding_id for source in lease.sources),
            )
            if (
                self._sparse_graph_binding_signatures.get(graph_key)
                != binding_signature
            ):
                for layer_id, layer_name in enumerate(target_layer_names):
                    self._sparse_graph_transfers[
                        (graph_key, layer_name)
                    ].bind_batch(
                        lease.sources,
                        layer_id,
                    )
                self._sparse_graph_binding_signatures[graph_key] = binding_signature
        except BaseException:
            self._lmcache_engine.finish_sparse_graph_step()
            lease.release()
            raise
        self._active_sparse_graph_key = graph_key
        self._active_sparse_graph_lease = lease
        return True

    def end_sparse_decode_graph_step(self, forward_context: Any) -> None:
        """Fence the replay and defer source unpinning without host sync."""
        lease = self._active_sparse_graph_lease
        if lease is None:
            return
        self._lmcache_engine.finish_sparse_graph_step()
        event = torch.npu.Event()
        event.record()
        self._last_sparse_graph_event = event
        self._active_sparse_graph_key = None
        self._active_sparse_graph_lease = None

    def sparse_decode_graph_load(
        self,
        graph_key: Any,
        layer_name: str,
        kv_caches: tuple[torch.Tensor, torch.Tensor],
        selected_token_ids: torch.Tensor,
        selected_token_counts: torch.Tensor,
        target_slot_mapping: torch.Tensor,
    ) -> None:
        """Capture the device-only load using stable per-key transfer state."""
        key = (graph_key, layer_name)
        transfer = self._sparse_graph_transfers.get(key)
        if transfer is None:
            transfer = SparseGraphTransfer(
                kv_caches,
                target_slot_mapping,
                int(self._lmcache_engine._lmcache_chunk_size),
                int(self._vllm_config.model_config.max_model_len),
                request_capacity=int(selected_token_ids.shape[0]),
            )
            self._sparse_graph_transfers[key] = transfer
            self._sparse_graph_layer_names.setdefault(graph_key, []).append(
                layer_name
            )
        transfer.load(
            selected_token_ids,
            selected_token_counts,
            target_slot_mapping,
        )

    def validate_sparse_decode_graph_capture(
        self,
        graph_keys: tuple[Any, ...],
        target_layer_names: tuple[str, ...],
    ) -> None:
        """Fail startup when any requested key/layer transfer was not captured."""
        expected = {
            (graph_key, layer_name)
            for graph_key in graph_keys
            for layer_name in target_layer_names
        }
        missing = expected.difference(self._sparse_graph_transfers)
        if missing:
            raise RuntimeError(
                "Sparse LMCache graph transfer capture is incomplete: "
                f"missing={tuple(missing)}"
            )

    def shutdown(self) -> None:
        """Synchronize once before releasing graph-owned host source pins."""
        synchronize_graph_sources = bool(
            self._active_sparse_graph_lease is not None
            or self._pending_sparse_graph_leases
            or self._last_sparse_graph_event is not None
        )
        if self._active_sparse_graph_lease is not None:
            self._lmcache_engine.finish_sparse_graph_step()
            self._active_sparse_graph_lease = None
        if synchronize_graph_sources:
            torch.npu.synchronize()
            for _, lease in self._pending_sparse_graph_leases:
                lease.release()
            self._pending_sparse_graph_leases.clear()
            self._last_sparse_graph_event = None
        self._lmcache_engine.release_sparse_graph_bindings()
        super().shutdown()
