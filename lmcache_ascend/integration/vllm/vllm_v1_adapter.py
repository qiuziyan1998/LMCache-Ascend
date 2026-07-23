# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
)
from lmcache.logging import init_logger
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        self.store_async = self.config.store_async
        if (
            role != KVConnectorRole.SCHEDULER
            and self.kv_role != "kv_consumer"
            and self.use_layerwise
            and self.store_async
        ):
            raise ValueError(
                "Layerwise storing is not supported with async store"
            )
        logger.debug("store_async: %s", self.store_async)

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
        if self.kv_role != "kv_consumer" and self.lmcache_engine is not None:
            self.lmcache_engine.wait_for_pending_sync_stores()

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
        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
