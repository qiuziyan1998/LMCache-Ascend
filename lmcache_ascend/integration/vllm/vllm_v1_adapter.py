# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import queue
import threading
import time
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
from vllm.v1.outputs import KVConnectorSaveCompletion
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AsyncDecodePersistenceError

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request

logger = init_logger(__name__)

ASYNC_DECODE_SAVE_MAX_RETRIES = 3
ASYNC_DECODE_SAVE_RETRY_DELAYS = (0.1, 0.5, 2.0)


@dataclass(slots=True)
class _AsyncDecodeSaveTask:
    request: ReqMeta
    ordering_event: Any
    attempt: int = 0
    dispatched: bool = False
    done: bool = False
    results: list[tuple[int, Any]] = field(default_factory=list)

    @property
    def owner(self) -> str:
        return (
            f"{self.request.req_id}:{self.request.decode_save_generation}:"
            f"{self.request.decode_save_job_id}"
        )


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
        self._async_decode_max_retries = max(
            0,
            int(
                os.environ.get(
                    "LMCACHE_ASYNC_DECODE_SAVE_MAX_RETRIES",
                    str(ASYNC_DECODE_SAVE_MAX_RETRIES),
                )
            ),
        )
        if (
            role != KVConnectorRole.SCHEDULER
            and self.kv_role != "kv_consumer"
            and self.use_layerwise
            and self.store_async
        ):
            raise ValueError(
                "Global store_async is not supported with layerwise storing; "
                "use LMCACHE_ASYNC_DECODE_SAVE for decode windows"
            )
        self._async_decode_worker_enabled = bool(
            role != KVConnectorRole.SCHEDULER
            and self.kv_role != "kv_consumer"
            and self._async_decode_save
        )
        if self._async_decode_worker_enabled:
            self._async_decode_device_id = torch.npu.current_device()
            self._async_decode_queue: queue.Queue[Optional[str]] = queue.Queue(
                maxsize=self._async_decode_save_max_pending_per_worker
            )
            self._async_decode_lock = threading.RLock()
            self._async_decode_cv = threading.Condition(
                self._async_decode_lock
            )
            self._async_decode_jobs: dict[
                str, deque[_AsyncDecodeSaveTask]
            ] = {}
            self._async_decode_ready_reqs: set[str] = set()
            self._async_decode_known_reqs: set[str] = set()
            self._async_decode_finished_waiting: set[str] = set()
            self._async_decode_finished_ready: set[str] = set()
            self._async_decode_error: Optional[BaseException] = None
            self._async_decode_completion_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="lmcache-decode-persist",
            )
            self._async_decode_dispatcher = threading.Thread(
                target=self._async_decode_dispatch_loop,
                daemon=True,
                name="lmcache-ascend-decode-store",
            )
            self._async_decode_dispatcher.start()
        logger.debug("store_async: %s", self.store_async)

    def _uses_async_decode_save(self, request: ReqMeta) -> bool:
        return bool(
            getattr(self, "_async_decode_worker_enabled", False)
            and request.is_decode_window_save
            and request.decode_save_job_id > 0
        )

    def _raise_async_decode_error(self) -> None:
        error = getattr(self, "_async_decode_error", None)
        if error is not None:
            raise RuntimeError("background async decode save failed") from error

    def _submit_async_decode_save(self, request: ReqMeta) -> None:
        self._raise_async_decode_error()
        assert self.lmcache_engine is not None
        with self._async_decode_lock:
            self._async_decode_known_reqs.add(request.req_id)
            if request.decode_save_is_final:
                # Speculative runners call get_finished before their deferred
                # connector finalization enqueues this final-tail job.
                self._async_decode_finished_waiting.add(request.req_id)

        is_passive = getattr(self.lmcache_engine, "_is_passive", None)
        if callable(is_passive) and is_passive():
            if request.decode_save_is_final:
                with self._async_decode_lock:
                    self._async_decode_finished_waiting.discard(request.req_id)
                    self._async_decode_finished_ready.add(request.req_id)
            return

        ordering_event = torch.npu.Event()
        ordering_event.record()
        task = _AsyncDecodeSaveTask(request, ordering_event)
        with self._async_decode_lock:
            jobs = self._async_decode_jobs.setdefault(request.req_id, deque())
            if (
                jobs
                and jobs[-1].request.decode_window_end
                != request.decode_window_start
            ):
                raise RuntimeError(
                    "async decode-save jobs are not contiguous on worker: "
                    f"req_id={request.req_id}, previous_end="
                    f"{jobs[-1].request.decode_window_end}, "
                    f"start={request.decode_window_start}"
                )
            jobs.append(task)
            self._enqueue_async_decode_request(request.req_id)

    def _enqueue_async_decode_request(self, req_id: str) -> None:
        """Queue one request once; dispatcher requeues it for round-robin."""
        if req_id in self._async_decode_ready_reqs:
            return
        try:
            self._async_decode_queue.put_nowait(req_id)
        except queue.Full as exc:
            self._set_async_decode_error(exc)
            raise RuntimeError(
                "async decode-save dispatcher queue exceeded scheduler limit"
            ) from exc
        self._async_decode_ready_reqs.add(req_id)

    def _take_queued_async_decode_task(
        self, req_id: str
    ) -> Optional[_AsyncDecodeSaveTask]:
        """Take one undispatched task and rotate a remaining request to tail."""
        with self._async_decode_lock:
            self._async_decode_ready_reqs.discard(req_id)
            jobs = self._async_decode_jobs.get(req_id, ())
            task = next(
                (job for job in jobs if not job.dispatched and not job.done),
                None,
            )
            if task is None:
                return None
            task.dispatched = True
            if any(not job.dispatched and not job.done for job in jobs):
                self._enqueue_async_decode_request(req_id)
            return task

    def _async_decode_dispatch_loop(self) -> None:
        torch.npu.set_device(self._async_decode_device_id)
        while True:
            req_id = self._async_decode_queue.get()
            if req_id is None:
                self._async_decode_queue.task_done()
                return
            task = self._take_queued_async_decode_task(req_id)
            if task is None:
                self._async_decode_queue.task_done()
                continue
            try:
                results = self._run_async_decode_npu_store(task)
            except AsyncDecodePersistenceError as exc:
                self._async_decode_completion_executor.submit(
                    self._retry_async_decode_save,
                    task,
                    exc,
                )
            except BaseException as exc:
                self._set_async_decode_error(exc)
            else:
                self._async_decode_completion_executor.submit(
                    self._wait_async_decode_persistence,
                    task,
                    results,
                )
            finally:
                self._async_decode_queue.task_done()

    def _run_async_decode_npu_store(
        self, task: _AsyncDecodeSaveTask
    ) -> list[tuple[int, Any]]:
        assert self.lmcache_engine is not None
        request = task.request
        required_groups = self._decode_window_save_required_groups(request)
        group_order = sorted(required_groups, reverse=True)
        results: list[tuple[int, Any]] = []
        for kv_group in group_order:
            self._refresh_kvcaches_list()
            kvcaches = self._kvcaches_for_group(kv_group)
            if not kvcaches:
                raise RuntimeError(
                    "async decode save is missing registered KV caches: "
                    f"req_id={request.req_id}, kv_group={kv_group}"
                )
            store_inputs = self._prepare_layerwise_store_inputs(
                request,
                request.save_spec,
                kv_group,
            )
            if store_inputs is None:
                raise RuntimeError(
                    "async decode save produced no store inputs: "
                    f"req_id={request.req_id}, kv_group={kv_group}"
                )
            (
                token_ids,
                slot_mapping,
                store_mask,
                skip_leading_tokens,
                store_kwargs,
                _,
            ) = store_inputs
            store_kwargs.update(
                {
                    "ordering_event": task.ordering_event,
                    "async_decode_job_key": task.owner,
                }
            )
            storer = self.lmcache_engine.store_layer(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                sync=True,
                req_id=request.req_id,
                **store_kwargs,
            )
            completed, result = self._finalize_layerwise_storer(storer)
            if (
                not completed
                or result is None
                or result.committed_end < int(request.decode_window_end or 0)
            ):
                raise AsyncDecodePersistenceError(
                    "async decode store did not cover the requested range: "
                    f"req_id={request.req_id}, kv_group={kv_group}, "
                    f"committed_end={getattr(result, 'committed_end', None)}, "
                    f"expected={request.decode_window_end}"
                )
            results.append((kv_group, result))
        return results

    def _wait_async_decode_persistence(
        self,
        task: _AsyncDecodeSaveTask,
        results: list[tuple[int, Any]],
    ) -> None:
        assert self.lmcache_engine is not None
        try:
            self.lmcache_engine.wait_for_pending_decode_save(task.owner)
        except AsyncDecodePersistenceError as exc:
            self._retry_async_decode_save(task, exc)
            return
        self._finish_async_decode_task(task, results)

    def _retry_async_decode_save(
        self,
        task: _AsyncDecodeSaveTask,
        error: BaseException,
    ) -> None:
        if task.attempt >= self._async_decode_max_retries:
            self._set_async_decode_error(error)
            return
        delay_index = min(task.attempt, len(ASYNC_DECODE_SAVE_RETRY_DELAYS) - 1)
        time.sleep(ASYNC_DECODE_SAVE_RETRY_DELAYS[delay_index])
        with self._async_decode_lock:
            task.attempt += 1
            task.dispatched = False
            task.results.clear()
            self._enqueue_async_decode_request(task.request.req_id)

    def _finish_async_decode_task(
        self,
        task: _AsyncDecodeSaveTask,
        results: list[tuple[int, Any]],
    ) -> None:
        assert self.lmcache_engine is not None
        with self._async_decode_lock:
            task.results = results
            task.done = True
            jobs = self._async_decode_jobs.get(task.request.req_id)
            if jobs is None:
                self._set_async_decode_error(
                    RuntimeError("async decode request queue disappeared")
                )
                return
            with self.lmcache_engine.state_guard():
                while jobs and jobs[0].done:
                    head = jobs.popleft()
                    for kv_group, result in head.results:
                        self._consume_completed_layerwise_store(
                            head.request,
                            kv_group,
                            True,
                            result,
                        )
                    expected = self._decode_window_save_expected_start
                    expected[head.request.req_id] = int(
                        head.request.decode_window_end or 0
                    )
                    self._clear_decode_window_save_groups_for_window(
                        head.request
                    )
            if not jobs:
                self._async_decode_jobs.pop(task.request.req_id, None)
                if task.request.req_id in self._async_decode_finished_waiting:
                    self._async_decode_finished_waiting.discard(
                        task.request.req_id
                    )
                    self._async_decode_finished_ready.add(task.request.req_id)
                self._async_decode_cv.notify_all()
        # Publish the physical completion even when an earlier job is still
        # pending. Scheduler-side ordered state prevents frontier advancement
        # but can release this exact job's source block lease immediately.
        self._record_async_decode_save_completion(task.request)

    def _set_async_decode_error(self, error: BaseException) -> None:
        with self._async_decode_lock:
            if self._async_decode_error is None:
                self._async_decode_error = error
            self._async_decode_cv.notify_all()

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
        self._raise_async_decode_error()
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
        with (
            self._async_decode_lock
            if getattr(self, "_async_decode_worker_enabled", False)
            else nullcontext()
        ):
            async_req_ids = finished_req_ids.intersection(
                getattr(self, "_async_decode_known_reqs", set())
            )
            pending_async = {
                req_id
                for req_id in async_req_ids
                if req_id in getattr(self, "_async_decode_jobs", {})
            }
            if pending_async:
                self._async_decode_finished_waiting.update(pending_async)
            ready_async = async_req_ids - pending_async
            regular_req_ids = finished_req_ids - async_req_ids

        finished_sending = set(
            self.lmcache_engine.get_finished_stores(regular_req_ids) or ()
        )
        releasable_req_ids = (
            finished_sending if self.store_async else regular_req_ids
        )
        releasable_req_ids.update(ready_async)
        finished_sending.update(ready_async)
        if releasable_req_ids:
            self._release_finished_worker_requests(releasable_req_ids)
        if ready_async:
            with self._async_decode_lock:
                self._async_decode_known_reqs.difference_update(ready_async)
        return finished_sending

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        self._raise_async_decode_error()
        finished_sending, finished_recving = super().get_finished(
            finished_req_ids
        )
        if not getattr(self, "_async_decode_worker_enabled", False):
            return finished_sending, finished_recving
        with self._async_decode_lock:
            ready = set(self._async_decode_finished_ready)
            self._async_decode_finished_ready.clear()
            self._async_decode_known_reqs.difference_update(ready)
        if ready:
            self._release_finished_worker_requests(ready)
            if finished_sending is None:
                finished_sending = set()
            finished_sending.update(ready)
        return finished_sending, finished_recving

    def get_decode_save_completions(self) -> list[KVConnectorSaveCompletion]:
        self._raise_async_decode_error()
        return super().get_decode_save_completions()

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

    def shutdown(self) -> None:
        if getattr(self, "_async_decode_worker_enabled", False):
            with self._async_decode_cv:
                self._async_decode_cv.wait_for(
                    lambda: not self._async_decode_jobs
                    or self._async_decode_error is not None
                )
            if self._async_decode_error is None:
                self._async_decode_queue.put(None)
                self._async_decode_dispatcher.join()
                self._async_decode_completion_executor.shutdown(
                    wait=True,
                    cancel_futures=False,
                )
            else:
                logger.critical(
                    "Async decode save failed before shutdown drain completed: %r",
                    self._async_decode_error,
                )
        super().shutdown()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        base_delay_free, return_params = super().request_finished(
            request, block_ids
        )
        delay_free = base_delay_free or (
            self.store_async and self.kv_role != "kv_consumer"
        )
        return delay_free, return_params
