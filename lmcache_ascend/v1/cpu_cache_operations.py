# SPDX-License-Identifier: Apache-2.0
"""Ascend CPU source and checkpoint operations, with no device-engine dependency."""

# Standard
from typing import Any, Optional, Iterable
from concurrent.futures import Future
from collections import deque
from urllib.parse import urlsplit
import secrets
import time

# First Party
from lmcache.integration.vllm.preemption_checkpoint import LOCAL_CHECKPOINT_CONFIG
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cpu_cache_operations import CPUCacheOperations
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryObj
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.serving_perf import (
    serving_perf_enabled,
    serving_perf_log,
    serving_perf_now,
)
from lmcache_ascend.v1.remote_fill import RemoteFillDecoderLayout

logger = init_logger(__name__)
_CHECKPOINT_RECLAIM_SCAN_ENTRIES = 4096


def _remote_fill_advertise_host_from_session(remote_session: str) -> str:
    """Extract the decoder host already advertised by its GlobalTE session."""

    value = str(remote_session).strip()
    if not value:
        raise ValueError("remote-fill GlobalTE session is empty")
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if not parsed.hostname:
        raise ValueError(
            "remote-fill cannot infer a routable control host from GlobalTE session"
        )
    return parsed.hostname


def _validate_remote_fill_control_port(
    base_port: int,
    dp_rank: int,
    dp_size: int,
    *,
    internal_api_enabled: bool,
    internal_api_port_start: int,
) -> int:
    """Validate the complete DP port range and return this rank's port."""

    control_port = int(base_port) + dp_rank
    last_control_port = int(base_port) + dp_size - 1
    if dp_size <= 0 or not 0 <= dp_rank < dp_size:
        raise ValueError("remote-fill decoder DP topology is invalid")
    if not 0 < int(base_port) < 65536 or last_control_port >= 65536:
        raise ValueError("remote-fill decoder DP control port is invalid")
    if internal_api_enabled:
        internal_start = int(internal_api_port_start)
        internal_end = internal_start + dp_size - 1
        if max(int(base_port), internal_start) <= min(last_control_port, internal_end):
            raise ValueError(
                "remote-fill decoder control ports overlap the internal API port range"
            )
    return control_port


class AscendCPUCacheOperations(CPUCacheOperations):
    """Use exact Ascend page geometry and the existing CPU checkpoint store."""

    def _num_layers_for_kv_group(self, kv_group: int) -> int:
        return self.num_layers_for_group(kv_group)

    def _resolve_shared_rank0_layer_pages(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        page_chunks: int,
        base_page_keys: Optional[list[CacheEngineKey]] = None,
        exact_chunk_locations: Optional[list[str]] = None,
    ) -> tuple[list[list[MemoryObj]], int]:
        """Resolve exact-size full or partial layer pages with legacy fallback."""
        perf_enabled = serving_perf_enabled()
        resolve_started = serving_perf_now() if perf_enabled else 0.0
        resolve_thread_started = time.thread_time_ns() if perf_enabled else 0
        local_get_ms = legacy_probe_ms = remote_probe_ms = remote_get_ms = 0.0
        tail_resolve_ms = validate_pin_ms = 0.0
        if not 0 < page_chunks <= len(keys_layer_major[0]):
            raise ValueError("Layer-page retrieval requires at least one chunk")
        page_keys = (
            list(base_page_keys[:page_chunks])
            if base_page_keys is not None
            else [
                key.without_layer()
                for key in keys_layer_major[0][:page_chunks]
                if isinstance(key, LayerCacheEngineKey)
            ]
        )
        if len(page_keys) != page_chunks:
            raise ValueError("Layer-page retrieval requires one base key per page")
        if exact_chunk_locations is not None:
            return self._resolve_shared_rank0_exact_layer_pages(
                req_id=req_id,
                phase=phase,
                kv_group=kv_group,
                keys_layer_major=keys_layer_major,
                page_chunks=page_chunks,
                base_page_keys=page_keys,
                chunk_locations=exact_chunk_locations,
            )

        local = self._shared_local_cpu_backend()
        assert self.storage_manager is not None
        phase_started = serving_perf_now() if perf_enabled else 0.0
        pages, local_count = local.batched_get_layer_page_prefix(page_keys)
        if perf_enabled:
            local_get_ms = (serving_perf_now() - phase_started) * 1000
        owned: list[MemoryObj] = list(pages)
        pinned: list[MemoryObj] = []
        try:
            tail_start = local_count
            while tail_start < page_chunks:
                phase_started = serving_perf_now() if perf_enabled else 0.0
                legacy_suffix = local.contains_all_exact(
                    page_keys[tail_start].split_layers(
                        self._num_layers_for_kv_group(kv_group)
                    )
                )
                if perf_enabled:
                    legacy_probe_ms += (serving_perf_now() - phase_started) * 1000
                if legacy_suffix:
                    break
                remote = self.storage_manager.storage_backends.get("RemoteBackend")
                contains = getattr(remote, "batched_contains_layer_pages", None)
                retrieve = getattr(remote, "batched_get_layer_pages", None)
                if not callable(contains) or not callable(retrieve):
                    raise RuntimeError("RemoteBackend does not support layer pages")
                remote_keys = page_keys[tail_start:page_chunks]
                phase_started = serving_perf_now() if perf_enabled else 0.0
                remote_count = contains(remote_keys)
                if perf_enabled:
                    remote_probe_ms += (serving_perf_now() - phase_started) * 1000
                if not 0 <= remote_count <= len(remote_keys):
                    raise ValueError("Remote layer-page lookup returned invalid count")
                phase_started = serving_perf_now() if perf_enabled else 0.0
                fetched = retrieve(remote_keys[:remote_count]) if remote_count else []
                if perf_enabled:
                    remote_get_ms += (serving_perf_now() - phase_started) * 1000
                if len(fetched) != remote_count:
                    raise ValueError("Remote layer-page result count is inconsistent")
                pages.extend(fetched)
                owned.extend(fetched)
                tail_start += remote_count
                if tail_start < page_chunks:
                    # Eviction can leave arbitrarily alternating tier runs.
                    phase_started = serving_perf_now() if perf_enabled else 0.0
                    fetched, count = local.batched_get_layer_page_prefix(
                        page_keys[tail_start:page_chunks]
                    )
                    if perf_enabled:
                        local_get_ms += (serving_perf_now() - phase_started) * 1000
                    pages.extend(fetched)
                    owned.extend(fetched)
                    tail_start += count
                    local_count += count
                    if not count:
                        # Neither tier has a merged page at this boundary.
                        break

            legacy_page_layers = (
                [
                    list(layer)
                    for layer in zip(
                        *(
                            key.split_layers(self._num_layers_for_kv_group(kv_group))
                            for key in page_keys[tail_start:page_chunks]
                        ),
                        strict=True,
                    )
                ]
                if tail_start < page_chunks
                else [[] for _ in range(self._num_layers_for_kv_group(kv_group))]
            )
            tail_keys_layer_major = [
                legacy_page_layers[layer_id]
                + list(keys_layer_major[layer_id][page_chunks:])
                for layer_id in range(self._num_layers_for_kv_group(kv_group))
            ]
            phase_started = serving_perf_now() if perf_enabled else 0.0
            tail = (
                # This retains the legacy layerwise fallback, but measures it
                # separately from merged-page retrieval.
                self._resolve_shared_rank0_page_first_layers(
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    keys_layer_major=tail_keys_layer_major,
                )
                if tail_keys_layer_major[0]
                else [[] for _ in keys_layer_major]
            )
            if perf_enabled:
                tail_resolve_ms = (serving_perf_now() - phase_started) * 1000
            tail_objects = [obj for layer in tail for obj in layer]
            owned.extend(tail_objects)
            pinned.extend(tail_objects)
            if len(pages) != tail_start:
                raise ValueError(
                    f"Layer-page result count {len(pages)} != {tail_start}"
                )
            phase_started = serving_perf_now() if perf_enabled else 0.0
            invalid_page = False
            for chunk_index, page in enumerate(pages):
                token_count = int(getattr(page, "valid_tokens", 0))
                expected_shape, _, _ = self._expected_shared_cpu_chunk_metadata(
                    kv_group=kv_group,
                    num_tokens=token_count,
                )
                invalid_page |= (
                    page.num_layers != self._num_layers_for_kv_group(kv_group)
                    or page.get_shape() != expected_shape
                )
            if invalid_page or not LayerPageMemoryObj.pin_many(pages):
                raise ValueError("Invalid or unpinnable layer-page object")
            pinned.extend(pages)
            for chunk_index, page in enumerate(pages):
                self._validate_rank0_shared_mem_obj(
                    page,
                    req_id=req_id,
                    phase=phase,
                    layer_id=0,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                )
            if perf_enabled:
                validate_pin_ms = (serving_perf_now() - phase_started) * 1000
                elapsed_ms = (serving_perf_now() - resolve_started) * 1000
                if elapsed_ms >= 100.0:
                    serving_perf_log(
                        logger,
                        "rank0_page_resolve_slow",
                        started=resolve_started,
                        req_id=req_id,
                        phase=phase,
                        kv_group=kv_group,
                        pages=page_chunks,
                        local_pages=local_count,
                        local_get_ms=round(local_get_ms, 3),
                        legacy_probe_ms=round(legacy_probe_ms, 3),
                        remote_probe_ms=round(remote_probe_ms, 3),
                        remote_get_ms=round(remote_get_ms, 3),
                        tail_resolve_ms=round(tail_resolve_ms, 3),
                        validate_pin_ms=round(validate_pin_ms, 3),
                        thread_cpu_ms=round(
                            (time.thread_time_ns() - resolve_thread_started) / 1e6,
                            3,
                        ),
                    )
            return [list(pages) + layer for layer in tail], tail_start
        except Exception:
            for obj in reversed(pinned):
                obj.unpin()
            self._release_shared_retrieve_objs(owned, unpin=False)
            raise

    def _validate_remote_fill_decoder_layout(
        self,
        layout: RemoteFillDecoderLayout,
    ) -> None:
        """Match both negotiated groups to the model-visible cache layout."""

        layer_groups = tuple(
            getattr(
                getattr(self.metadata, "kv_layer_groups_manager", None),
                "kv_layer_groups",
                (),
            )
        )
        if layer_groups and len(layer_groups) != 2:
            raise ValueError(
                "remote-fill requires exactly two registered decoder cache "
                f"groups, found {len(layer_groups)}"
            )

        for kv_group in (0, 1):
            shape, dtype, fmt = self._expected_shared_cpu_chunk_metadata(
                kv_group=kv_group,
                num_tokens=1,
            )
            group = layout.group(kv_group)
            registered_layers = (
                layer_groups[kv_group].num_layers if layer_groups else None
            )
            if (
                registered_layers is not None
                and registered_layers != layout.num_layers_for_group(kv_group)
            ):
                raise ValueError(
                    "remote-fill requires registered cache-bearing layer "
                    "coverage for both groups: "
                    f"kv_group={kv_group}, "
                    f"registered_layers={registered_layers}, "
                    f"negotiated_layers={layout.num_layers_for_group(kv_group)}"
                )
            if (
                dtype != group.dtype
                or fmt != group.fmt
                or int(shape.numel())
                != (
                    group.layer_token_dims[0]
                    if group.layer_token_dims
                    else group.raw_token_dim
                )
            ):
                raise ValueError(
                    "remote-fill negotiated group layout disagrees with "
                    "decoder cache metadata: "
                    f"kv_group={kv_group}, "
                    f"decoder_shape={tuple(shape)}, "
                    f"decoder_elements_per_token={int(shape.numel())}, "
                    f"decoder_dtype={dtype}, decoder_format={fmt.name}, "
                    f"negotiated_elements_per_token={group.raw_token_dim}, "
                    f"negotiated_dtype={group.dtype}, "
                    f"negotiated_format={group.fmt.name}"
                )

    def _remote_fill_capacity_available(self, requested_bytes: int) -> bool:
        if requested_bytes < 0:
            return False
        try:
            free_bytes, heap_bytes = (
                self._shared_local_cpu_backend().get_allocator_capacity_bytes()
            )
        except (NotImplementedError, RuntimeError):
            return False
        reserve = max(
            int(self.config.remote_fill_min_free_bytes),
            int(heap_bytes * float(self.config.remote_fill_min_free_ratio)),
        )
        return free_bytes - requested_bytes >= reserve

    def _remote_fill_reclaim_capacity(self, requested_bytes: int) -> bool:
        if requested_bytes < 0:
            return False
        reclaim = getattr(
            self._shared_local_cpu_backend(),
            "reclaim_evictable_capacity",
            None,
        )
        if not callable(reclaim):
            return False
        return bool(
            reclaim(
                requested_bytes,
                min_free_bytes=int(self.config.remote_fill_min_free_bytes),
                min_free_ratio=float(self.config.remote_fill_min_free_ratio),
                num_layers=getattr(self.metadata, "runtime_kv_group_layer_counts", None)
                or self.num_layers,
                cause="remote_fill_capacity_reclaim",
            )
        )

    def checkpoint_backend(self) -> Any:
        """Return the existing registered LocalCPU heap for checkpoint pages."""
        local = self._shared_local_cpu_backend()
        if local is None or not local.use_hot:
            raise RuntimeError("Local checkpoint storage is unavailable")
        return local

    def checkpoint_page_key(
        self, spec: Any, group: int, start: int, end: int
    ) -> CacheEngineKey:
        """Make an isolated local-only key; unsealed bytes cannot be prefix hits."""
        return CacheEngineKey(
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            secrets.token_bytes(16),
            self._shared_cpu_dtype_for_kv_group(group),
            {
                "lmcache.tag.checkpoint": (
                    f"{spec.req_id}:{spec.generation}:{start}:{end}"
                )
            },
            kv_group=group,
        )

    @staticmethod
    def is_checkpoint_page_key(key: CacheEngineKey) -> bool:
        """Identify private staging keys without touching canonical prefix keys."""
        return "lmcache.tag.checkpoint" in (key.request_configs or {})

    def validate_checkpoint_page(self, group: int, page: LayerPageMemoryObj) -> None:
        """Reject incompatible local layouts before constructing transfer operands."""
        if (
            page.num_layers != self._num_layers_for_kv_group(group)
            or page.get_dtype() != self._shared_cpu_dtype_for_kv_group(group)
            or page.metadata.fmt != self._memory_format_for_kv_group(group)
        ):
            raise ValueError("Local checkpoint page layout mismatch")

    def load_checkpoint_prefix(
        self, tokens: tuple[int, ...], group: int, configs: Any
    ) -> Any:
        """Retain the exact original boundary page, fetching only that page on miss."""
        start, end, key = deque(
            self.token_database.process_tokens(
                tokens=list(tokens),
                request_configs=configs,
                kv_group=group,
            ),
            maxlen=1,
        ).pop()
        cached = self.get_checkpoint_prefix(key, group, end - start, with_widths=False)
        if cached is not None:
            return cached[0]
        page, _ = self.allocate_checkpoint_fragment(group, end - start)
        try:
            self.storage_manager.batched_get_external_pages(
                [key],
                [
                    [
                        page.layer_data_ptr(i)
                        for i in range(self._num_layers_for_kv_group(group))
                    ]
                ],
                [
                    [
                        page.layer_size_bytes(i)
                        for i in range(self._num_layers_for_kv_group(group))
                    ]
                ],
                (page.raw_data,),
                "checkpoint-prefix",
            )
            return page
        except NativeExternalPageTransferUnknownError:
            self._checkpoint_quarantined_owners = getattr(
                self, "_checkpoint_quarantined_owners", []
            )
            self._checkpoint_quarantined_owners.append(page)
            raise
        except BaseException:
            page.ref_count_down()
            raise

    def _lookup_remote_fill_two_group_prefix(
        self,
        chunks: list,
        *,
        search_range: list,
        lookup_id: str | None,
        pin: bool,
        request_configs: Any = None,
        diagnostics: Any = None,
    ) -> int:
        generation = (request_configs or {}).get(LOCAL_CHECKPOINT_CONFIG)
        if generation is None:
            return super()._lookup_remote_fill_two_group_prefix(
                chunks,
                search_range=search_range,
                lookup_id=lookup_id,
                pin=pin,
                request_configs=request_configs,
                diagnostics=diagnostics,
            )
        worker = getattr(self, "checkpoint_worker", None)
        local = getattr(self, "checkpoint_store", None) or (
            worker.local if worker else None
        )
        request_id = getattr(self, "_lookup_request_ids", {}).get(lookup_id, lookup_id)
        manifest = local.manifest(request_id, int(generation)) if local else None
        if manifest is None or not chunks:
            return 0
        configs = dict(request_configs)
        configs.pop(LOCAL_CHECKPOINT_CONFIG)
        prefix = list(
            self.token_database.process_tokens(
                tokens=list(manifest.tokens[: manifest.prefix_end]),
                request_configs=configs,
            )
        )
        hit = (
            super()._lookup_remote_fill_two_group_prefix(
                prefix,
                search_range=search_range,
                lookup_id=None,
                pin=False,
                request_configs=configs,
            )
            if prefix
            else 0
        )
        if hit != manifest.prefix_end:
            return hit
        expected = list(
            self.token_database.process_tokens(
                tokens=list(manifest.tokens[: chunks[-1][1]]),
                request_configs=configs,
            )
        )
        if chunks != expected:
            return 0
        return local.available(request_id, int(generation), chunks[-1][1])

    def checkpoint_token_results(
        self,
        tokens: list[int],
        request_configs: Any,
        kv_group: int,
        state: dict,
        *,
        start: int = 0,
    ) -> Iterable[tuple[int, int, CacheEngineKey]]:
        """Hash once per operation; only build keys for its checkpoint windows.

        State is private to one immutable token/configuration snapshot and start.
        The complete prefix hash chain is computed even when its keys are unused.
        """
        plan = state.get("checkpoint_chunks")
        if plan is None:
            plan = tuple(
                (a, b, digest)
                for a, b, digest in self.token_database.process_tokens(
                    tokens=tokens,
                    make_key=False,
                    request_configs=request_configs,
                    kv_group=kv_group,
                )
                if a >= start
            )
            state["checkpoint_chunks"] = plan
        generated = self.token_database.process_tokens(
            hashes=[entry[2] for entry in plan],
            offsets=[entry[1] - entry[0] for entry in plan],
            request_configs=request_configs,
            kv_group=kv_group,
        )
        return (
            (a, b, key) for (a, b, _), (_, _, key) in zip(plan, generated, strict=True)
        )

    def probe_checkpoint_pages(self, keys: list) -> list[bool]:
        """Distinguish confirmed missing remote pages from transport errors."""
        return self.storage_manager.batched_external_pages_exist(keys, strict=True)

    def persist_checkpoint_page(
        self, key: CacheEngineKey, page: Any, req_id: str
    ) -> Future:
        """Submit a sealed page; the caller retains its owner until completion."""
        return self.storage_manager.batched_put_external_pages(
            [key],
            [[page.layer_data_ptr(i) for i in range(page.num_layers)]],
            [[page.layer_size_bytes(i) for i in range(page.num_layers)]],
            (page.raw_data,),
            None,
            req_id,
        )

    def load_checkpoint_page(
        self, key: CacheEngineKey, group: int, tokens: int, req_id: str
    ) -> Any:
        """Load a canonical checkpoint page, retaining unknown native destinations."""
        try:
            page, _ = self.allocate_checkpoint_fragment(group, tokens)
        except MemoryError:
            if not self.reclaim_checkpoint_capacity(tokens, {group: None}):
                raise
            page, _ = self.allocate_checkpoint_fragment(group, tokens)
        try:
            self.storage_manager.batched_get_external_pages(
                [key],
                [[page.layer_data_ptr(i) for i in range(page.num_layers)]],
                [[page.layer_size_bytes(i) for i in range(page.num_layers)]],
                (page.raw_data,),
                req_id,
            )
            self.validate_checkpoint_page(group, page)
            return page
        except NativeExternalPageTransferUnknownError:
            self.__dict__.setdefault("_checkpoint_quarantined_owners", []).append(page)
            raise
        except BaseException as error:
            page.ref_count_down()
            failed = getattr(error, "failed_pages", ())
            if failed and all(status <= 0 for _, status, _ in failed):
                if not self.probe_checkpoint_pages([key])[0]:
                    return None
            raise
