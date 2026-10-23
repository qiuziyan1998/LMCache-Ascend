# SPDX-License-Identifier: Apache-2.0
"""Ascend CPU storage services using the registered layout and existing backends."""

# Standard
from dataclasses import dataclass, field
from threading import RLock
from types import SimpleNamespace
from typing import Any
import secrets

# Third Party
import msgspec

# First Party
from lmcache.integration.vllm.preemption_checkpoint import (
    LOCAL_CHECKPOINT_CONFIG,
    CheckpointRestoreMiss,
)
from lmcache.v1.cpu_cache import StorageFailedSafe
from lmcache.v1.cpu_cache_operations import (
    CPUCacheOperations,
    _RemoteFillChunkLookupPlan,
    _RemoteFillLookupPlan,
)
from lmcache.v1.cpu_cache_runtime import CPUCacheRuntime
from lmcache.v1.mooncake_layout import mooncake_payload_layout
from lmcache.v1.memory_management import LayerPageMemoryObj
from lmcache.v1.rpc.storage_protocol import (
    LookupReply,
    OperationIdentity,
    PrepareSpec,
    checked_uint,
    pack_uints,
    restore_hash,
    unpack_uints,
)
from lmcache.v1.rpc.storage_layout import freeze_storage_layout
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache_ascend.v1.cpu_cache_operations import (
    AscendCPUCacheOperations,
    _remote_fill_advertise_host_from_session,
    _validate_remote_fill_control_port,
)
from lmcache_ascend.v1.local_checkpoint import (
    CheckpointPage,
    LocalCheckpoint,
    LocalCheckpointStore,
)
from lmcache_ascend.v1.remote_fill import (
    build_decoder_layout,
    create_decoder_remote_fill_runtime,
)


@dataclass
class _WriteReservation:
    identity: OperationIdentity
    purpose: str
    generation: int
    prefix_end: int
    fragments: dict[int, list[tuple[int, int, Any]]] = field(default_factory=dict)
    leases: list[str] = field(default_factory=list)
    prefix_sources: tuple[CheckpointPage, ...] = ()
    captured_end: int = 0
    captured: bool = False
    committing: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    layer_ids: dict[int, list[int]] = field(default_factory=dict)


class AscendCPUCacheRuntime(AscendCPUCacheOperations, CPUCacheRuntime):
    """Own decoder CPU pages, RemoteFill receiver and local checkpoint offers."""

    def __init__(self, *args: Any, bootstrap_config: Any = None, **kwargs: Any) -> None:
        self.bootstrap_config = bootstrap_config
        self._remote_fill_runtime = None
        self._remote_fill_fatal_transfers: tuple[str, ...] = ()
        self._checkpoint_quarantined_owners: list[Any] = []
        self._write_lock = RLock()
        self._write_reservations: dict[str, _WriteReservation] = {}
        super().__init__(*args, **kwargs)
        self.checkpoint_store = LocalCheckpointStore(self)
        self.save_only_first_rank = bool(
            self.config.get_extra_config_value(
                "save_only_first_rank", self.metadata.use_mla
            )
            and self.metadata.use_mla
        )
        self.save_indexer_only_first_rank = self.save_only_first_rank

    def bootstrap_descriptor(self) -> dict[str, Any]:
        """Advertise packed checkpoint support through the existing bootstrap."""
        return dict(super().bootstrap_descriptor(), checkpoint_tokens_u32_le_v1=True)

    def on_layout_bound(self) -> None:
        """Finalize hashing and advertise RemoteFill after every worker is ready."""
        if self.token_database is None:
            self.token_database = ChunkedTokenDatabase(self.config, self.metadata)
        expected = bytes if self.layout.hash_kind == "bytes" else int
        if self.token_database.chunk_hash_type is not expected:
            raise ValueError("Storage hash representation disagrees with workers")
        if (
            expected is bytes
            and self.token_database.chunk_hash_bytes * 8 != self.layout.hash_width
        ):
            raise ValueError("Storage digest width disagrees with workers")
        expected_layout = freeze_storage_layout(
            self.config,
            self.metadata,
            self.token_database,
            allocator_alignment=self.allocator_alignment,
            plane_token_bytes={
                group.group_id: group.plane_token_bytes for group in self.layout.groups
            },
        )
        if expected_layout != self.layout:
            raise ValueError(
                "Storage source identity disagrees with registered workers"
            )
        self._initialize_decoder_remote_fill()

    def checkpoint_plane_widths(self, group: int) -> tuple[Any, ...]:
        """Return element widths for the existing CPU checkpoint assembler."""
        widths = []
        for layer, planes in enumerate(self.layout.groups[group].plane_token_bytes):
            _, dtype, _ = self._expected_shared_cpu_chunk_metadata(
                kv_group=group,
                num_tokens=1,
                layer_id=layer,
            )
            if any(width % dtype.itemsize for width in planes):
                raise ValueError("Checkpoint plane is not dtype aligned")
            widths.append(tuple(width // dtype.itemsize for width in planes))
        return widths[0] if all(row == widths[0] for row in widths) else tuple(widths)

    def allocate_checkpoint_fragment(
        self,
        group: int,
        tokens: int,
        caches: Any = None,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Reserve an exact registered page without eviction or native device access."""
        if caches is not None or tokens <= 0:
            raise ValueError("CPU capture reservations contain layout values only")
        local = self.checkpoint_backend()
        geometry = [
            self._expected_shared_cpu_chunk_metadata(
                kv_group=group,
                num_tokens=tokens,
                layer_id=layer,
            )
            for layer in range(self.num_layers_for_group(group))
        ]
        pages = local.batched_allocate_layer_pages(
            [shape for shape, _, _ in geometry],
            [dtype for _, dtype, _ in geometry],
            1,
            len(geometry),
            geometry[0][2],
            busy_loop=False,
            eviction=False,
            valid_tokens=tokens,
            full_tokens=tokens,
        )
        if not pages:
            raise MemoryError("Checkpoint CPU staging allocation refused")
        return pages[0], self.checkpoint_plane_widths(group)

    def reclaim_checkpoint_capacity(
        self, tokens: int, group_caches: dict[int, Any]
    ) -> bool:
        """Use the allocator's actual aligned page budget and bounded reclaim path."""
        if tokens <= 0 or not group_caches:
            return False
        local = self.checkpoint_backend()
        allocator = local.get_memory_allocator()
        aligned = getattr(
            getattr(allocator, "address_manager", None), "compute_aligned_size", None
        )
        required = 0
        for group in group_caches:
            size = sum(
                shape.numel() * dtype.itemsize
                for shape, dtype, _ in (
                    self._expected_shared_cpu_chunk_metadata(
                        kv_group=group,
                        num_tokens=tokens,
                        layer_id=layer,
                    )
                    for layer in range(self.num_layers_for_group(group))
                )
            )
            alignment = self.allocator_alignment
            required += (
                aligned(size)
                if aligned
                else (size + alignment - 1) // alignment * alignment
            )
        return local.reclaim_evictable_capacity(
            required,
            min_free_bytes=0,
            min_free_ratio=0,
            num_layers=self._transfer_layer_counts,
            cause="checkpoint_capacity_reclaim",
            max_scan_entries=4096,
        )

    def get_checkpoint_prefix(self, key: Any, group: int, tokens: int) -> Any:
        """Retain an exact registered local page before checkpoint normalization."""
        pages, count = self.checkpoint_backend().batched_get_layer_page_prefix([key])
        if not count:
            return None
        page = pages[0]
        try:
            self.validate_checkpoint_page(group, page)
            if not page.is_valid() or page.valid_tokens != tokens:
                page.ref_count_down()
                return None
            return page, self.checkpoint_plane_widths(group)
        except BaseException:
            page.ref_count_down()
            raise

    def prepare_checkpoint_restore_sources(
        self,
        spec: PrepareSpec,
        identity: OperationIdentity,
        reply: LookupReply,
    ) -> tuple[int, list[Any]]:
        """Normalize accepted local history using the existing checkpoint store."""
        generation = reply.request_configs.get(LOCAL_CHECKPOINT_CONFIG)
        if generation is None or generation != spec.checkpoint_generation:
            raise StorageFailedSafe(
                "Checkpoint generation disagrees with lookup ticket"
            )
        manifest = self.checkpoint_store.manifest(identity.request_id, generation)
        if manifest is None:
            raise StorageFailedSafe(
                "Local checkpoint offer is unavailable",
                details={"checkpoint_miss_end": 0},
            )
        configs = dict(reply.request_configs)
        configs.pop(LOCAL_CHECKPOINT_CONFIG)
        owners = []
        try:
            base, owners = self.checkpoint_store.normalize(
                identity.request_id,
                generation,
                list(manifest.tokens[: spec.source_end]),
                configs,
            )
            candidates, _ = self._chunks(
                {
                    "starts": reply.starts,
                    "ends": reply.ends,
                    "hashes": reply.hashes,
                    "request_configs": configs,
                }
            )
            prefix = [chunk for chunk in candidates if chunk[0] < base]
            if prefix:
                hit = CPUCacheOperations._lookup_remote_fill_two_group_prefix(
                    self,
                    prefix,
                    search_range=list(
                        self.config.retrieve_locations
                        or self.storage_manager.storage_backends
                    ),
                    lookup_id=spec.ticket_id,
                    pin=True,
                    request_configs=configs,
                )
                if hit != base:
                    raise CheckpointRestoreMiss(
                        hit, "Checkpoint persistent prefix became unavailable"
                    )
            retained = self._remote_fill_lookup_plans.get(spec.ticket_id)
            tail = [chunk for chunk in candidates if base <= chunk[0] < spec.source_end]
            group_keys = [
                [
                    self._lookup_key_for_kv_group(
                        key, kv_group=group, request_configs=configs
                    ).get_first_layer()
                    for _, _, key in tail
                ]
                for group in (0, 1)
            ]
            count, pins = self.storage_manager.batched_contains_two_group_layer_pages(
                group_keys[0],
                group_keys[1],
                ["LocalCPUBackend"],
                True,
            )
            for location, keys in pins.items():
                self.lookup_pins[spec.ticket_id][location].extend(keys)
            if count != len(tail):
                raise CheckpointRestoreMiss(
                    tail[count][0],
                    "Normalized checkpoint was evicted before preparation",
                )
            plans = tuple(retained.chunks) if retained else ()
            plans += tuple(
                _RemoteFillChunkLookupPlan(
                    start,
                    end,
                    key.chunk_hash,
                    ("LocalCPUBackend", "LocalCPUBackend"),
                    (True, True),
                )
                for start, end, key in tail
            )
            self._remote_fill_lookup_plans[spec.ticket_id] = _RemoteFillLookupPlan(
                plans
            )
            return base, owners
        except BaseException as error:
            self._release_shared_retrieve_objs(owners, unpin=False)
            if isinstance(error, CheckpointRestoreMiss):
                raise StorageFailedSafe(
                    str(error),
                    details={"checkpoint_miss_end": error.available_end},
                ) from error
            raise

    def get_remote_fill_placement_info(self) -> dict[str, Any] | None:
        """Return the existing pointer-free receiver placement after startup."""
        runtime = self._remote_fill_runtime
        if runtime is None or not runtime.available or not self.is_healthy():
            return None
        return runtime.placement.to_dict()

    def reserve_write(
        self, value: dict[str, Any], identity: OperationIdentity
    ) -> dict[str, Any]:
        """Reserve a batch of private registered destinations before worker capture."""
        self._require_ready()
        if self.storage_manager.is_frozen():
            return {"status": "SKIPPED", "groups": (), "captured_end": 0}
        purpose = value.get("purpose", "store")
        if purpose not in ("store", "checkpoint"):
            raise StorageFailedSafe("Unknown CPU write purpose")
        reservation_id = secrets.token_hex(16)
        reservation = _WriteReservation(
            identity,
            purpose,
            int(value.get("generation", identity.load_generation)),
            int(value.get("prefix_end", 0)),
            options=dict(value),
        )
        with self._write_lock:
            if len(self._write_reservations) >= max(
                1, int(self.config.store_async_max_queue_size or 2)
            ):
                raise StorageFailedSafe(
                    "CPU write staging is busy",
                    details={"store_refused": True} if purpose == "store" else None,
                )
            self._write_reservations[reservation_id] = reservation
        acquired = []
        try:
            if purpose == "checkpoint":
                self._reserve_checkpoint_pages(reservation, value)
            else:
                self._reserve_store_objects(reservation, value)
            groups = []
            for group, fragments in reservation.fragments.items():
                objects = [page for _, _, page in fragments]
                lease = self.registry.add_lease(
                    identity,
                    objects,
                    participants=(identity.client_session,),
                    producer_holds=3,
                    release=lambda owned: self._release_shared_retrieve_objs(
                        list(owned), unpin=False
                    ),
                )
                acquired.extend(objects)
                reservation.leases.append(lease)
                if isinstance(objects[0], LayerPageMemoryObj):
                    geometries, geometry_ids = [], []
                    for page in objects:
                        geometry = self.source_geometry(group, page)
                        geometry.validate(
                            self.num_layers_for_group(group),
                            int(page.metadata.phy_size),
                        )
                        if geometry not in geometries:
                            geometries.append(geometry)
                        geometry_ids.append(geometries.index(geometry))
                    item = {
                        "representation": "page",
                        "plane_widths": self.checkpoint_plane_widths(group),
                        "page_offsets": pack_uints(
                            int(page.metadata.address) for page in objects
                        ),
                        "page_extents": pack_uints(
                            int(page.metadata.phy_size) for page in objects
                        ),
                        "geometry_ids": pack_uints(geometry_ids, 32),
                        "geometries": tuple(geometries),
                    }
                else:
                    entry = next(
                        (
                            item
                            for item in reservation.options.get("groups", ())
                            if item["group_id"] == group
                        ),
                        {},
                    )
                    item = {
                        "representation": entry.get("representation", "flat"),
                        "objects": tuple(
                            {
                                "offset": int(obj.metadata.address),
                                "extent": int(obj.metadata.phy_size),
                                "shapes": tuple(
                                    tuple(shape) for shape in obj.get_shapes()
                                ),
                                "dtypes": tuple(
                                    str(dtype).removeprefix("torch.")
                                    for dtype in obj.get_dtypes()
                                ),
                                "memory_format": int(obj.get_memory_format().value),
                                "logical_size": obj.get_size(),
                                "valid_tokens": end - start,
                            }
                            for start, end, obj in fragments
                        ),
                    }
                    if group in reservation.layer_ids:
                        item["layer_ids"] = tuple(reservation.layer_ids[group])
                item.update(
                    group_id=group,
                    lease_id=lease,
                    starts=tuple(start for start, _, _ in fragments),
                    ends=tuple(end for _, end, _ in fragments),
                )
                groups.append(item)
            if self.registry.cancellation_requested(identity):
                raise StorageFailedSafe(
                    "CPU write reservation was cancelled before export"
                )
            reply = dict(
                self.bootstrap_descriptor(),
                reservation_id=reservation_id,
                captured_end=reservation.captured_end,
                groups=tuple(groups),
                status="RESERVED",
            )
            self.registry.publish_leases(identity, reservation.leases)
            return reply
        except BaseException as error:
            for lease in reservation.leases:
                self.registry.producer_done(lease, known_complete=True)
                self.registry.producer_done(lease, known_complete=True)
                self.registry.producer_done(lease, known_complete=True)
                self.registry.discard_unexported_lease(identity, lease)
            acquired_ids = {id(obj) for obj in acquired}
            self._release_shared_retrieve_objs(
                [
                    obj
                    for fragments in reservation.fragments.values()
                    for _, _, obj in fragments
                    if id(obj) not in acquired_ids
                ],
                unpin=False,
            )
            with self._write_lock:
                self._write_reservations.pop(reservation_id, None)
            if isinstance(error, (MemoryError, ValueError)):
                raise StorageFailedSafe(
                    str(error),
                    details={"store_refused": True}
                    if purpose == "store" and isinstance(error, MemoryError)
                    else None,
                ) from error
            raise

    def commit_write(
        self, value: dict[str, Any], identity: OperationIdentity
    ) -> dict[str, Any]:
        """Publish only fenced worker bytes, preserving persistence completion."""
        reservation = self._write_reservation(value["reservation_id"], identity)
        phase = value.get("phase", "store")
        with self._write_lock:
            if reservation.committing:
                raise StorageFailedSafe("CPU write publication is already running")
            reservation.committing = True
        try:
            tokens: tuple[int, ...] = ()
            if reservation.purpose == "checkpoint" and phase != "captured":
                if phase != "seal":
                    raise StorageFailedSafe(
                        "Checkpoint publication requires accepted-token sealing"
                    )
                # Decode only after count/extent/frontier checks and before any
                # producer hold is released or checkpoint pages are published.
                tokens = self._checkpoint_seal_tokens(reservation, value)
            if not reservation.captured:
                if (
                    value.get("proof") != "fenced"
                    or identity.client_session != reservation.identity.client_session
                ):
                    raise StorageFailedSafe(
                        "CPU publication requires a worker final-access fence"
                    )
                reservation.captured = True
                for lease in reservation.leases:
                    self.registry.producer_done(lease, known_complete=True)
            if phase == "captured":
                return {"status": "CAPTURED", "captured_end": reservation.captured_end}
            if reservation.purpose == "checkpoint":
                self._seal_checkpoint(reservation, tokens)
            else:
                self._commit_store(reservation)
            for lease in reservation.leases:
                self.registry.producer_done(lease, known_complete=True)
            with self._write_lock:
                self._write_reservations.pop(value["reservation_id"], None)
            return {"status": "COMMITTED", "committed_end": reservation.captured_end}
        finally:
            with self._write_lock:
                reservation.committing = False

    def cancel_write(
        self, value: dict[str, Any], identity: OperationIdentity
    ) -> dict[str, Any]:
        """Retain unknown DMA destinations; release only explicitly drained writers."""
        reservation = self._write_reservation(
            value["reservation_id"], identity, missing_ok=True
        )
        if reservation is None:
            return {"status": "CANCELLED"}
        with self._write_lock:
            if reservation.committing:
                return {"status": "PENDING"}
            proof = value.get("proof")
            if (
                not reservation.captured
                and identity.client_session != reservation.identity.client_session
            ):
                raise StorageFailedSafe(
                    "Only the capturing worker may prove DMA completion"
                )
            if proof not in ("fenced", "never_imported"):
                for lease in reservation.leases:
                    self.registry.producer_done(lease, known_complete=False)
                self.registry.mark_fatal("Worker capture completion is unknown")
                return {"status": "QUARANTINED"}
            remaining = 1 if reservation.captured else 2
            reservation.captured = True
        for lease in reservation.leases:
            for _ in range(remaining):
                self.registry.producer_done(lease, known_complete=True)
        with self._write_lock:
            self._write_reservations.pop(value["reservation_id"], None)
        return {"status": "CANCELLED"}

    def is_healthy(self) -> bool:
        """Include the native receiver's latched unknown-completion state."""
        return not self._remote_fill_fatal_transfers and super().is_healthy()

    def close(self) -> None:
        """Close RemoteFill before its registered CPU storage, after lease drain."""
        if self._remote_fill_fatal_transfers or self._checkpoint_quarantined_owners:
            raise RuntimeError(
                "Ascend CPU storage still owns unresolved native transfers"
            )
        with self._lock:
            if self._closed:
                return
            self._closing = True
        try:
            self.check_close_ready()
            if self._remote_fill_runtime is not None:
                self._remote_fill_runtime.close()
                # Shutdown itself may discover an armed destination. Retain
                # the lifecycle and allocator owners until paired restart.
                if self._remote_fill_fatal_transfers:
                    raise RuntimeError(
                        "RemoteFill shutdown found unresolved native transfers"
                    )
                self._remote_fill_runtime = None
            super().close()
        finally:
            with self._lock:
                self._closing = False

    def _dispatch(self, request: Any) -> Any:
        if request.method == "reserve_write":
            return self.reserve_write(request.payload, request.identity)
        if request.method == "commit_write":
            return self.commit_write(request.payload, request.identity)
        if request.method == "cancel_write":
            return self.cancel_write(request.payload, request.identity)
        if request.method == "remote_fill_placement":
            return self.get_remote_fill_placement_info()
        if request.method == "remote_fill_metrics":
            return (
                self._remote_fill_runtime.metrics_snapshot()
                if self._remote_fill_runtime
                else None
            )
        if request.method in ("forget_checkpoint", "checkpoint_forget"):
            self.checkpoint_store.forget(
                request.identity.request_id,
                request.payload.get("generation"),
            )
            return None
        return super()._dispatch(request)

    def _write_reservation(
        self,
        reservation_id: str,
        identity: OperationIdentity,
        *,
        missing_ok: bool = False,
    ) -> _WriteReservation | None:
        with self._write_lock:
            reservation = self._write_reservations.get(reservation_id)
            if reservation is None and missing_ok:
                return None
            if reservation is None or (
                reservation.identity.request_id,
                reservation.identity.request_attempt,
            ) != (identity.request_id, identity.request_attempt):
                raise StorageFailedSafe(
                    "CPU write reservation belongs to another request attempt"
                )
            return reservation

    def _reserve_checkpoint_pages(
        self, reservation: _WriteReservation, value: dict[str, Any]
    ) -> None:
        if not self.config.decode_preemption_checkpoint:
            raise StorageFailedSafe("Decoder preemption checkpointing is disabled")
        base, end, resident = (
            int(value["base"]),
            int(value["end"]),
            int(value["resident_start"]),
        )
        if (
            not 0 <= base <= reservation.prefix_end <= end
            or not base <= resident <= end
        ):
            raise StorageFailedSafe("Invalid checkpoint frontiers")
        operation = value.get("source_operation")
        source_manifest = operation.get("manifest") if operation else None
        if source_manifest is not None:
            from lmcache.v1.rpc.storage_protocol import GroupManifest

            source_manifest = msgspec.convert(source_manifest, type=GroupManifest)
        lease_id = value.get("source_lease_id") or (
            source_manifest.lease_id if source_manifest is not None else None
        )
        source = (
            self.validate_source_operation(operation, reservation.identity)
            if operation
            else self.source_key_plans.get(lease_id)
        )
        prefix_sources = value.get("prefix_sources") or []
        if source is not None:
            old_identity, reply, group = source
            if group != 0 or (
                old_identity.request_id,
                old_identity.request_attempt,
            ) != (
                reservation.identity.request_id,
                reservation.identity.request_attempt,
            ):
                raise StorageFailedSafe(
                    "Checkpoint prefix source belongs to another request"
                )
            canonical = [
                {"start": a, "end": min(b, resident), "object_end": b, "hash": h}
                for a, b, h in zip(reply.starts, reply.ends, reply.hashes, strict=True)
                if b > reservation.prefix_end and a < resident
            ]
            if prefix_sources and prefix_sources != canonical:
                raise StorageFailedSafe(
                    "Checkpoint prefix keys disagree with their source lease"
                )
            prefix_sources = canonical
        elif prefix_sources or resident > max(base, reservation.prefix_end):
            raise StorageFailedSafe(
                "Checkpoint prefix has no live source ownership proof"
            )
        records = []
        for entry in prefix_sources:
            count = entry["object_end"] - entry["start"]
            raw_hash = restore_hash(
                entry["hash"], self.layout.hash_width, self.layout.hash_kind
            )
            _, _, key = next(
                iter(
                    self.token_database.process_tokens(
                        hashes=[raw_hash],
                        offsets=[count],
                        request_configs=value.get("request_configs") or {},
                        kv_group=0,
                    )
                )
            )
            records.append(CheckpointPage(entry["start"], entry["end"], key))
        reservation.prefix_sources = tuple(records)
        cursor, reclaimed = base, False
        while cursor < end:
            desired_end = min(
                end, (cursor // self.config.chunk_size + 1) * self.config.chunk_size
            )
            stop = desired_end
            while stop > cursor:
                candidate = {}
                try:
                    for group in (0, 1):
                        start = max(cursor, resident) if group == 0 else cursor
                        if start >= stop:
                            continue
                        try:
                            page, _ = self.allocate_checkpoint_fragment(
                                group, stop - start
                            )
                        except MemoryError:
                            if reclaimed:
                                raise
                            reclaimed = True
                            missing = {
                                g: None for g in (group, 1) if g not in candidate
                            }
                            if not self.reclaim_checkpoint_capacity(
                                stop - cursor, missing
                            ):
                                raise
                            page, _ = self.allocate_checkpoint_fragment(
                                group, stop - start
                            )
                        candidate[group] = (start, stop, page)
                except MemoryError:
                    self._release_shared_retrieve_objs(
                        [page for _, _, page in candidate.values()], unpin=False
                    )
                    stop = cursor + (stop - cursor) // 2
                    continue
                except BaseException:
                    self._release_shared_retrieve_objs(
                        [page for _, _, page in candidate.values()], unpin=False
                    )
                    raise
                for group, fragment in candidate.items():
                    reservation.fragments.setdefault(group, []).append(fragment)
                cursor = stop
                break
            if stop != desired_end:
                break
        if cursor <= reservation.prefix_end:
            raise MemoryError("Checkpoint CPU staging allocation refused")
        reservation.captured_end = cursor

    def _reserve_store_objects(
        self, reservation: _WriteReservation, value: dict[str, Any]
    ) -> None:
        seen = set()
        for entry in value["groups"]:
            group = int(entry["group_id"])
            if group not in (0, 1) or group in seen:
                raise StorageFailedSafe("Invalid or duplicated CPU store group")
            seen.add(group)
            representation = entry.get("representation", "flat")
            if representation not in ("flat", "page", "layers"):
                raise StorageFailedSafe("Unknown CPU store representation")
            starts, ends, hashes = (
                tuple(entry["starts"]),
                tuple(entry["ends"]),
                tuple(entry["hashes"]),
            )
            if not len(starts) == len(ends) == len(hashes):
                raise StorageFailedSafe("Store chunk cardinalities disagree")
            if not starts:
                continue
            for raw_hash in hashes:
                restore_hash(raw_hash, self.layout.hash_width, self.layout.hash_kind)
            for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                if (
                    not 0 <= start < end
                    or end - start > self.config.chunk_size
                    or index
                    and starts[index] != ends[index - 1]
                ):
                    raise StorageFailedSafe("Invalid store chunk range")
            if representation == "page":
                if not self.storage_manager.supports_batched_put_layer_pages(
                    value.get("location")
                ):
                    raise StorageFailedSafe(
                        "Layer page publication is unavailable",
                        details={"fallback": "layers"},
                    )
                geometry = [
                    self._expected_shared_cpu_chunk_metadata(
                        kv_group=group,
                        num_tokens=self.config.chunk_size,
                        layer_id=layer,
                    )
                    for layer in range(self.num_layers_for_group(group))
                ]
                pages = (
                    self.checkpoint_backend().batched_allocate_layer_pages(
                        [shape for shape, _, _ in geometry],
                        [dtype for _, dtype, _ in geometry],
                        len(starts),
                        len(geometry),
                        geometry[0][2],
                        busy_loop=self.config.get_extra_config_value(
                            "force_store_wait", False
                        ),
                        valid_tokens=[
                            end - start for start, end in zip(starts, ends, strict=True)
                        ],
                        full_tokens=self.config.chunk_size,
                    )
                    if starts
                    else []
                )
                if pages is None:
                    raise StorageFailedSafe(
                        "Layer page allocation refused", details={"fallback": "layers"}
                    )
                if len(pages) != len(starts):
                    self._release_shared_retrieve_objs(pages, unpin=False)
                    raise ValueError("Layer page allocation cardinality disagrees")
                reservation.fragments[group] = list(
                    zip(starts, ends, pages, strict=True)
                )
                reservation.captured_end = max(ends, default=reservation.captured_end)
                continue
            for start, end in zip(starts, ends, strict=True):
                if representation == "layers":
                    candidate = []
                    try:
                        for layer in range(self.num_layers_for_group(group)):
                            shape, dtype, fmt = (
                                self._expected_shared_cpu_chunk_metadata(
                                    kv_group=group,
                                    num_tokens=end - start,
                                    layer_id=layer,
                                )
                            )
                            obj = self.storage_manager.allocate(
                                shape,
                                dtype,
                                fmt=fmt,
                                busy_loop=self.config.get_extra_config_value(
                                    "force_store_wait", False
                                ),
                            )
                            if obj is None:
                                break
                            obj.metadata.valid_tokens = end - start
                            candidate.append(obj)
                    except BaseException:
                        self._release_shared_retrieve_objs(candidate, unpin=False)
                        raise
                    if len(candidate) != self.num_layers_for_group(group):
                        self._release_shared_retrieve_objs(candidate, unpin=False)
                        break
                    reservation.fragments.setdefault(group, []).extend(
                        (start, end, obj) for obj in candidate
                    )
                    reservation.layer_ids.setdefault(group, []).extend(
                        range(len(candidate))
                    )
                    reservation.captured_end = max(reservation.captured_end, end)
                    continue
                shapes = self.metadata.get_shapes(end - start)
                dtypes = self.metadata.get_dtypes()
                obj = self.storage_manager.allocate(
                    shapes[group],
                    dtypes[group],
                    fmt=self._memory_format_for_kv_group(group),
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                )
                if obj is None:
                    break
                obj.metadata.valid_tokens = end - start
                reservation.fragments.setdefault(group, []).append((start, end, obj))
                reservation.captured_end = max(reservation.captured_end, end)
        if not reservation.fragments:
            raise MemoryError("Local CPU store allocation refused")

    def _checkpoint_seal_tokens(
        self, reservation: _WriteReservation, value: dict[str, Any]
    ) -> tuple[int, ...]:
        packed = "tokens_u32_le" in value or "token_count" in value
        if packed and "tokens" in value:
            raise StorageFailedSafe("Ambiguous checkpoint token representation")
        if packed:
            count = value.get("token_count")
            frame = value.get("tokens_u32_le")
            try:
                checked_uint(count)
            except ValueError as error:
                raise StorageFailedSafe("Invalid checkpoint token count") from error
            if not isinstance(frame, bytes) or len(frame) != 4 * count:
                raise StorageFailedSafe("Checkpoint tokens have the wrong byte length")
        else:
            if "tokens" not in value:
                raise StorageFailedSafe("Checkpoint token representation is missing")
            tokens = tuple(value["tokens"])
            count = len(tokens)
        if (
            not max(reservation.options["base"], reservation.prefix_end)
            < count
            <= (reservation.captured_end)
        ):
            raise StorageFailedSafe("Invalid checkpoint seal frontier")
        return unpack_uints(frame, count, 32) if packed else tokens

    def _seal_checkpoint(
        self, reservation: _WriteReservation, tokens: tuple[int, ...]
    ) -> None:
        if (
            not max(reservation.options["base"], reservation.prefix_end)
            < len(tokens)
            <= reservation.captured_end
        ):
            raise StorageFailedSafe("Invalid checkpoint seal frontier")
        if self.storage_manager.is_frozen():
            raise StorageFailedSafe(
                "Checkpoint publication refused while cache is frozen"
            )
        groups: list[list[CheckpointPage]] = [list(reservation.prefix_sources), []]
        keys, pages = [], []
        spec = SimpleNamespace(
            req_id=reservation.identity.request_id, generation=reservation.generation
        )
        for group, fragments in reservation.fragments.items():
            for start, stop, page in fragments:
                if start >= len(tokens):
                    continue
                key = self.checkpoint_page_key(spec, group, start, stop)
                groups[group].append(CheckpointPage(start, min(stop, len(tokens)), key))
                keys.append(key)
                pages.append(page)
        self.checkpoint_backend().batched_submit_layer_pages(keys[::-1], pages[::-1])
        self.checkpoint_store.publish(
            reservation.identity.request_id,
            reservation.generation,
            LocalCheckpoint(
                tokens,
                reservation.prefix_end,
                tuple(tuple(sorted(group, key=lambda p: p.start)) for group in groups),
            ),
        )

    def _commit_store(self, reservation: _WriteReservation) -> None:
        configs = reservation.options.get("request_configs") or {}
        for group, fragments in reservation.fragments.items():
            entry = next(
                item
                for item in reservation.options["groups"]
                if item["group_id"] == group
            )
            base_keys = [
                key
                for _, _, key in self.token_database.process_tokens(
                    hashes=[
                        restore_hash(h, self.layout.hash_width, self.layout.hash_kind)
                        for h in entry["hashes"]
                    ],
                    offsets=[
                        end - start
                        for start, end in zip(
                            entry["starts"], entry["ends"], strict=True
                        )
                    ],
                    request_configs=configs,
                    kv_group=group,
                )
            ]
            by_range = dict(
                zip(
                    zip(entry["starts"], entry["ends"], strict=True),
                    base_keys,
                    strict=True,
                )
            )
            keys = [by_range[start, end] for start, end, _ in fragments]
            if group in reservation.layer_ids:
                keys = [
                    key.split_layers(self.num_layers_for_group(group))[layer]
                    for key, layer in zip(
                        keys, reservation.layer_ids[group], strict=True
                    )
                ]
            objects = [obj for _, _, obj in fragments]
            for obj in objects:
                # StorageManager consumes a distinct submission reference.
                obj.ref_count_up()
            try:
                if isinstance(objects[0], LayerPageMemoryObj):
                    futures = self.storage_manager.batched_put_layer_pages(
                        keys,
                        objects,
                        location=reservation.options.get("location"),
                        req_id=reservation.identity.request_id,
                    )
                else:
                    futures = self.storage_manager.batched_put(
                        keys,
                        objects,
                        location=reservation.options.get("location"),
                        transfer_spec=reservation.options.get("transfer_spec"),
                    )
            except BaseException:
                self._release_shared_retrieve_objs(objects, unpin=False)
                raise
            for future in futures:
                future.result(timeout=self.config.blocking_timeout_secs)

    def _initialize_decoder_remote_fill(self) -> None:
        if (
            not self.config.enable_remote_lmcache_store
            or self.config.pd_role == "sender"
        ):
            return
        if not self.save_only_first_rank or not self.config.dsa_two_groups:
            raise ValueError(
                "Process RemoteFill requires replicated two-group CPU ownership"
            )
        if self._workers and not self.config.shared_cpu_cache_strict:
            raise ValueError(
                "Process RemoteFill requires strict shared CPU registration"
            )
        layout_tag, _ = mooncake_payload_layout(self.config, self.metadata)
        layout = build_decoder_layout(
            self.config,
            self.metadata,
            layout_tag=layout_tag,
            num_layers=self.num_layers,
        )
        self._validate_remote_fill_decoder_layout(layout)
        extra = self.metadata.kv_connector_extra_config or {}
        dp_rank = int(extra.get("lmcache_remote_fill_destination_dp_rank", 0))
        dp_size = int(extra.get("lmcache_remote_fill_destination_dp_size", 1))
        port = _validate_remote_fill_control_port(
            int(self.config.remote_fill_control_port_start or 19000),
            dp_rank,
            dp_size,
            internal_api_enabled=self.config.internal_api_server_enabled,
            internal_api_port_start=self.config.internal_api_server_port_start,
        )
        native_session = self.storage_manager.get_remote_fill_destination_session()
        if not native_session:
            raise ValueError("Process RemoteFill has no registered CPU native session")
        host = str(self.config.remote_fill_control_host or "0.0.0.0")
        advertised = (
            self.config.remote_fill_control_advertise_host
            or _remote_fill_advertise_host_from_session(native_session)
        )
        runtime = create_decoder_remote_fill_runtime(
            config=self.config,
            metadata=self.metadata,
            local_backend=self._shared_local_cpu_backend(),
            layout=layout,
            destination_engine_epoch=secrets.randbits(63) or 1,
            destination_dp_rank=dp_rank,
            destination_dp_size=dp_size,
            destination_remote_session=native_session,
            control_host=host,
            control_advertise_host=advertised,
            control_port=port,
            shared_cache_generation=self.shared_cpu_cache_generation,
            capacity_available=self._remote_fill_capacity_available,
            capacity_reclaimer=self._remote_fill_reclaim_capacity,
            export_address=(
                self.native_io.export_address if self.native_io is not None else None
            ),
            fatal_restart=self._remote_fill_require_paired_restart,
            global_te_push=True,
            save_only_first_rank=self.save_only_first_rank,
            save_indexer_only_first_rank=self.save_indexer_only_first_rank,
            shared_cpu_cache_strict=self.config.shared_cpu_cache_strict,
            chunk_hash_type=self.token_database.chunk_hash_type,
            chunk_hash_bytes=self.token_database.chunk_hash_bytes,
            shared_group1=not self._persistent_direct_hbm_split_group_enabled(),
        )
        try:
            runtime.start()
            if not runtime.available:
                raise RuntimeError("Process RemoteFill receiver failed during startup")
        except BaseException:
            runtime.close()
            raise
        self._remote_fill_runtime = runtime
        protect = getattr(
            self.storage_manager.backend_factory, "protect_remote_registration", None
        )
        if callable(protect):
            protect()

    def _remote_fill_require_paired_restart(
        self, transfer_ids: tuple[str, ...]
    ) -> None:
        normalized = tuple(
            sorted(set(self._remote_fill_fatal_transfers).union(transfer_ids))
        )
        if normalized:
            self._remote_fill_fatal_transfers = normalized
            self.registry.mark_fatal(
                "RemoteFill armed native transfer requires paired restart"
            )
