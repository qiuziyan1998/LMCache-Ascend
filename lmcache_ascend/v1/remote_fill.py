# SPDX-License-Identifier: Apache-2.0
"""Decoder-owned direct remote LocalCPU fill integration."""

# Standard
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import ipaddress
import json
import os
import secrets
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Protocol

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cold_start_perf import (
    cold_start_perf_enabled,
    cold_start_perf_log,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryFormat,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import (
    mooncake_valid_tokens,
    resolve_mooncake_dsa_raw_token_dims,
    resolve_remote_fill_identity,
)
from lmcache.v1.remote_fill import (
    PROTOCOL_VERSION,
    ControlPage,
    FinishRequest,
    NegotiationSpec,
    PageDisposition,
    PreparedPage,
    ProtocolLimits,
    RemoteFillRpcServer,
    RemoteFillService,
    RemoteFillStateCore,
    ReservedPageView,
    UnsafePageLifecycleError,
    content_digest,
    log_remote_fill_validation_failure,
)
from lmcache.v1.remote_fill.native import DIRECT_PUSH_H0_QUALIFICATION_V1
from lmcache.v1.rpc.zmq_transport import ZmqRouterServerTransport
from lmcache.v1.storage_backend.local_cpu_backend import (
    LayerPageAdmissionRollbackError,
)
import torch


logger = init_logger(__name__)

REMOTE_FILL_H0_QUALIFICATION_ENV = "LMCACHE_REMOTE_FILL_H0_QUALIFICATION"
REMOTE_FILL_MODEL_LAYOUT = "mla-dsa-layer-page-v3"
_DESCRIPTOR_VERIFICATION_CAPABILITY_BYTES = 32


def remote_fill_token_hash_identity(
    hash_algorithm: str,
    chunk_hash_type: type[int] | type[bytes],
    chunk_hash_bytes: int | None,
) -> str:
    """Return the exact cross-process token-hash ABI identity.

    The same vLLM algorithm name can return integers in older releases and
    raw digest bytes in newer releases.  Include that representation in the
    existing negotiation identity so mixed hash semantics fail closed.

    Args:
        hash_algorithm: Configured vLLM hash algorithm name.
        chunk_hash_type: Actual output type of the loaded hash function.
        chunk_hash_bytes: Fixed digest width for byte hashes, else ``None``.

    Returns:
        A stable identity suitable for ``NegotiationSpec.token_hash_algorithm``.

    Raises:
        TypeError: If ``chunk_hash_type`` is unsupported.
        ValueError: If the name or output contract is invalid.
    """

    algorithm = hash_algorithm.strip()
    if not algorithm:
        raise ValueError("remote-fill token hash algorithm must not be empty")
    if chunk_hash_type is int:
        if chunk_hash_bytes is not None:
            raise ValueError("integer chunk hashes must not declare a digest width")
        return algorithm if algorithm == "builtin" else f"{algorithm}:int"
    if chunk_hash_type is bytes:
        if type(chunk_hash_bytes) is not int or chunk_hash_bytes <= 0:
            raise ValueError("byte chunk hashes require a positive digest width")
        if algorithm == "builtin":
            raise ValueError("builtin token hashing must produce integers")
        return f"{algorithm}:bytes:{chunk_hash_bytes}"
    raise TypeError("chunk_hash_type must be int or bytes")


def _format_tcp_host(host: str) -> str:
    """Bracket literal IPv6 addresses for ZeroMQ TCP endpoints."""

    host = host.strip()
    normalized = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return (
            f"[{normalized}]"
            if ipaddress.ip_address(normalized).version == 6
            else normalized
        )
    except ValueError:
        return host


def _validate_advertised_control_host(host: str) -> str:
    """Reject bind-only or local-only addresses from placement metadata."""

    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host or host in {"*", "0.0.0.0", "::", "localhost"}:
        raise ValueError(
            "remote-fill advertised control host must be a routable address"
        )
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.is_unspecified or address.is_loopback:
        raise ValueError(
            "remote-fill advertised control host must not be wildcard or loopback"
        )
    return host


def _log_remote_fill_event(event: str, **fields: object) -> None:
    """Emit one low-cardinality decoder lifecycle event."""

    logger.info(
        "[LMCACHE_REMOTE_FILL] %s",
        json.dumps(
            {"schema": 1, "event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class RemoteFillLocalBackend(Protocol):
    """Public LocalCPU operations required by remote fill."""

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Return whether ``key`` is currently committed."""

    def batched_allocate_layer_pages(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        batch_size: int,
        num_layers: int,
        fmt: MemoryFormat,
        busy_loop: bool = True,
        valid_tokens: int | list[int] | None = None,
        full_tokens: int | None = None,
        eviction: bool = True,
    ) -> list[LayerPageMemoryObj] | None:
        """Allocate hidden physical layer pages."""

    def commit_external_two_group_prefix_if_absent(
        self,
        required_group0_keys: Sequence[CacheEngineKey],
        required_group1_keys: Sequence[CacheEngineKey],
        ready_reservations: Mapping[CacheEngineKey, LayerPageMemoryObj],
    ) -> Any:
        """Atomically publish exact complete two-group coverage."""


class RemoteFillServer(Protocol):
    """Server loop boundary used by the decoder service host."""

    def serve_once(self) -> bool:
        """Serve at most one control request."""

    def close(self) -> None:
        """Close the server transport."""


@dataclass(frozen=True, slots=True)
class RemoteFillGroupLayout:
    """One decoder LocalCPU page layout."""

    kv_group: int
    raw_token_dim: int
    dtype: torch.dtype
    fmt: MemoryFormat

    def full_shape(self, chunk_size: int) -> torch.Size:
        """Return the flat full-page shape for one layer."""

        return torch.Size([chunk_size * self.raw_token_dim])

    def expected_bytes(self, valid_tokens: int, num_layers: int) -> int:
        """Return exact bytes for one physical all-layer page."""

        return valid_tokens * self.raw_token_dim * self.dtype.itemsize * num_layers


@dataclass(frozen=True, slots=True)
class RemoteFillDecoderLayout:
    """Static decoder layout validated once at startup."""

    layout_tag: str
    cache_namespace_tag: str
    model_artifact_id: str
    model_name: str
    key_world_size: int
    chunk_size: int
    num_layers: int
    groups: tuple[RemoteFillGroupLayout, RemoteFillGroupLayout]

    def group(self, kv_group: int) -> RemoteFillGroupLayout:
        """Return layout for ``kv_group``.

        Raises:
            ValueError: If the group is not 0 or 1.
        """

        if kv_group not in (0, 1):
            raise ValueError(f"remote fill requires KV group 0 or 1, got {kv_group}")
        return self.groups[kv_group]

    @property
    def group_dimensions(self) -> tuple[int, int]:
        """Return the startup negotiation dimensions."""

        return (self.groups[0].raw_token_dim, self.groups[1].raw_token_dim)


@dataclass(frozen=True, slots=True)
class RemoteFillPlacementInfo:
    """Pointer-free decoder capability advertised to the routing layer."""

    enabled: bool
    destination_engine_id: str
    destination_engine_epoch: int
    control_endpoint: str
    shared_cache_generation: int
    protocol_version: int
    layout_tag: str
    cache_namespace_tag: str
    model_artifact_id: str
    tp_rank: int
    dp_rank: int
    destination_tp_size: int
    destination_dp_size: int
    global_te_push: bool
    token_hash_algorithm: str
    python_hash_seed: str
    descriptor_verification_capability: str = field(repr=False)

    def to_dict(self) -> dict[str, int | str | bool]:
        """Return a transport-safe discovery record."""

        return {
            "enabled": self.enabled,
            "destination_engine_id": self.destination_engine_id,
            "destination_engine_epoch": self.destination_engine_epoch,
            "control_endpoint": self.control_endpoint,
            "shared_cache_generation": self.shared_cache_generation,
            "protocol_version": self.protocol_version,
            "layout_tag": self.layout_tag,
            "cache_namespace_tag": self.cache_namespace_tag,
            "model_artifact_id": self.model_artifact_id,
            "tp_rank": self.tp_rank,
            "dp_rank": self.dp_rank,
            "destination_tp_size": self.destination_tp_size,
            "destination_dp_size": self.destination_dp_size,
            "global_te_push": self.global_te_push,
            "token_hash_algorithm": self.token_hash_algorithm,
            "python_hash_seed": self.python_hash_seed,
            "descriptor_verification_capability": (
                self.descriptor_verification_capability
            ),
        }


@dataclass(slots=True)
class _HiddenPage:
    """Exactly-once owner for one hidden, pinned LocalCPU page."""

    key: CacheEngineKey
    page: LayerPageMemoryObj
    unpinned: bool = False
    reference_released: bool = False
    released: bool = False
    lock: Lock = field(default_factory=Lock)

    def release(self) -> None:
        """Release the reservation pin and allocator-owned reference once."""

        with self.lock:
            if self.released:
                return
            if not self.unpinned:
                self.page.unpin()
                self.unpinned = True
            if not self.reference_released:
                self.page.ref_count_down()
                self.reference_released = True
            self.released = True


def _parse_raw_token_dimensions(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
) -> tuple[int, int]:
    resolved, source = resolve_mooncake_dsa_raw_token_dims(config, metadata)
    dimensions = (int(resolved.get(0, 0)), int(resolved.get(1, 0)))
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "remote-fill raw token dimensions could not be resolved: " + source
        )
    return dimensions


def build_decoder_layout(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    *,
    layout_tag: str,
    num_layers: int,
) -> RemoteFillDecoderLayout:
    """Build and validate the startup-only two-group LocalCPU layout.

    Args:
        config: Resolved LMCache configuration.
        metadata: Decoder model and rank metadata.
        layout_tag: Startup-computed payload ABI tag.
        num_layers: Cache-bearing layer count, including MTP when configured.

    Returns:
        An immutable two-group decoder layout.

    Raises:
        ValueError: If required identity or layout metadata is missing.
    """

    dimensions = _parse_raw_token_dimensions(config, metadata)
    namespace, artifact_id = resolve_remote_fill_identity(config, metadata)
    if not layout_tag or not artifact_id or not namespace:
        raise ValueError("remote-fill layout identity must not be empty")
    if num_layers <= 0:
        raise ValueError("remote-fill layer count must be positive")
    dtypes = metadata.get_dtypes()
    group0_dtype = dtypes[0] if dtypes else metadata.kv_dtype
    group1_dtype = dtypes[1] if len(dtypes) > 1 else group0_dtype
    return RemoteFillDecoderLayout(
        layout_tag=layout_tag,
        cache_namespace_tag=namespace,
        model_artifact_id=artifact_id,
        model_name=metadata.model_name,
        key_world_size=(
            1
            if bool(
                config.get_extra_config_value(
                    "save_only_first_rank",
                    metadata.use_mla,
                )
            )
            and metadata.use_mla
            else int(metadata.world_size)
        ),
        chunk_size=int(config.chunk_size),
        num_layers=num_layers,
        groups=(
            RemoteFillGroupLayout(
                kv_group=0,
                raw_token_dim=dimensions[0],
                dtype=group0_dtype,
                fmt=MemoryFormat.KV_MLA_LATENT_FMT,
            ),
            RemoteFillGroupLayout(
                kv_group=1,
                raw_token_dim=dimensions[1],
                dtype=group1_dtype,
                fmt=MemoryFormat.KV_DSA_INDEX_FMT,
            ),
        ),
    )


def _descriptor_verification_capability(value: bytes | None) -> bytes:
    """Return one exact decoder-incarnation descriptor verification key."""

    capability = (
        secrets.token_bytes(_DESCRIPTOR_VERIFICATION_CAPABILITY_BYTES)
        if value is None
        else value
    )
    if (
        not isinstance(capability, bytes)
        or len(capability) != _DESCRIPTOR_VERIFICATION_CAPABILITY_BYTES
    ):
        raise ValueError("remote-fill verification capability must contain 32 bytes")
    return capability


def build_remote_fill_negotiation_spec(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    *,
    layout: RemoteFillDecoderLayout,
    destination_engine_id: str,
    destination_dp_rank: int,
    dp_size: int,
    global_te_push: bool,
    destination_remote_session: str,
    chunk_hash_type: type[int] | type[bytes],
    chunk_hash_bytes: int | None,
) -> NegotiationSpec:
    """Build the byte-identical source/destination negotiation identity.

    Args:
        config: Resolved LMCache feature configuration.
        metadata: Model and tensor-parallel metadata.
        layout: Validated two-group page layout.
        destination_engine_id: Selected decoder engine identity.
        destination_dp_rank: Selected decoder data-parallel rank.
        dp_size: Deployment data-parallel size.
        global_te_push: Whether the native transport premise is activated.
        destination_remote_session: Decoder GlobalTE session identifier.
        chunk_hash_type: Actual output type of the decoder hash function.
        chunk_hash_bytes: Fixed digest width for byte hashes.

    Returns:
        The public core protocol negotiation descriptor.

    Raises:
        ValueError: If a destination or hashing identity is invalid.
    """

    engine_id = destination_engine_id.strip()
    remote_session = destination_remote_session.strip()
    if not engine_id or not remote_session:
        raise ValueError("remote-fill decoder engine and native session are required")
    if dp_size <= 0 or not 0 <= destination_dp_rank < dp_size:
        raise ValueError("remote-fill destination DP rank is out of range")
    hash_algorithm = str(config.pre_caching_hash_algorithm).strip()
    hash_identity = remote_fill_token_hash_identity(
        hash_algorithm,
        chunk_hash_type,
        chunk_hash_bytes,
    )
    python_hash_seed = (
        os.getenv("PYTHONHASHSEED", "") if hash_algorithm == "builtin" else ""
    )
    if hash_algorithm == "builtin" and not python_hash_seed:
        raise ValueError("builtin token hashing requires an explicit PYTHONHASHSEED")
    return NegotiationSpec(
        cache_namespace_tag=layout.cache_namespace_tag,
        layout_tag=layout.layout_tag,
        model_artifact_id=layout.model_artifact_id,
        chunk_size=layout.chunk_size,
        model_layout=REMOTE_FILL_MODEL_LAYOUT,
        group_dimensions=layout.group_dimensions,
        layer_count=layout.num_layers,
        save_only_first_rank=True,
        shared_group1=True,
        tp_size=int(metadata.world_size),
        dp_size=dp_size,
        global_te_push=global_te_push,
        destination_engine_id=engine_id,
        destination_dp_rank=destination_dp_rank,
        destination_remote_session=remote_session,
        token_hash_algorithm=hash_identity,
        python_hash_seed=python_hash_seed,
    )


def remote_fill_h0_qualified(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this deployment explicitly activated the H0 contract.

    Args:
        environment: Environment mapping, defaulting to ``os.environ``.

    Returns:
        ``True`` only for the exact versioned native qualification token.
    """

    env = os.environ if environment is None else environment
    return (
        env.get(REMOTE_FILL_H0_QUALIFICATION_ENV, "") == DIRECT_PUSH_H0_QUALIFICATION_V1
    )


class AscendRemoteFillPageLifecycle:
    """Allocate hidden TP0 pages and atomically publish exact coverage."""

    def __init__(
        self,
        *,
        local_backend: RemoteFillLocalBackend,
        layout: RemoteFillDecoderLayout,
        capacity_available: Callable[[int], bool],
        chunk_hash_type: type[int] | type[bytes],
        chunk_hash_bytes: int | None,
        pin_pages: Callable[[Sequence[LayerPageMemoryObj]], bool] = (
            LayerPageMemoryObj.pin_many
        ),
    ) -> None:
        """Create a decoder page lifecycle.

        Args:
            local_backend: Rank0 LocalCPU backend public interface.
            layout: Validated static two-group page layout.
            capacity_available: Nonblocking ordinary-cache headroom check.
            chunk_hash_type: Exact chunk-hash representation negotiated with
                the producer.
            chunk_hash_bytes: Fixed digest width when ``chunk_hash_type`` is
                ``bytes``; otherwise ``None``.
            pin_pages: Atomic pin primitive, injectable for tests.

        Raises:
            TypeError: If ``chunk_hash_type`` is neither ``int`` nor ``bytes``.
            ValueError: If the hash type and byte width are inconsistent.
        """

        if chunk_hash_type not in (int, bytes):
            raise TypeError("chunk_hash_type must be int or bytes")
        if chunk_hash_type is bytes:
            if type(chunk_hash_bytes) is not int or chunk_hash_bytes <= 0:
                raise ValueError(
                    "byte chunk hashes require a positive fixed digest width"
                )
        elif chunk_hash_bytes is not None:
            raise ValueError("integer chunk hashes must not declare a digest width")
        self._local = local_backend
        self._layout = layout
        self._capacity_available = capacity_available
        self._chunk_hash_type = chunk_hash_type
        self._chunk_hash_bytes = chunk_hash_bytes
        self._pin_pages = pin_pages

    def prepare_pages(
        self,
        transfer_id: str,
        window_id: int,
        pages: tuple[ControlPage, ...],
        reserve_missing: bool,
    ) -> tuple[PreparedPage, ...]:
        """Snapshot existing pages and allocate only absent destinations.

        Existing pages are deliberately not pinned. They are planning-time
        snapshots and are rechecked by the atomic final commit.

        Args:
            transfer_id: Request-attempt transfer identity.
            window_id: Bounded manifest window identity.
            pages: Complete two-group control pages in logical order.
            reserve_missing: Whether absent pages may be allocated.

        Returns:
            One preparation result per input page, in input order.

        Raises:
            ValueError: If page metadata is not canonical or layout-compatible.
        """

        del transfer_id, window_id
        keys = tuple(self._validate_control_page(page) for page in pages)
        selectors = tuple((page.kv_group, page.chunk_index) for page in pages)
        if len(set(keys)) != len(keys) or len(set(selectors)) != len(selectors):
            raise ValueError("remote-fill page manifest contains duplicate pages")
        self._validate_window_pairs(pages, keys)
        results: list[PreparedPage | None] = [None] * len(pages)

        def finalize_results() -> tuple[PreparedPage, ...]:
            if any(result is None for result in results):
                raise RuntimeError("remote-fill page preparation omitted an input page")
            prepared = tuple(result for result in results if result is not None)
            _log_remote_fill_event(
                "remote_fill_reserve",
                pages=len(prepared),
                allocated=sum(
                    result.disposition is PageDisposition.ALLOCATED
                    for result in prepared
                ),
                existing=sum(
                    result.disposition is PageDisposition.EXISTING
                    for result in prepared
                ),
                missing=sum(
                    result.disposition is PageDisposition.MISSING
                    for result in prepared
                ),
                reserved_bytes=sum(
                    result.destination_length
                    for result in prepared
                    if result.disposition is PageDisposition.ALLOCATED
                ),
                reserve_missing=reserve_missing,
            )
            return prepared

        missing_indices: list[int] = []
        for index, key in enumerate(keys):
            if self._local.contains(key, pin=False):
                results[index] = PreparedPage(
                    handle=None,
                    disposition=PageDisposition.EXISTING,
                )
            else:
                missing_indices.append(index)

        if not reserve_missing:
            for index in missing_indices:
                results[index] = PreparedPage(
                    handle=None,
                    disposition=PageDisposition.MISSING,
                )
            return finalize_results()
        requested_bytes = sum(pages[index].expected_bytes for index in missing_indices)
        if requested_bytes and not self._capacity_available(requested_bytes):
            for index in missing_indices:
                results[index] = PreparedPage(
                    handle=None,
                    disposition=PageDisposition.MISSING,
                )
            return finalize_results()

        allocated: dict[int, LayerPageMemoryObj] = {}
        pinned = False
        try:
            for kv_group in (0, 1):
                indices = [
                    index
                    for index in missing_indices
                    if pages[index].kv_group == kv_group
                ]
                if not indices:
                    continue
                group_layout = self._layout.group(kv_group)
                group_pages = self._local.batched_allocate_layer_pages(
                    [group_layout.full_shape(self._layout.chunk_size)],
                    [group_layout.dtype],
                    len(indices),
                    self._layout.num_layers,
                    group_layout.fmt,
                    busy_loop=False,
                    valid_tokens=[pages[index].valid_tokens for index in indices],
                    full_tokens=self._layout.chunk_size,
                    eviction=False,
                )
                if group_pages is None or len(group_pages) != len(indices):
                    if group_pages:
                        self._release_unpinned(group_pages)
                    raise MemoryError("remote-fill hidden page allocation failed")
                allocated.update(zip(indices, group_pages, strict=True))
            allocated_pages = list(allocated.values())
            if allocated_pages and not self._pin_pages(allocated_pages):
                raise MemoryError("remote-fill hidden page pin failed")
            pinned = bool(allocated_pages)
            for index, page in allocated.items():
                control = pages[index]
                if page.get_size() != control.expected_bytes:
                    raise ValueError("remote-fill allocation byte count mismatch")
                handle = _HiddenPage(keys[index], page)
                results[index] = PreparedPage(
                    handle=handle,
                    destination_ptr=page.data_ptr,
                    destination_length=page.get_size(),
                    reservation_base=page.data_ptr,
                    reservation_length=page.get_size(),
                    disposition=PageDisposition.ALLOCATED,
                )
        except Exception:
            self._release_allocated(tuple(allocated.values()), pinned=pinned)
            raise
        try:
            return finalize_results()
        except Exception:
            self._release_allocated(tuple(allocated.values()), pinned=pinned)
            raise

    def release_prepared_pages(
        self,
        pages: tuple[PreparedPage, ...],
        reason: str,
    ) -> None:
        """Release allocated pages rejected before service association.

        Args:
            pages: Raw preparation results not installed in service state.
            reason: Low-cardinality lifecycle reason for diagnostics.

        Raises:
            TypeError: If an allocated result lacks the decoder-owned handle.
        """

        invalid_handles = 0
        released_pages = 0
        released_bytes = 0
        for prepared in pages:
            if prepared.disposition is not PageDisposition.ALLOCATED:
                continue
            handle = prepared.handle
            if not isinstance(handle, _HiddenPage):
                invalid_handles += 1
            else:
                released_pages += 1
                released_bytes += handle.page.get_size()
                handle.release()
        _log_remote_fill_event(
            "remote_fill_release",
            reason=reason,
            pages=released_pages,
            released_bytes=released_bytes,
            invalid_handles=invalid_handles,
            associated=False,
        )
        if invalid_handles:
            raise TypeError(
                "remote-fill prepared pages contain "
                f"{invalid_handles} invalid allocated handles"
            )

    def release_pages(
        self,
        pages: tuple[ReservedPageView, ...],
        reason: str,
    ) -> None:
        """Release allocated hidden pages exactly once.

        Args:
            pages: Prepared page views owned by the service state.
            reason: Low-cardinality lifecycle reason for diagnostics.

        Raises:
            TypeError: If an allocated page lacks the decoder-owned handle.
        """

        invalid_handles = 0
        released_pages = 0
        released_bytes = 0
        for view in pages:
            if view.prepared.disposition is not PageDisposition.ALLOCATED:
                continue
            handle = view.prepared.handle
            if not isinstance(handle, _HiddenPage):
                invalid_handles += 1
            else:
                released_pages += 1
                released_bytes += handle.page.get_size()
                handle.release()
        _log_remote_fill_event(
            "remote_fill_release",
            reason=reason,
            pages=released_pages,
            released_bytes=released_bytes,
            invalid_handles=invalid_handles,
            associated=True,
        )
        if invalid_handles:
            raise TypeError(
                "remote-fill reserved pages contain "
                f"{invalid_handles} invalid allocated handles"
            )

    def commit_pages(
        self,
        transfer_id: str,
        required_pages: tuple[ControlPage, ...],
        pages: tuple[ReservedPageView, ...],
        finish: FinishRequest,
    ) -> bool:
        """Atomically publish a complete exact two-group prefix.

        Args:
            transfer_id: Transaction identity used only for diagnostics.
            required_pages: Every authoritative page through required store end.
            pages: Existing snapshots and terminal-success hidden allocations.
            finish: Final persistent and token-coverage evidence.

        Returns:
            ``True`` only when the LocalCPU atomic helper committed complete
            two-group coverage. On success, all hidden reservation references
            are released after cache ownership has been acquired.
        """

        commit_started = monotonic()

        def log_commit(**fields: object) -> None:
            payload = {
                **fields,
                "elapsed_ms": round((monotonic() - commit_started) * 1000, 3),
            }
            _log_remote_fill_event("remote_fill_local_commit", **payload)
            cold_start_perf_log(
                logger,
                "remote_fill_local_commit",
                transfer_id=transfer_id,
                **payload,
            )

        group0_keys, group1_keys = self._validate_required_prefix(
            required_pages,
            finish,
        )
        ready: dict[CacheEngineKey, LayerPageMemoryObj] = {}
        handles: list[_HiddenPage] = []
        for view in pages:
            if view.prepared.disposition is not PageDisposition.ALLOCATED:
                continue
            handle = view.prepared.handle
            if not isinstance(handle, _HiddenPage):
                raise TypeError("remote-fill allocated page has an invalid handle")
            key = self._validate_control_page(view.control_page)
            if key != handle.key or key in ready:
                raise ValueError("remote-fill ready reservation identity mismatch")
            ready[key] = handle.page
            handles.append(handle)
        _log_remote_fill_event(
            "remote_fill_finish",
            required_store_end=finish.required_store_end,
            persistent_common_end=finish.persistent_common_end,
            required_pairs=len(group0_keys),
            ready_pages=len(ready),
        )
        try:
            result = self._local.commit_external_two_group_prefix_if_absent(
                group0_keys,
                group1_keys,
                ready,
            )
        except LayerPageAdmissionRollbackError as error:
            log_commit(
                committed=False,
                error=True,
                rollback_safe=False,
                required_pairs=len(group0_keys),
                ready_pages=len(ready),
            )
            raise UnsafePageLifecycleError(
                "remote-fill LocalCPU admission rollback was incomplete"
            ) from error
        except Exception:
            log_commit(
                committed=False,
                error=True,
                rollback_safe=True,
                required_pairs=len(group0_keys),
                ready_pages=len(ready),
            )
            raise
        if not bool(result.committed):
            log_commit(
                committed=False,
                error=False,
                required_pairs=len(group0_keys),
                ready_pages=len(ready),
                lock_wait_ms=round(result.lock_wait_seconds * 1000, 3),
                lock_hold_ms=round(result.lock_hold_seconds * 1000, 3),
            )
            return False
        try:
            for handle in handles:
                handle.release()
        except Exception as error:
            log_commit(
                committed=True,
                error=True,
                rollback_safe=False,
                required_pairs=len(group0_keys),
                ready_pages=len(ready),
            )
            raise UnsafePageLifecycleError(
                "remote-fill reservation ownership transfer failed after publish"
            ) from error
        diagnostic_fields: dict[str, object] = {
            "inserted_keys": len(result.inserted_keys),
            "existing_keys": len(result.existing_keys),
        }
        retention_trace_id = getattr(result, "retention_trace_id", None)
        if isinstance(retention_trace_id, int):
            diagnostic_fields["retention_trace_id"] = retention_trace_id
        if cold_start_perf_enabled():
            try:
                diagnostic_fields["required_key_digest"] = content_digest(
                    (
                        tuple(key.to_string() for key in group0_keys),
                        tuple(key.to_string() for key in group1_keys),
                    )
                )
            except Exception:
                # Publication and ownership transfer are already complete;
                # observability must never turn that success into a failure.
                diagnostic_fields["key_digest_error"] = True
                logger.warning(
                    "Failed to fingerprint remote-fill commit keys",
                    exc_info=True,
                )
        log_commit(
            committed=True,
            error=False,
            required_pairs=len(group0_keys),
            ready_pages=len(ready),
            published_bytes=sum(page.get_size() for page in ready.values()),
            lock_wait_ms=round(result.lock_wait_seconds * 1000, 3),
            lock_hold_ms=round(result.lock_hold_seconds * 1000, 3),
            **diagnostic_fields,
        )
        return True

    def _validate_control_page(self, page: ControlPage) -> CacheEngineKey:
        key = CacheEngineKey.from_string(
            page.canonical_key,
            chunk_hash_type=self._chunk_hash_type,
        )
        if (
            self._chunk_hash_type is bytes
            and len(key.chunk_hash) != self._chunk_hash_bytes
        ):
            raise ValueError(
                "remote-fill page chunk hash has an unexpected digest width"
            )
        if key.to_string() != page.canonical_key:
            raise ValueError("remote-fill page key is not canonical")
        if key.kv_group != page.kv_group:
            raise ValueError("remote-fill key group does not match control page")
        if key.worker_id != 0:
            raise ValueError("remote-fill physical page must be owned by TP0")
        if key.model_name != self._layout.model_name:
            raise ValueError("remote-fill page model identity mismatch")
        if key.world_size != self._layout.key_world_size:
            raise ValueError("remote-fill page key world size mismatch")
        if page.destination_tp_rank != 0:
            raise ValueError("remote-fill destination TP rank must be zero")
        if page.layer_count != self._layout.num_layers:
            raise ValueError("remote-fill page layer count mismatch")
        if page.layout_tag != self._layout.layout_tag:
            raise ValueError("remote-fill page layout tag mismatch")
        if page.chunk_index < 0 or page.chunk_start != (
            page.chunk_index * self._layout.chunk_size
        ):
            raise ValueError("remote-fill page chunk position mismatch")
        if page.chunk_end - page.chunk_start != page.valid_tokens:
            raise ValueError("remote-fill page valid-token interval mismatch")
        if not 0 < page.valid_tokens <= self._layout.chunk_size:
            raise ValueError("remote-fill page valid-token count is out of range")
        if mooncake_valid_tokens(key, self._layout.chunk_size) != page.valid_tokens:
            raise ValueError("remote-fill key valid-token tag mismatch")
        group = self._layout.group(page.kv_group)
        if key.dtype != group.dtype:
            raise ValueError("remote-fill page key dtype mismatch")
        expected_bytes = group.expected_bytes(
            page.valid_tokens,
            self._layout.num_layers,
        )
        if page.expected_bytes != expected_bytes:
            raise ValueError("remote-fill page expected byte count mismatch")
        return key

    def _validate_required_prefix(
        self,
        pages: tuple[ControlPage, ...],
        finish: FinishRequest,
    ) -> tuple[tuple[CacheEngineKey, ...], tuple[CacheEngineKey, ...]]:
        if finish.required_store_end < 0:
            raise ValueError("remote-fill required store end must be nonnegative")
        if finish.required_store_end == 0:
            if pages or finish.final_partial_valid_tokens:
                raise ValueError("remote-fill empty prefix metadata is inconsistent")
            return (), ()
        expected_pairs = (
            finish.required_store_end + self._layout.chunk_size - 1
        ) // self._layout.chunk_size
        if len(pages) != expected_pairs * 2:
            raise ValueError("remote-fill required page manifest is incomplete")
        by_selector: dict[tuple[int, int], tuple[ControlPage, CacheEngineKey]] = {}
        for page in pages:
            key = self._validate_control_page(page)
            selector = (page.chunk_index, page.kv_group)
            if selector in by_selector:
                raise ValueError("remote-fill required page manifest is duplicated")
            by_selector[selector] = (page, key)
        group0: list[CacheEngineKey] = []
        group1: list[CacheEngineKey] = []
        for chunk_index in range(expected_pairs):
            pair = [by_selector.get((chunk_index, kv_group)) for kv_group in (0, 1)]
            if any(item is None for item in pair):
                raise ValueError("remote-fill required page manifest has a hole")
            (page0, key0), (page1, key1) = pair  # type: ignore[misc]
            expected_end = min(
                finish.required_store_end,
                (chunk_index + 1) * self._layout.chunk_size,
            )
            expected_valid = expected_end - chunk_index * self._layout.chunk_size
            if (
                page0.valid_tokens != expected_valid
                or page1.valid_tokens != expected_valid
                or page0.chunk_start != page1.chunk_start
                or page0.chunk_end != page1.chunk_end
                or key0.chunk_hash != key1.chunk_hash
                or key0.model_name != key1.model_name
                or key0.world_size != key1.world_size
                or key0.worker_id != key1.worker_id
                or self._shared_key_tags(key0) != self._shared_key_tags(key1)
            ):
                raise ValueError("remote-fill two-group page identity mismatch")
            group0.append(key0)
            group1.append(key1)
        final_valid = finish.required_store_end % self._layout.chunk_size
        if finish.final_partial_valid_tokens != final_valid:
            raise ValueError("remote-fill final partial-page metadata mismatch")
        return tuple(group0), tuple(group1)

    def _validate_window_pairs(
        self,
        pages: tuple[ControlPage, ...],
        keys: tuple[CacheEngineKey, ...],
    ) -> None:
        by_selector = {
            (page.chunk_index, page.kv_group): (page, key)
            for page, key in zip(pages, keys, strict=True)
        }
        chunk_indices = sorted({page.chunk_index for page in pages})
        if not chunk_indices:
            raise ValueError("remote-fill page window must not be empty")
        if chunk_indices != list(range(chunk_indices[0], chunk_indices[-1] + 1)):
            raise ValueError("remote-fill page window must be contiguous")
        for chunk_index in chunk_indices:
            group0 = by_selector.get((chunk_index, 0))
            group1 = by_selector.get((chunk_index, 1))
            if group0 is None or group1 is None:
                raise ValueError("remote-fill page window requires both groups")
            page0, key0 = group0
            page1, key1 = group1
            if (
                page0.chunk_start != page1.chunk_start
                or page0.chunk_end != page1.chunk_end
                or page0.valid_tokens != page1.valid_tokens
                or key0.chunk_hash != key1.chunk_hash
                or self._shared_key_tags(key0) != self._shared_key_tags(key1)
            ):
                raise ValueError("remote-fill window group identities do not match")

    @staticmethod
    def _shared_key_tags(key: CacheEngineKey) -> tuple:
        return tuple(tag for tag in (key.tags or ()) if tag[0] != "dsa_idx")

    @staticmethod
    def _release_unpinned(pages: Sequence[LayerPageMemoryObj]) -> None:
        for page in pages:
            page.ref_count_down()

    @staticmethod
    def _release_allocated(
        pages: Sequence[LayerPageMemoryObj],
        *,
        pinned: bool,
    ) -> None:
        for page in pages:
            if pinned:
                page.unpin()
            page.ref_count_down()


class DecoderRemoteFillServiceHost:
    """Run the bounded RemoteFillService on one decoder TP0 thread."""

    def __init__(
        self,
        *,
        service: RemoteFillService,
        server: RemoteFillServer,
        fatal_restart: Callable[[tuple[str, ...]], None],
        thread_name: str = "lmcache-remote-fill-tp0",
    ) -> None:
        """Create a stopped service host.

        Args:
            service: Hardware-independent remote-fill service.
            server: Existing LMCache RPC server adapter.
            fatal_restart: Callback for armed-timeout paired restart evidence.
            thread_name: Diagnostic thread name.
        """

        self._service = service
        self._server = server
        self._fatal_restart = fatal_restart
        self._stop = Event()
        self._closed = Event()
        self._fatal_reported = False
        self._start_lock = Lock()
        self._started = False
        self._last_metrics = self._read_metrics()
        self._next_idle_metrics_at = monotonic() + 1.0
        self._thread = Thread(
            target=self._run,
            daemon=True,
            name=thread_name,
        )

    def start(self) -> None:
        """Start the service thread exactly once."""

        with self._start_lock:
            if self._started:
                if self._closed.is_set():
                    raise RuntimeError("remote-fill service host is already closed")
                return
            if self._closed.is_set():
                raise RuntimeError("remote-fill service host is already closed")
            self._started = True
            try:
                self._thread.start()
            except BaseException:
                self._started = False
                self._shutdown_service_and_server()
                raise

    @property
    def available(self) -> bool:
        """Return whether the control service can still accept requests."""

        with self._start_lock:
            return bool(
                self._started
                and not self._stop.is_set()
                and not self._closed.is_set()
                and self._thread.is_alive()
            )

    def close(self, timeout: float = 5.0) -> None:
        """Stop the service after its bounded receive interval.

        Args:
            timeout: Maximum seconds to wait for the server thread.

        Raises:
            TimeoutError: If the server does not stop within ``timeout``.
        """

        with self._start_lock:
            self._stop.set()
            started = self._started
            if not started:
                self._shutdown_service_and_server()
        if started:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("remote-fill service thread did not stop")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                served = self._server.serve_once()
                fatal = self._service.run_maintenance()
                self._emit_metrics_if_changed(
                    force=bool(served or fatal),
                )
                if fatal and not self._fatal_reported:
                    self._fatal_reported = True
                    self._fatal_restart(fatal)
                    self._stop.set()
        except Exception as error:
            log_remote_fill_validation_failure(
                logger,
                code="RF-D-005",
                stage="decoder_control_service",
                action="DISABLE_DIRECT_SERVICE_AND_AUDIT_ARMED_STATE",
                reason=(
                    "the endpoint is no longer advertised; ordinary LMCache "
                    "serving continues only if shutdown finds no armed transfer"
                ),
                error=error,
                severity="error",
            )
            self._stop.set()
        finally:
            self._shutdown_service_and_server()

    def _shutdown_service_and_server(self) -> None:
        if self._closed.is_set():
            return
        try:
            fatal = tuple(self._service.shutdown())
            if fatal and not self._fatal_reported:
                self._fatal_reported = True
                self._fatal_restart(fatal)
        finally:
            self._server.close()
            self._closed.set()

    def _read_metrics(self) -> dict[str, int] | None:
        snapshot = getattr(self._service, "metrics_snapshot", None)
        if not callable(snapshot):
            return None
        return {
            str(name): int(value)
            for name, value in asdict(snapshot()).items()
        }

    def _emit_metrics_if_changed(self, *, force: bool) -> None:
        now = monotonic()
        if not force and now < self._next_idle_metrics_at:
            return
        self._next_idle_metrics_at = now + 1.0
        current = self._read_metrics()
        previous = self._last_metrics
        self._last_metrics = current
        if current is None or previous is None:
            return
        changes: dict[str, int] = {}
        for name, value in current.items():
            old_value = previous.get(name, 0)
            if name.endswith("_total"):
                delta = value - old_value
                if delta:
                    changes[f"{name}_delta"] = delta
            elif value != old_value:
                changes[name] = value
        if not changes:
            return
        _log_remote_fill_event(
            "remote_fill_metrics",
            **changes,
        )
        submitted = changes.get("submitted_bytes_total_delta", 0)
        if submitted > 0:
            _log_remote_fill_event(
                "remote_fill_arm",
                submitted_bytes=submitted,
            )


@dataclass(slots=True)
class DecoderRemoteFillRuntime:
    """Decoder service state and pointer-free discovery information."""

    lifecycle: AscendRemoteFillPageLifecycle
    state: RemoteFillStateCore
    service: RemoteFillService
    host: DecoderRemoteFillServiceHost
    placement: RemoteFillPlacementInfo

    def start(self) -> None:
        """Start the control service."""

        self.host.start()

    @property
    def available(self) -> bool:
        """Return whether decoder placement may still be advertised."""

        return self.host.available

    def close(self) -> None:
        """Drain safe reservations and stop the control service."""

        self.host.close()

    def metrics_snapshot(self) -> dict[str, int]:
        """Return fixed-cardinality decoder protocol metrics."""

        return {
            str(name): int(value)
            for name, value in asdict(self.service.metrics_snapshot()).items()
        }


def create_decoder_remote_fill_runtime(
    *,
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    local_backend: RemoteFillLocalBackend,
    layout: RemoteFillDecoderLayout,
    destination_engine_epoch: int,
    destination_dp_rank: int,
    destination_dp_size: int,
    destination_remote_session: str,
    control_host: str,
    control_advertise_host: str | None = None,
    control_port: int,
    shared_cache_generation: int,
    capacity_available: Callable[[int], bool],
    fatal_restart: Callable[[tuple[str, ...]], None],
    global_te_push: bool,
    save_only_first_rank: bool,
    save_indexer_only_first_rank: bool,
    shared_cpu_cache_strict: bool,
    chunk_hash_type: type[int] | type[bytes],
    chunk_hash_bytes: int | None,
    descriptor_verification_capability: bytes | None = None,
    server_factory: Callable[[RemoteFillService], RemoteFillServer] | None = None,
) -> DecoderRemoteFillRuntime:
    """Create a TP0 decoder runtime without starting its thread.

    Args:
        config: Resolved default-off feature configuration.
        metadata: Decoder TP0 metadata.
        local_backend: Rank0 shared LocalCPU backend.
        layout: Validated two-group page layout.
        destination_engine_epoch: Fresh decoder-process epoch.
        destination_dp_rank: Selected data-parallel rank.
        destination_dp_size: Deployment data-parallel size.
        destination_remote_session: Registered decoder GlobalTE session.
        control_host: Interface on which the decoder service binds.
        control_advertise_host: Routable host advertised to the prefiller. If
            omitted, ``control_host`` must itself be routable.
        control_port: Dedicated control port for this decoder DP rank.
        shared_cache_generation: Current registered shared-slab generation.
        capacity_available: Nonblocking ordinary-cache headroom check.
        fatal_restart: Whole-deployment restart callback.
        global_te_push: Whether Gate H0-qualified native push is available.
        save_only_first_rank: Resolved physical Group 0 ownership policy.
        save_indexer_only_first_rank: Resolved Group 1 ownership policy.
        shared_cpu_cache_strict: Whether every passive view passed preflight.
        chunk_hash_type: Actual output type of the decoder token hash function.
        chunk_hash_bytes: Actual fixed digest width for byte hashes.
        descriptor_verification_capability: Optional injected decoder-runtime
            descriptor verification key for tests. Production generates one.
        server_factory: Optional RPC server factory for tests.

    Returns:
        A stopped decoder runtime.

    Raises:
        ValueError: If the rank, endpoint, or topology is incompatible.
    """

    if not metadata.is_first_rank() or metadata.worker_id != 0:
        raise ValueError("remote-fill decoder service must run on physical TP0")
    if destination_engine_epoch <= 0:
        raise ValueError("remote-fill decoder engine epoch must be positive")
    if not bool(config.enable_remote_lmcache_store):
        raise ValueError("remote-fill decoder runtime requires the feature flag")
    if not metadata.use_mla or not bool(config.local_cpu):
        raise ValueError("remote-fill decoder requires MLA LocalCPU storage")
    if not (
        config.use_layerwise
        and config.dsa_two_groups
        and config.enable_sparse_attention
        and save_only_first_rank
        and save_indexer_only_first_rank
    ):
        raise ValueError("remote-fill decoder requires TP0 ownership of both groups")
    if metadata.world_size > 1 and not (
        bool(config.enable_shared_cpu_cache) and shared_cpu_cache_strict
    ):
        raise ValueError("remote-fill TP>1 requires strict shared LocalCPU storage")
    if metadata.world_size > 1 and shared_cache_generation <= 0:
        raise ValueError("remote-fill TP>1 requires a positive shared generation")
    if destination_dp_rank < 0 or not 0 <= destination_dp_rank < destination_dp_size:
        raise ValueError("remote-fill destination DP rank is out of range")
    if not control_host or not 0 < control_port < 65536:
        raise ValueError("remote-fill control endpoint is invalid")
    advertised_host = _validate_advertised_control_host(
        control_advertise_host or control_host
    )
    engine_id = str(metadata.engine_id or "").strip()
    if not engine_id:
        raise ValueError("remote-fill decoder requires destination engine_id")
    limits = ProtocolLimits(
        max_rpc_message_bytes=int(config.remote_fill_max_rpc_message_bytes),
        max_control_pages_per_window=int(
            config.remote_fill_max_control_pages_per_window
        ),
        max_window_bytes=int(config.remote_fill_max_inflight_bytes),
        max_active_transactions=int(config.remote_fill_max_active_transactions),
        max_inflight_windows_per_transaction=int(
            config.remote_fill_max_inflight_windows_per_request
        ),
        max_reserved_bytes=int(config.remote_fill_max_reserved_bytes),
        max_bytes_per_transaction=int(config.remote_fill_max_bytes_per_request),
    )
    negotiation = build_remote_fill_negotiation_spec(
        config,
        metadata,
        layout=layout,
        destination_engine_id=engine_id,
        destination_dp_rank=destination_dp_rank,
        dp_size=destination_dp_size,
        global_te_push=global_te_push,
        destination_remote_session=destination_remote_session,
        chunk_hash_type=chunk_hash_type,
        chunk_hash_bytes=chunk_hash_bytes,
    )
    lifecycle = AscendRemoteFillPageLifecycle(
        local_backend=local_backend,
        layout=layout,
        capacity_available=capacity_available,
        chunk_hash_type=chunk_hash_type,
        chunk_hash_bytes=chunk_hash_bytes,
    )
    verification_capability = _descriptor_verification_capability(
        descriptor_verification_capability
    )
    state = RemoteFillStateCore(
        destination_engine_epoch=destination_engine_epoch,
        shared_cache_generation=shared_cache_generation,
        descriptor_verification_key=verification_capability,
        negotiation=negotiation,
        page_lifecycle=lifecycle,
        limits=limits,
        reservation_ttl_sec=float(config.remote_fill_reservation_ttl_sec),
        descriptor_ttl_sec=float(config.remote_fill_descriptor_ttl_sec),
        native_hard_timeout_sec=(
            float(config.remote_fill_native_hard_timeout_ms) / 1000.0
        ),
        terminal_record_ttl_sec=float(config.remote_fill_terminal_record_ttl_sec),
    )
    service = RemoteFillService(state, limits)
    bind_endpoint = f"{_format_tcp_host(control_host)}:{control_port}"
    advertised_endpoint = f"{_format_tcp_host(advertised_host)}:{control_port}"
    if server_factory is None:
        transport = ZmqRouterServerTransport(
            bind_endpoint,
            recv_timeout_ms=100,
            protocol="tcp",
            max_frame_bytes=limits.max_rpc_message_bytes + 16,
            max_frame_count=3,
        )
        server: RemoteFillServer = RemoteFillRpcServer(transport, service)
    else:
        server = server_factory(service)
    host = DecoderRemoteFillServiceHost(
        service=service,
        server=server,
        fatal_restart=fatal_restart,
    )
    placement = RemoteFillPlacementInfo(
        enabled=True,
        destination_engine_id=engine_id,
        destination_engine_epoch=destination_engine_epoch,
        control_endpoint=f"tcp://{advertised_endpoint}",
        shared_cache_generation=shared_cache_generation,
        protocol_version=PROTOCOL_VERSION,
        layout_tag=layout.layout_tag,
        cache_namespace_tag=layout.cache_namespace_tag,
        model_artifact_id=layout.model_artifact_id,
        tp_rank=0,
        dp_rank=destination_dp_rank,
        destination_tp_size=int(metadata.world_size),
        destination_dp_size=destination_dp_size,
        global_te_push=global_te_push,
        token_hash_algorithm=negotiation.token_hash_algorithm,
        python_hash_seed=negotiation.python_hash_seed,
        descriptor_verification_capability=verification_capability.hex(),
    )
    return DecoderRemoteFillRuntime(
        lifecycle=lifecycle,
        state=state,
        service=service,
        host=host,
        placement=placement,
    )
