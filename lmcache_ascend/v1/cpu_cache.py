# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU storage construction for the decoder storage owner."""

# Standard
from collections import OrderedDict
from typing import Any, AbstractSet
import asyncio


class CPUBackendFactory:
    """Reuse one pre-registered allocator for LocalCPU and Mooncake backends.

    ``memory_allocator`` must be allocated during the pre-profile bootstrap.
    The factory never constructs an engine, connector, device stream or slab.
    Recreating the slab owner requires a service restart; remote recreation
    reuses the live LocalCPU backend. Unsupported backends raise ``ValueError``.
    """

    def __init__(
        self,
        memory_allocator: Any,
        *,
        local_backend: Any = None,
        defer_layout: bool = False,
    ) -> None:
        if memory_allocator is None:
            raise ValueError("CPU storage requires a precreated allocator")
        self.memory_allocator = memory_allocator
        if (
            local_backend is not None
            and local_backend.memory_allocator is not memory_allocator
        ):
            raise ValueError("Bootstrap backend does not own the supplied allocator")
        self._bootstrap_backend = local_backend
        self.defer_layout = defer_layout
        self._local_backend: Any = None
        self._remote_backend: Any = None
        self._layout_metadata: Any = None
        self._remote_registration_protected = False

    def __call__(
        self,
        config: Any,
        metadata: Any,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cpu",
        lmcache_worker: Any = None,
        skip_backends: AbstractSet[str] | None = None,
        existing_backends: OrderedDict[str, Any] | None = None,
    ) -> OrderedDict[str, Any]:
        """Return new CPU backends, preserving existing allocator ownership.

        Arguments match ``CreateStorageBackends``. ``dst_device`` is ignored:
        service destinations are always CPU. Existing backends are not returned
        again. Invalid configurations or a replacement allocator owner raise
        ``ValueError`` before constructing any backend.
        """
        # First Party
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        extra = config.extra_config or {}
        unsupported = [
            name
            for name in (
                "enable_pd",
                "enable_p2p",
                "gds_path",
                "maru_path",
                "storage_plugins",
                "remote_storage_plugins",
            )
            if getattr(config, name, None)
        ]
        if getattr(config, "local_disk", None) and config.max_local_disk_size > 0:
            unsupported.append("local_disk")
        if extra.get("enable_nixl_storage"):
            unsupported.append("enable_nixl_storage")
        if unsupported:
            raise ValueError("CPU storage does not support: " + ", ".join(unsupported))
        if (
            not config.local_cpu
            or config.max_local_cpu_size <= 0
            or metadata.role == "scheduler"
        ):
            raise ValueError("CPU storage requires an enabled LocalCPU owner")
        if config.remote_url and not config.remote_url.startswith("mooncakestore://"):
            raise ValueError("CPU storage supports only Mooncake remote storage")

        skip = skip_backends or set()
        existing = existing_backends or {}
        if set(existing) - {"LocalCPUBackend", "RemoteBackend"}:
            raise ValueError("CPU storage has unsupported existing backends")
        local = existing.get("LocalCPUBackend")
        if local is None:
            if "LocalCPUBackend" in skip or self._local_backend is not None:
                raise ValueError("CPU allocator owner cannot be replaced")
        elif (
            not isinstance(local, LocalCPUBackend)
            or local.memory_allocator is not self.memory_allocator
            or (self._local_backend is not None and local is not self._local_backend)
        ):
            raise ValueError("CPU storage must reuse its registered allocator owner")

        created: OrderedDict[str, Any] = OrderedDict()
        if local is None:
            local = self._bootstrap_backend or LocalCPUBackend(
                config,
                metadata,
                dst_device="cpu",
                lmcache_worker=lmcache_worker,
                memory_allocator=self.memory_allocator,
            )
            self._bootstrap_backend = None
            created["LocalCPUBackend"] = local
        self._local_backend = local
        if "RemoteBackend" in existing:
            self._remote_backend = getattr(
                existing["RemoteBackend"], "real_backend", existing["RemoteBackend"]
            )
        if (
            config.remote_url
            and "RemoteBackend" not in skip
            and "RemoteBackend" not in existing
        ):
            options = {"defer_layout": True} if self.defer_layout else {}
            remote = RemoteBackend(
                config,
                self._layout_metadata or metadata,
                loop,
                local,
                dst_device="cpu",
                **options,
            )
            self._remote_backend = remote
            if extra.get("audit_backend_enabled", False):
                from lmcache.v1.storage_backend.audit_backend import AuditBackend

                remote = AuditBackend(remote)
            created["RemoteBackend"] = remote
        return created

    def bind_layout(self, metadata: Any) -> None:
        """Bind the startup transport once and reuse full metadata on recreation."""
        if self._remote_backend is not None and self.defer_layout:
            self._remote_backend.bind_layout(metadata)
        self._layout_metadata = metadata
        self.defer_layout = False

    def protect_remote_registration(self) -> None:
        """Keep the receiver's advertised GlobalTE CPU registration live."""
        self._remote_registration_protected = True

    def _validate_remote_close(self, backend_name: str) -> None:
        if backend_name == "RemoteBackend" and self._remote_registration_protected:
            raise ValueError(
                "RemoteFill CPU registration requires a drained service restart"
            )

    def validate_recreate(self, backend_name: str) -> None:
        """Reject owner teardown before it can invalidate registered readers."""
        if backend_name == "LocalCPUBackend":
            raise ValueError("Recreating the CPU allocator owner requires a restart")
        self._validate_remote_close(backend_name)

    def validate_close(self, backend_name: str) -> None:
        """Keep allocator mappings live during individual backend administration."""
        if backend_name == "LocalCPUBackend":
            raise ValueError(
                "Closing the CPU allocator owner requires a drained service stop"
            )
        self._validate_remote_close(backend_name)


def create_storage_runtime(runtime_config: dict[str, Any]) -> Any:
    """Create the sole decoder CPU owner in a fresh interpreter before profiling.

    Only resolved value configuration and static model metadata cross startup.
    Model registration later binds the same slab and Mooncake session to their
    authoritative physical groups; no inference distributed group is initialized.
    """
    # Standard
    from dataclasses import fields
    import os

    # Third Party
    import torch
    import torch_npu  # noqa: F401

    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.event_manager import EventManager
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.observability import PrometheusLogger
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache_ascend.v1.cpu_cache_runtime import AscendCPUCacheRuntime

    config = LMCacheEngineConfig.from_dict(runtime_config["config"])
    if not (
        config.decoder_storage_mode == "process"
        and config.pd_role == "receiver"
        and config.local_cpu
        and config.enable_shared_cpu_cache
        and config.use_layerwise
        and config.dsa_two_groups
    ):
        raise ValueError("Process CPU storage requires a shared two-group decoder")
    if (
        config.enable_controller
        or config.enable_blending
        or config.enable_p2p
        or config.enable_pd
    ):
        raise ValueError(
            "Process decoder storage does not support "
            "controller/blending/P2P/PD backends"
        )
    values = dict(runtime_config["metadata"])
    allowed = {item.name for item in fields(LMCacheMetadata)}
    if set(values) - allowed:
        raise ValueError("Storage bootstrap metadata contains unknown fields")
    if values.get("indexer_hbm_block_map") is not None:
        raise ValueError("Worker HBM addressing cannot cross the storage boundary")
    values["kv_dtype"] = getattr(torch, str(values["kv_dtype"]).removeprefix("torch."))
    values["kv_shape"] = tuple(values["kv_shape"])
    for key in ("runtime_kv_group_layer_counts", "runtime_kv_group_layer_names"):
        if values.get(key) is not None:
            values[key] = tuple(
                tuple(row) if isinstance(row, list) else row for row in values[key]
            )
    values.update(worker_id=0, local_worker_id=0, role="worker")
    metadata = LMCacheMetadata(**values)
    owner_device = int(runtime_config["owner_device"])
    torch.npu.set_device(owner_device)
    config.extra_config = dict(config.extra_config or {})
    slab_name = str(runtime_config["slab_name"])
    if not slab_name.startswith("/"):
        raise ValueError("Storage slab requires an absolute POSIX shared-memory name")
    config.extra_config["shm_name"] = slab_name
    config.extra_config["shared_cpu_cache_resolved_name"] = slab_name
    size_gb = config.get_extra_config_value("shared_cpu_cache_size_gb", None)
    if size_gb is not None:
        config.max_local_cpu_size = float(size_gb)
    if os.name == "posix" and os.path.isdir("/dev/shm"):
        stat = os.statvfs("/dev/shm")
        if int(config.max_local_cpu_size * 1024**3) > stat.f_bavail * stat.f_frsize:
            raise MemoryError("Configured decoder CPU slab does not fit /dev/shm")
    PrometheusLogger.GetOrCreate(metadata, config=config)
    local = LocalCPUBackend(config, metadata, dst_device="cpu")
    manager = None
    try:
        allocator = local.memory_allocator
        root_allocator = getattr(allocator, "_allocator", allocator)
        if getattr(root_allocator, "shm_name", None) != slab_name:
            raise ValueError(
                "Decoder CPU allocator did not create its named shared slab"
            )
        slab_size = int(root_allocator.buffer.numel())
        factory = CPUBackendFactory(allocator, local_backend=local, defer_layout=True)
        manager = StorageManager(
            config,
            metadata,
            EventManager(),
            backend_factory=factory,
            copy_stream_factory=None,
        )
        sessions = {
            name: bytes.fromhex(nonce) if isinstance(nonce, str) else nonce
            for name, nonce in runtime_config["participants"].items()
        }
        return AscendCPUCacheRuntime(
            config,
            metadata,
            None,
            storage_manager=manager,
            layout=None,
            cache_instance_id=runtime_config["cache_instance_id"],
            service_epoch=runtime_config["service_epoch"],
            slab_name=slab_name,
            slab_size=slab_size,
            slab_generation=int(runtime_config.get("slab_generation", 1)),
            sessions=sessions,
            bootstrap_config=runtime_config,
        )
    except BaseException:
        if manager is not None:
            manager.close()
        else:
            local.close()
        raise
