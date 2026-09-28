#!/usr/bin/env python3
"""Exercise the installed LMCache-Ascend allocators, without loading a model."""

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import uuid


def emit(event, **fields):
    print(json.dumps(dict(event=event, pid=os.getpid(), **fields)), flush=True)


def mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def memory_cgroups(cgroup, mountinfo):
    """Resolve memory-controller paths against visible mounts, including bind roots."""
    for line in cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, group = parts
        version = 2 if controllers == "" else 1
        if version == 1 and "memory" not in controllers.split(","):
            continue
        for mount in mountinfo.splitlines():
            left, sep, right = mount.partition(" - ")
            fields, fs = left.split(), right.split()
            if not sep or len(fields) < 6 or len(fs) < 3:
                continue
            if fs[0] != ("cgroup2" if version == 2 else "cgroup"):
                continue
            if version == 1 and "memory" not in fs[2].split(","):
                continue
            root, target = mount_path(fields[3]), mount_path(fields[4])
            if group == root:
                relative = ""
            elif root == "/":
                relative = group.lstrip("/")
            elif group.startswith(root.rstrip("/") + "/"):
                relative = group[len(root):].lstrip("/")
            else:
                continue
            if ".." not in Path(relative).parts:
                yield version, Path(target), Path(target) / relative


def cgroup_snapshot(cgroup, mountinfo):
    result = {}
    for version, mount, current in memory_cgroups(cgroup, mountinfo):
        names = (("memory.max", "memory.current", "memory.events") if version == 2 else
                 ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.failcnt",
                  "memory.oom_control", "memory.stat"))
        while True:
            values = {}
            for name in names:
                try:
                    values[name] = (current / name).read_text().strip()
                except OSError:
                    pass
            if values:
                result[str(current)] = dict(version=version, **values)
            if current == mount:
                break
            current = current.parent
    return result


def snapshot():
    result = {}
    for name in ("/proc/meminfo", "/proc/self/status", "/proc/self/limits",
                 "/proc/self/cgroup", "/proc/self/mountinfo"):
        try:
            result[name] = Path(name).read_text()
        except OSError:
            pass
    try:
        stat = os.statvfs("/dev/shm")
        result["shm_total_bytes"] = stat.f_blocks * stat.f_frsize
        result["shm_available_bytes"] = stat.f_bavail * stat.f_frsize
    except (OSError, AttributeError):
        pass
    result["memory_cgroups"] = cgroup_snapshot(
        result.get("/proc/self/cgroup", ""), result.pop("/proc/self/mountinfo", ""))
    return result


def regions(total, chunk):
    if total <= 0 or chunk < 0:
        raise ValueError("positive total and nonnegative chunk required")
    chunk = chunk or total
    return [min(chunk, total - start) for start in range(0, total, chunk)]


def allocate(ops, args, size, name, owner):
    if args.mode == "private":
        return ops.alloc_pinned_ptr(size, 0)
    if args.mode == "numa":
        return ops.alloc_pinned_numa_ptr(size, args.numa_node)
    if owner:
        return ops.alloc_shm_pinned_ptr(size, name, args.interleave_nodes)
    # Match the engine's default read-only attempt, then writable fallback.
    if args.attach == "auto":
        try:
            return ops.attach_shm_pinned_ptr(size, name, False)
        except Exception as exc:
            emit("readonly_attach_failed", error=str(exc), name=name)
    return ops.attach_shm_pinned_ptr(size, name, args.attach != "readonly")


def release(ops, args, ptr, size, name, owner):
    if args.mode == "private":
        ops.free_pinned_ptr(ptr)
    elif args.mode == "numa":
        ops.free_pinned_numa_ptr(ptr, size)
    elif owner:
        ops.free_shm_pinned_ptr(ptr, size, name)
    else:
        ops.detach_shm_pinned_ptr(ptr, size)


def worker(args, device, sizes, names, owner, conn):
    allocations = []
    ok = True
    try:
        # Set before importing the extension; native shared-slab phase logs
        # distinguish reserve/populate/owner_register/attach_register stalls.
        os.environ["PD_SERVING_PERF"] = "detail"
        import torch
        import torch_npu
        from lmcache_ascend import c_ops

        torch.npu.set_device(device)
        torch.npu.synchronize()
        emit("worker_start", device=device, owner=owner, torch=torch.__version__,
             torch_npu=torch_npu.__version__, extension=c_ops.__file__,
             environment={k: os.environ.get(k) for k in
                          ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
             memory=snapshot())
        for size, name in zip(sizes, names):
            emit("allocate_start", device=device, bytes=size, name=name)
            start = time.perf_counter()
            ptr = allocate(c_ops, args, size, name, owner)
            allocations.append((ptr, size, name))
            # Validate both ends against the production registration registry.
            if not c_ops.get_device_ptr(ptr, size) or not c_ops.get_device_ptr(ptr + size - 1, 1):
                raise RuntimeError("allocation returned without a complete device mapping")
            emit("allocate_complete", device=device, bytes=size,
                 elapsed_s=time.perf_counter() - start, memory=snapshot())
        conn.send("ready")
        conn.recv()  # Keep all registrations live until parent releases us.
    except BaseException as exc:
        ok = False
        emit("worker_failed", device=device, error=repr(exc), memory=snapshot())
    finally:
        for ptr, size, name in reversed(allocations):
            try:
                start = time.perf_counter()
                release(c_ops, args, ptr, size, name, owner)
                emit("release_complete", device=device, bytes=size,
                     elapsed_s=time.perf_counter() - start)
            except Exception as exc:
                ok = False
                emit("release_failed", device=device, error=repr(exc))
        conn.close()
    if not ok:
        raise SystemExit(1)


def trial(args, total):
    sizes = regions(total, args.chunk_bytes)
    before = snapshot()
    emit("preflight", mode=args.mode, total_bytes=total, memory=before)
    available = before.get("shm_available_bytes")
    # Every region remains live. Splitting cannot evade tmpfs capacity.
    page = os.sysconf("SC_PAGE_SIZE") if args.mode == "shared" else 1
    required = sum((size + page - 1) // page * page for size in sizes)
    if args.mode == "shared" and available is not None and required > available:
        emit("trial_result", total_bytes=total, success=False,
             failure_stage="shm_capacity_preflight", required_bytes=required,
             available_bytes=available, registration_attempted=False)
        return False
    prefix = "lmcache_regbench_" + uuid.uuid4().hex
    names = [f"/{prefix}_{i}" for i in range(len(sizes))]
    ctx = mp.get_context("spawn")
    workers = []
    ok = True
    emit("trial_start", mode=args.mode, total_bytes=total, region_bytes=sizes,
         devices=args.devices, physical_bytes=total * (1 if args.mode == "shared" else len(args.devices)))
    try:
        for rank, device in enumerate(args.devices):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(args, device, sizes, names, rank == 0, child))
            proc.start()
            child.close()
            workers.append((proc, parent))
            if not parent.poll(args.timeout):
                raise TimeoutError(f"device {device}: no ready message within {args.timeout}s")
            if parent.recv() != "ready":
                raise RuntimeError(f"device {device}: unexpected worker response")
        emit("all_registered", total_bytes=total, devices=args.devices)
        time.sleep(args.hold_seconds)
    except (Exception, KeyboardInterrupt) as exc:
        ok = False
        emit("trial_failed", error=repr(exc))
    finally:
        # Detach passive mappings before freeing/unlinking the owner's slab.
        for proc, conn in reversed(workers):
            try:
                conn.send("release")
            except (BrokenPipeError, EOFError, OSError):
                pass
            proc.join(args.timeout)
            if proc.is_alive():
                ok = False
                emit("worker_timeout", worker_pid=proc.pid)
                proc.kill()
                proc.join(10)
            if proc.exitcode != 0:
                ok = False
            conn.close()
        # Recover only this trial's uniquely named slabs after crashes/timeouts.
        if args.mode == "shared":
            for name in names:
                Path("/dev/shm", name.lstrip("/")).unlink(missing_ok=True)
    emit("trial_result", total_bytes=total, success=ok)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-gib", type=float, nargs="+", default=[1])
    parser.add_argument("--chunk-gib", type=float, default=0, help="0: one allocation; otherwise retain multiple chunks")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="logical NPU IDs; shared mode maps ONE slab into every worker")
    parser.add_argument("--mode", choices=["shared", "private", "numa"], default="shared")
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--interleave-nodes", type=int, nargs="*", default=[])
    parser.add_argument("--attach", choices=["auto", "readonly", "writable"], default="auto")
    parser.add_argument("--timeout", type=float, default=600, help="per-worker startup AND cleanup deadline in seconds")
    parser.add_argument("--hold-seconds", type=float, default=2)
    args = parser.parse_args()
    if (any(not math.isfinite(s) or s <= 0 for s in args.sizes_gib)
            or not math.isfinite(args.chunk_gib) or args.chunk_gib < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.hold_seconds) or args.hold_seconds < 0
            or any(d < 0 for d in args.devices) or not 0 <= args.numa_node < 64
            or any(n < 0 for n in args.interleave_nodes)):
        parser.error("invalid size, time, device or NUMA node")
    if sys.platform != "linux":
        parser.error("requires Linux and an installed Ascend runtime")
    args.chunk_bytes = int(args.chunk_gib * 2**30)
    if any(int(s * 2**30) < 1 for s in args.sizes_gib) or (args.chunk_gib and args.chunk_bytes < 1):
        parser.error("sizes must be at least one byte")
    for gib in args.sizes_gib:
        if not trial(args, int(gib * 2**30)):
            return 1  # Stop at first failure; avoid cascading pressure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
