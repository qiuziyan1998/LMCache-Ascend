#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone Linux/NPU feasibility probe; no LMCache/vLLM imports or patches.

One W subprocess owns each NPU allocation. One M subprocess imports them and
uses the installed Mooncake Store GET. The supervisor never initializes NPU.
See README.txt before running. This is a transport test, not a throughput test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import gc
import json
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import uuid


GUARD = 4096
VIEW_BYTES = 512 * 1024
SENTINEL = 165
CASES = {
    "contiguous": (((4096, 65536),), ((196608, 32768),)),
    "fragmented": (
        ((4096, 16384), (65536, 32768), (147456, 16384)),
        ((229376, 4096), (270336, 2048), (335872, 1024)),
    ),
    "partial": (
        ((8192, 384), (131072, 1280), (200704, 256)),
        ((290816, 64), (307200, 512), (352256, 192)),
    ),
}


def log(probe_event: str, **values) -> None:
    print(json.dumps({"probe": probe_event, "pid": os.getpid(), **values}), flush=True)


def check_status(value, operation: str) -> None:
    if value is not None and value != 0:
        raise RuntimeError(f"{operation} returned {value!r}")


def payload(size: int, device: int, iteration: int, case: str, page: int) -> bytes:
    seed = device * 47 + iteration * 31 + tuple(CASES).index(case) * 71 + page * 113
    period = bytes((i + seed) % 251 for i in range(251))
    return (period * ((size + 250) // 251))[:size]


def expected(device: int, iteration: int, case: str) -> bytearray:
    result = bytearray([SENTINEL]) * (VIEW_BYTES + 2 * GUARD)
    for page, runs in enumerate(CASES[case]):
        data = payload(sum(size for _, size in runs), device, iteration, case, page)
        pos = 0
        for offset, size in runs:
            result[GUARD + offset:GUARD + offset + size] = data[pos:pos + size]
            pos += size
    return result


def validate_runs(runs, capacity: int) -> None:
    intervals = sorted((offset, offset + size) for row in runs for offset, size in row)
    for index, (start, end) in enumerate(intervals):
        if start < 0 or end <= start or end > capacity:
            raise ValueError("Destination outside exported view")
        if index and intervals[index - 1][1] > start:
            raise ValueError("Overlapping destination runs")


def plain_handle(value) -> bool:
    """Refuse accidental tensor/Future serialization across the probe boundary."""
    return type(value) in (int, bool, str, bytes, type(None)) or (
        type(value) in (tuple, list) and all(plain_handle(item) for item in value)
    )


def child_runtime():
    # spawn starts a fresh interpreter; device imports happen only in children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    gc.disable()
    import torch
    import torch_npu  # noqa: F401

    return torch


def hold_resources(conn, reason: str) -> None:
    """Keep the child's stack/owners alive when safe retirement is unproven."""
    log("QUARANTINED", reason=reason)
    while True:
        try:
            # There is intentionally no force-free command.
            if conn.poll(1):
                conn.recv()
        except (EOFError, OSError):
            time.sleep(1)


def owner_main(conn, device: int) -> None:
    root = view = storage = None
    try:
        torch = child_runtime()
        torch.npu.set_device(device)
        root = torch.full(
            (VIEW_BYTES + 2 * GUARD,), SENTINEL,
            dtype=torch.uint8, device=f"npu:{device}",
        )
        view = root.narrow(0, GUARD, VIEW_BYTES)
        torch.npu.synchronize()
        storage = view.untyped_storage()
        handle = storage._share_npu_()
        if not plain_handle(handle):
            raise TypeError("Unexpected NPU IPC handle contents; refusing tensor pickling")
        descriptor = {
            "device": device, "handle": handle,
            "shape": tuple(view.shape), "stride": tuple(view.stride()),
            "storage_offset": int(view.storage_offset()),
            "storage_bytes": int(storage.nbytes()), "view_bytes": VIEW_BYTES,
            "exporter_pointer": int(view.data_ptr()),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__, "device_name": torch.npu.get_device_name(device),
        }
        conn.send({"event": "owner_ready", "descriptor": descriptor})
        # The root and exported view remain explicitly owned until M retires.
        while True:
            command = conn.recv()
            if command["op"] == "reset":
                root.fill_(SENTINEL)
                torch.npu.synchronize()
                conn.send({"event": "reset_done", "device": device})
            elif command["op"] == "verify":
                # No M-local stream/event is used here. GET completion must
                # already make the writes visible to the exporting process.
                actual = root.cpu()
                want = torch.frombuffer(
                    expected(device, command["iteration"], command["case"]),
                    dtype=torch.uint8,
                )
                mismatch = (actual != want).nonzero().flatten()
                bad = mismatch[:8].tolist()
                conn.send({
                    "event": "verified", "device": device,
                    "ok": not bad, "mismatches": int(mismatch.numel()),
                    "first_offsets": bad,
                    "actual": [int(actual[i]) for i in bad],
                    "expected": [int(want[i]) for i in bad],
                })
            elif command["op"] == "release":
                torch.npu.synchronize()
                del view, storage, root
                conn.send({"event": "owner_released", "device": device})
                return
            else:
                raise ValueError(f"Unknown owner command: {command}")
    except BaseException:
        try:
            conn.send({"event": "owner_error", "error": traceback.format_exc()})
        except (EOFError, OSError):
            hold_resources(conn, "Owner lost its supervisor")
        # Only the supervisor, after importer retirement, may release this root.
        while True:
            try:
                if conn.recv().get("op") == "release":
                    if root is not None:
                        torch.npu.synchronize()
                    del view, storage, root
                    conn.send({"event": "owner_released", "device": device})
                    return
            except BaseException:
                hold_resources(conn, "Failed to retire an exporter safely")


class Reader:
    """Explicit transport instance; mirrors the shared-engine Store setup API."""

    def __init__(self, torch, options, device: int):
        self.torch, self.device = torch, device
        self.engine = self.store = None
        self.host_owners = []
        self.host_regs = []
        self.device_regs = []
        self.keys = []
        self.unknown = False
        self.setup_started = False
        from mooncake.engine import TransferEngine
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        self.engine, self.store = TransferEngine(), MooncakeDistributedStore()
        self.replica = ReplicateConfig()
        self.replica.replica_num = 1
        torch.npu.set_device(device)
        check_status(self.engine.initialize(
            options["host"], "P2PHANDSHAKE", "ascend", options["device_name"]
        ), "TransferEngine.initialize")
        session = f'{options["host"]}:{self.engine.get_rpc_port()}'
        self.setup_started = True
        check_status(self.store.setup(
            session, options["metadata_server"], 0, 0, "ascend",
            options["device_name"], options["master"], self.engine.get_engine(),
        ), "Store.setup")
        log("reader_ready", device=device, session=session,
            global_segment_bytes=0, local_buffer_bytes=0)

    def register_device(self, tensor) -> None:
        self.torch.npu.set_device(tensor.device)
        storage = tensor.untyped_storage()
        ptr, size = int(storage.data_ptr()), int(storage.nbytes())
        for base, extent, registered_device in self.device_regs:
            if ptr == base and size == extent and tensor.device.index == registered_device:
                return
            if ptr < base + extent and base < ptr + size:
                raise RuntimeError("Imported registrations overlap ambiguously")
        check_status(self.engine.register_memory(ptr, size), "TE.register_memory(imported HBM)")
        self.device_regs.append((ptr, size, tensor.device.index))

    def put_fixture(self, namespace: str, device: int, iteration: int, case: str):
        self.torch.npu.set_device(self.device)
        keys, ptrs, sizes = [], [], []
        for page, runs in enumerate(CASES[case]):
            size = sum(count for _, count in runs)
            capacity = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
            data = mmap.mmap(-1, capacity)
            data.write(payload(size, device, iteration, case, page))
            owner = self.torch.frombuffer(data, dtype=self.torch.uint8)
            ptr = int(owner.data_ptr())
            # Keep data and its Tensor alive even if a native call throws.
            self.host_owners.append((data, owner))
            check_status(self.store.register_buffer(ptr, capacity), "Store.register_buffer(CPU fixture)")
            self.host_regs.append(ptr)
            keys.append(f"{namespace}/d{device}/r{iteration}/{case}/p{page}")
            ptrs.append([ptr])
            sizes.append([size])
        self.keys.extend(keys)
        self.unknown = True
        statuses = self.store.batch_put_from_multi_buffers(keys, ptrs, sizes, self.replica)
        self.unknown = False  # A returned synchronous call is terminal.
        if statuses is None or len(statuses) != len(keys) or any(s != 0 for s in statuses):
            raise RuntimeError(f"Fixture PUT failed: {statuses!r}; check master/storage capacity")
        return keys

    def get(self, tensor, keys, case: str, barrier):
        self.torch.npu.set_device(tensor.device)
        validate_runs(CASES[case], int(tensor.numel()))
        base = int(tensor.data_ptr())  # M's imported view, never exporter_pointer.
        ptrs = [[base + offset for offset, _ in row] for row in CASES[case]]
        sizes = [[size for _, size in row] for row in CASES[case]]
        barrier.wait(timeout=30)
        started = time.perf_counter()
        # Shared mode uses one Reader concurrently. Any exception latches an
        # unknown outcome; a successful sibling must not clear that latch.
        try:
            statuses = self.store.batch_get_into_multi_buffers(keys, ptrs, sizes)
        except BaseException:
            self.unknown = True
            raise
        expected_sizes = [sum(row) for row in sizes]
        if statuses is None or list(statuses) != expected_sizes:
            raise RuntimeError(f"GET byte counts {statuses!r} != {expected_sizes!r}")
        return {
            "device": tensor.device.index, "bytes": sum(expected_sizes),
            "descriptors": sum(map(len, sizes)),
            "native_call_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def close(self):
        if self.unknown:
            raise RuntimeError("Native completion is unknown; refusing to unregister")
        warnings = []
        remove = getattr(self.store, "remove", None)
        for key in self.keys:
            try:
                if remove is None:
                    raise RuntimeError("Store.remove is unavailable")
                check_status(remove(key), "remove probe-owned key")
            except Exception as exc:
                warnings.append(f"{key}: {exc}")
        for ptr in reversed(self.host_regs):
            check_status(self.store.unregister_buffer(ptr), "unregister CPU fixture")
        self.host_regs.clear()
        for ptr, _, device in reversed(self.device_regs):
            self.torch.npu.set_device(device)
            check_status(self.engine.unregister_memory(ptr), "unregister imported HBM")
        self.device_regs.clear()
        if self.setup_started:
            check_status(self.store.close(), "Store.close")
        while self.host_owners:
            mapping, owner = self.host_owners.pop()
            del owner
            mapping.close()
        return warnings


def storage_main(conn, options, descriptors) -> None:
    readers, views, imported_storages = [], {}, []
    safe = False
    failure = None
    try:
        torch = child_runtime()
        import mooncake.store as module

        log("installed_module", module=module.__file__, torch=torch.__version__)
        for descriptor in descriptors:
            device = descriptor["device"]
            if descriptor["visible_devices"] != os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
                raise ValueError("W/M device visibility differs")
            torch.npu.set_device(device)
            storage = torch.UntypedStorage._new_shared_npu(device, *descriptor["handle"][1:])
            imported_storages.append(storage)
            if int(storage.nbytes()) != descriptor["storage_bytes"]:
                raise ValueError("Imported storage extent differs from export")
            view = torch.empty(0, dtype=torch.uint8, device=f"npu:{device}")
            view.set_(storage, descriptor["storage_offset"], descriptor["shape"], descriptor["stride"])
            views[device] = view
            log("imported", device=device, imported_pointer=int(view.data_ptr()),
                exporter_pointer=descriptor["exporter_pointer"], view_offset=descriptor["storage_offset"])
            if options["engine_mode"] == "per-device" or not readers:
                # Retain an object even when its constructor partially fails.
                reader = Reader.__new__(Reader)
                readers.append(reader)
                reader.__init__(torch, options, device)
            reader = readers[-1] if options["engine_mode"] == "per-device" else readers[0]
            reader.register_device(view)
        device_readers = {
            descriptor["device"]: readers[index] if options["engine_mode"] == "per-device" else readers[0]
            for index, descriptor in enumerate(descriptors)
        }
        conn.send({"event": "storage_ready", "module": module.__file__})
        with ThreadPoolExecutor(max_workers=len(descriptors)) as pool:
            while True:
                command = conn.recv()
                if command["op"] == "close":
                    break
                if command["op"] != "run":
                    raise ValueError("Unknown storage command")
                case, iteration = command["case"], command["iteration"]
                keys = {
                    device: reader.put_fixture(options["namespace"], device, iteration, case)
                    for device, reader in device_readers.items()
                }
                barrier = threading.Barrier(len(descriptors))
                futures = [
                    pool.submit(reader.get, views[device], keys[device], case, barrier)
                    for device, reader in device_readers.items()
                ]
                wait(futures)  # Only M waits; W allocations remain held.
                results = [future.result() for future in futures]
                # Exercise retention across a deliberately delayed completion reply.
                time.sleep(options["ack_delay_ms"] / 1000)
                conn.send({"event": "read_done", "results": results})
    except BaseException:
        failure = traceback.format_exc()
    try:
        if any(getattr(reader, "unknown", False) for reader in readers):
            raise RuntimeError("At least one original native call has unknown completion")
        warnings = []
        for reader in reversed(readers):
            warnings.extend(reader.close())
        # Retire all imported views before acknowledging that W can exit.
        views.clear()
        imported_storages.clear()
        # Loop locals retain the final view/storage until explicitly removed.
        if "view" in locals():
            del view
        if "storage" in locals():
            del storage
        safe = True
        conn.send({"event": "storage_closed", "safe": True, "error": failure, "warnings": warnings})
    except BaseException:
        detail = traceback.format_exc()
        safe = False
        try:
            conn.send({"event": "storage_failed", "safe": False, "error": failure, "cleanup_error": detail})
        except (EOFError, OSError):
            log("lost_supervisor", error=detail)
    if not safe:
        hold_resources(conn, "M could not establish safe importer retirement")


class UnexpectedReply(RuntimeError):
    def __init__(self, conn, reply):
        super().__init__(json.dumps(reply))
        self.conn, self.reply = conn, reply


def receive(conn, expected_event: str, timeout: float):
    if not conn.poll(timeout):
        raise TimeoutError(f"Timed out waiting for {expected_event}")
    result = conn.recv()
    if result.get("event") != expected_event:
        raise UnexpectedReply(conn, result)
    return result


def run(options, owner_target=owner_main, storage_target=storage_main):
    context = mp.get_context("spawn")
    owners = []
    storage_process = storage_conn = None
    storage_safe = True  # No importer exists yet.
    storage_failed_ack = False
    checks_passed = False
    report = {"options": options, "phases": [], "status": "FAIL"}
    try:
        for device in options["devices"]:
            parent, child = context.Pipe()
            process = context.Process(target=owner_target, args=(child, device), name=f"probe-W-{device}")
            process.start()
            child.close()
            owners.append((process, parent))
        descriptors = [receive(conn, "owner_ready", options["timeout"])["descriptor"] for _, conn in owners]
        report["exporters"] = [{key: value for key, value in item.items() if key != "handle"}
                               for item in descriptors]
        parent, child = context.Pipe()
        storage_conn = parent
        storage_process = context.Process(
            target=storage_target, args=(child, options, descriptors), name="probe-M"
        )
        storage_process.start()
        storage_safe = False
        child.close()
        report["storage"] = receive(parent, "storage_ready", options["timeout"])
        for iteration in range(options["iterations"]):
            for case in CASES:
                for _, conn in owners:
                    conn.send({"op": "reset"})
                for _, conn in owners:
                    receive(conn, "reset_done", options["timeout"])
                log("phase_started", iteration=iteration, case=case)
                parent.send({"op": "run", "iteration": iteration, "case": case})
                result = receive(parent, "read_done", options["timeout"])
                for _, conn in owners:
                    conn.send({"op": "verify", "iteration": iteration, "case": case})
                checks = [receive(conn, "verified", options["timeout"]) for _, conn in owners]
                phase = {"iteration": iteration, "case": case, **result, "checks": checks}
                report["phases"].append(phase)
                log("phase_result", **phase)
                if not all(check["ok"] for check in checks):
                    raise RuntimeError("Data or guard-region verification failed")
        parent.send({"op": "close"})
        closed = receive(parent, "storage_closed", options["timeout"])
        storage_safe = bool(closed["safe"])
        report["cleanup"] = closed
        if closed.get("error"):
            raise RuntimeError(closed["error"])
        checks_passed = True
    except BaseException as error:
        if isinstance(error, UnexpectedReply) and error.conn is storage_conn:
            reply = error.reply
            report["cleanup"] = reply
            storage_safe = reply.get("event") == "storage_closed" and reply.get("safe") is True
            storage_failed_ack = reply.get("event") == "storage_failed"
        report["error"] = traceback.format_exc()
        log("failure", error=report["error"])
    finally:
        if storage_process is not None and not storage_safe and not storage_failed_ack:
            # Drain queued completion before the cleanup acknowledgement. Never
            # infer DMA completion from timeout, process death or cancellation.
            try:
                storage_conn.send({"op": "close"})
                deadline = time.monotonic() + options["timeout"]
                while storage_conn.poll(max(0, deadline - time.monotonic())):
                    closed = storage_conn.recv()
                    if closed.get("event") in ("storage_closed", "storage_failed"):
                        report["cleanup"] = closed
                        storage_safe = closed.get("event") == "storage_closed" and closed.get("safe") is True
                        break
                    log("draining", event_received=closed.get("event"))
            except (EOFError, OSError):
                pass
        report["safe_to_release_owners"] = storage_safe
        Path(options["output"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not storage_safe:
            report["status"] = "UNKNOWN_COMPLETION_OR_RETIREMENT"
            Path(options["output"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
            log("QUARANTINED", report=options["output"],
                owner_pids=[p.pid for p, _ in owners], storage_pid=storage_process.pid)
            hold_resources(storage_conn, "Owners retained; inspect report and native logs")
        try:
            if storage_process is not None:
                storage_process.join(options["timeout"])
                if storage_process.is_alive() or storage_process.exitcode != 0:
                    raise RuntimeError("M did not exit cleanly after retirement acknowledgement")
            for process, conn in owners:
                if process.is_alive():
                    conn.send({"op": "release"})
                    deadline = time.monotonic() + options["timeout"]
                    while True:
                        if not conn.poll(max(0, deadline - time.monotonic())):
                            raise TimeoutError("Owner did not acknowledge release")
                        reply = conn.recv()
                        if reply.get("event") == "owner_released":
                            break
                        if reply.get("event") not in ("owner_ready", "owner_error"):
                            raise UnexpectedReply(conn, reply)
                process.join(options["timeout"])
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError("Owner did not exit cleanly after release")
                conn.close()
            if storage_conn is not None:
                storage_conn.close()
            report["status"] = "PASS" if checks_passed else "FAIL"
        except BaseException:
            report.update(status="RETIREMENT_FAILED", retirement_error=traceback.format_exc())
            Path(options["output"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
            if any(p.is_alive() for p, _ in owners) or (
                storage_process is not None and storage_process.is_alive()
            ):
                hold_resources(storage_conn, "Probe child retirement failed; inspect report")
        Path(options["output"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("summary", status=report["status"], report=options["output"])
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="This D host's routable Mooncake IP")
    parser.add_argument("--master", required=True, help="Existing Mooncake master HOST:PORT")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--device-name", default="", help="Same transport device_name as deployment")
    parser.add_argument("--devices", default="0", help="Visible logical NPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--engine-mode", choices=("per-device", "shared"), default="per-device")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--ack-delay-ms", type=int, default=100,
                        help="Delay M completion reply to exercise exporter retention")
    parser.add_argument("--timeout", type=float, default=180,
                        help="Control timeout; expiry retains potentially live allocations")
    parser.add_argument("--output", default="mooncake-hbm-ipc-report.json")
    options = vars(parser.parse_args())
    try:
        devices = [int(item) for item in options["devices"].split(",")]
    except ValueError:
        parser.error("--devices must contain comma-separated nonnegative integers")
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("--devices must be nonempty, unique and nonnegative")
    if options["iterations"] < 1 or options["timeout"] <= 0 or options["ack_delay_ms"] < 0:
        parser.error("Invalid iterations, timeout or acknowledgement delay")
    options.update(devices=devices, namespace="lmcache-ipc-probe-" + uuid.uuid4().hex)
    Path(options["output"]).parent.mkdir(parents=True, exist_ok=True)
    log("start", **options)
    return run(options)


if __name__ == "__main__":
    sys.exit(main())
