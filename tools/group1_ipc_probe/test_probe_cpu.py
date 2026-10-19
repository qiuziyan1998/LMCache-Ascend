# SPDX-License-Identifier: Apache-2.0
"""CPU-only helper/protocol tests. These do not qualify NPU IPC or Mooncake."""

import json
import contextlib
import ctypes
import io
import mmap
import multiprocessing as mp
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mooncake_hbm_ipc_probe as probe


def fake_owner(conn, device):
    if device == 99:
        conn.send({"event": "owner_error", "error": "injected owner setup failure"})
    else:
        conn.send({"event": "owner_ready", "descriptor": {"device": device}})
    while True:
        command = conn.recv()
        if command["op"] == "release":
            conn.send({"event": "owner_released", "device": device})
            return
        if command["op"] == "reset":
            conn.send({"event": "reset_done", "device": device})
        if command["op"] == "verify":
            conn.send({"event": "verified", "device": device, "ok": device != 9})


def fake_storage(conn, options, descriptors):
    mode = options.get("inject")
    if mode == "startup":
        conn.send({"event": "storage_closed", "safe": True, "error": "injected setup failure"})
        return
    conn.send({"event": "storage_ready"})
    while True:
        command = conn.recv()
        if command["op"] == "close":
            conn.send({"event": "storage_closed", "safe": True, "error": None})
            return
        if mode == "read":
            conn.send({"event": "storage_closed", "safe": True, "error": "injected known read failure"})
            return
        if mode == "late":
            time.sleep(options["timeout"] * 1.2)
        conn.send({"event": "read_done", "results": [{"device": d["device"]} for d in descriptors]})


class Helpers(unittest.TestCase):
    def test_pointer_diagnostics_query_base_and_offset_view(self):
        calls = []

        def query(ptr):
            calls.append(ptr)
            return ({"location": {"type": 7, "id": 0}}, 0)

        tensor = SimpleNamespace(
            device=SimpleNamespace(index=0), data_ptr=lambda: 8192,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 4096, nbytes=lambda: 16384),
        )
        with patch.dict("sys.modules", {"acl": SimpleNamespace(rt=SimpleNamespace(pointer_get_attributes=query))}):
            with contextlib.redirect_stdout(io.StringIO()):
                record = probe.pointer_diagnostic(tensor, "importer")
        self.assertEqual(calls, [4096, 8192])
        self.assertEqual(record["storage_attributes"], repr(({"location": {"type": 7, "id": 0}}, 0)))
        self.assertEqual(record["view_pointer"], 8192)

    def test_pointer_query_error_is_diagnostic_not_a_registration_override(self):
        def query(ptr):
            raise RuntimeError("query unsupported")

        tensor = SimpleNamespace(
            device=SimpleNamespace(index=0), data_ptr=lambda: 8192,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 4096, nbytes=lambda: 16384),
        )
        with patch.dict("sys.modules", {"acl": SimpleNamespace(rt=SimpleNamespace(pointer_get_attributes=query))}):
            with contextlib.redirect_stdout(io.StringIO()):
                record = probe.pointer_diagnostic(tensor, "importer")
        self.assertEqual(record["storage_attributes"], "RuntimeError: query unsupported")
        self.assertEqual(record["view_attributes"], "RuntimeError: query unsupported")

    def test_layout_and_guards(self):
        for case, rows in probe.CASES.items():
            probe.validate_runs(rows, probe.VIEW_BYTES)
            want = probe.expected(2, 7, case)
            self.assertEqual(want[:probe.GUARD], bytes([probe.SENTINEL]) * probe.GUARD)
            self.assertEqual(want[-probe.GUARD:], bytes([probe.SENTINEL]) * probe.GUARD)
            occupied = set()
            for page, runs in enumerate(rows):
                observed = b"".join(want[probe.GUARD + offset:probe.GUARD + offset + size]
                                    for offset, size in runs)
                self.assertEqual(observed, probe.payload(len(observed), 2, 7, case, page))
                for offset, size in runs:
                    occupied.update(range(probe.GUARD + offset, probe.GUARD + offset + size))
            self.assertTrue(all(value == probe.SENTINEL for i, value in enumerate(want) if i not in occupied))

    def test_invalid_bounds_and_overlap(self):
        for rows in ((((-1, 4),),), (((0, 0),),), (((7, 2),),), (((0, 4), (3, 1)),)):
            with self.assertRaises(ValueError):
                probe.validate_runs(rows, 8)

    def test_payload_distinguishes_request_rank_and_page(self):
        values = {
            probe.payload(64, device, iteration, "partial", page)
            for device in (0, 1) for iteration in (0, 1) for page in (0, 1)
        }
        self.assertEqual(len(values), 8)

    def test_value_only_handle(self):
        self.assertTrue(probe.plain_handle((0, b"ipc", 123, "counter", False)))
        self.assertFalse(probe.plain_handle((0, object())))
        self.assertFalse(probe.plain_handle({"tensor": 1}))

    def test_reply_preserves_cleanup_ack(self):
        left, right = mp.Pipe()
        try:
            reply = {"event": "storage_closed", "safe": True, "error": "setup failed"}
            right.send(reply)
            with self.assertRaises(probe.UnexpectedReply) as caught:
                probe.receive(left, "storage_ready", 1)
            self.assertEqual(caught.exception.reply, reply)
            self.assertIs(caught.exception.conn, left)
        finally:
            left.close()
            right.close()

    def test_native_exception_latches_unknown(self):
        calls = []

        def native(keys, ptrs, sizes):
            calls.append(keys)
            if len(calls) == 1:
                raise RuntimeError("injected native exception")
            return [sum(row) for row in sizes]

        reader = probe.Reader.__new__(probe.Reader)
        reader.unknown = False
        reader.torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda *_: None))
        reader.store = SimpleNamespace(batch_get_into_multi_buffers=native)
        tensor = SimpleNamespace(device=SimpleNamespace(index=0),
                                 numel=lambda: probe.VIEW_BYTES, data_ptr=lambda: 1000)
        with self.assertRaisesRegex(RuntimeError, "native exception"):
            reader.get(tensor, ["a", "b"], "partial", threading.Barrier(1))
        self.assertTrue(reader.unknown)
        reader.get(tensor, ["a", "b"], "partial", threading.Barrier(1))
        self.assertTrue(reader.unknown, "Successful sibling must not clear unknown completion")
        with self.assertRaisesRegex(RuntimeError, "refusing to unregister"):
            reader.close()

    def test_short_native_return_is_known_failure(self):
        reader = probe.Reader.__new__(probe.Reader)
        reader.unknown = False
        reader.torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda *_: None))
        reader.store = SimpleNamespace(batch_get_into_multi_buffers=lambda *_: [0, 0])
        tensor = SimpleNamespace(device=SimpleNamespace(index=0),
                                 numel=lambda: probe.VIEW_BYTES, data_ptr=lambda: 1000)
        with self.assertRaisesRegex(RuntimeError, "GET byte counts"):
            reader.get(tensor, ["a", "b"], "partial", threading.Barrier(1))
        self.assertFalse(reader.unknown)

    def test_real_cpu_fixture_alignment_content_and_retirement(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional CPU Torch buffer test requires torch")
        registered = {}

        def register(ptr, size):
            self.assertEqual(ptr % mmap.PAGESIZE, 0)
            self.assertEqual(size % mmap.PAGESIZE, 0)
            registered[ptr] = size
            return 0

        def put(keys, ptrs, sizes, replica):
            for page, (row, counts) in enumerate(zip(ptrs, sizes)):
                self.assertEqual(len(row), 1)
                self.assertLessEqual(counts[0], registered[row[0]])
                actual = ctypes.string_at(row[0], counts[0])
                self.assertEqual(actual, probe.payload(counts[0], 2, 1, "partial", page))
            return [0] * len(keys)

        reader = probe.Reader.__new__(probe.Reader)
        reader.device, reader.unknown, reader.setup_started = 0, False, True
        reader.host_owners, reader.host_regs, reader.device_regs, reader.keys = [], [], [], []
        reader.replica = object()
        reader.torch = SimpleNamespace(
            npu=SimpleNamespace(set_device=lambda *_: None),
            frombuffer=torch.frombuffer, uint8=torch.uint8,
        )
        reader.store = SimpleNamespace(
            register_buffer=register, batch_put_from_multi_buffers=put,
            unregister_buffer=lambda ptr: registered.pop(ptr) and 0,
            remove=lambda _: 0, close=lambda: None,
        )
        keys = reader.put_fixture("test-only", 2, 1, "partial")
        self.assertEqual(len(keys), 2)
        mappings = [mapping for mapping, _ in reader.host_owners]
        self.assertEqual(reader.close(), [])
        self.assertEqual(registered, {})
        self.assertTrue(all(mapping.closed for mapping in mappings))


class SpawnProtocol(unittest.TestCase):
    def exercise(self, devices=(0, 1), inject=None, timeout=10):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            options = {"devices": list(devices), "iterations": 2, "timeout": timeout,
                       "output": str(output), "inject": inject}
            result = probe.run(options, fake_owner, fake_storage)
            report = json.loads(output.read_text())
        self.assertTrue(report["safe_to_release_owners"])
        return result, report

    def test_real_spawn_round_trip(self):
        result, report = self.exercise()
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["phases"]), 6)

    def test_storage_setup_failure_releases_owners(self):
        result, report = self.exercise(inject="startup")
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_known_read_failure_releases_owners(self):
        result, report = self.exercise(inject="read")
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_data_mismatch_closes_importer_before_owners(self):
        result, report = self.exercise(devices=(0, 9))
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_partial_owner_startup_failure(self):
        result, report = self.exercise(devices=(99, 1))
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")

    def test_timeout_drains_late_completion_before_retirement(self):
        result, report = self.exercise(inject="late", timeout=1)
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["cleanup"]["event"], "storage_closed")


if __name__ == "__main__":
    unittest.main()
