"""Host-only checks; no allocator calls or NPU initialization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

SOURCE = Path(__file__).resolve().parents[2] / "benchmark/v1/kv_transfer/benchmark_cpu_registration.py"
spec = importlib.util.spec_from_file_location("registration_bench", SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class RegistrationBenchmarkTests(unittest.TestCase):
    def args(self, mode="shared", attach="auto"):
        return SimpleNamespace(mode=mode, attach=attach, numa_node=1, interleave_nodes=[0, 1])

    def test_partial_last_region_and_single_slab(self):
        self.assertEqual(bench.regions(11, 4), [4, 4, 3])
        self.assertEqual(bench.regions(11, 0), [11])
        with self.assertRaises(ValueError):
            bench.regions(0, 4)

    def test_owner_uses_real_shared_allocator(self):
        ops = Mock()
        bench.allocate(ops, self.args(), 16, "/test", True)
        ops.alloc_shm_pinned_ptr.assert_called_once_with(16, "/test", [0, 1])
        bench.release(ops, self.args(), 123, 16, "/test", True)
        ops.free_shm_pinned_ptr.assert_called_once_with(123, 16, "/test")

    def test_auto_attach_retries_writable_only_on_failure(self):
        ops = Mock()
        ops.attach_shm_pinned_ptr.side_effect = [RuntimeError("readonly refused"), 123]
        self.assertEqual(bench.allocate(ops, self.args(), 16, "/test", False), 123)
        self.assertEqual(ops.attach_shm_pinned_ptr.call_args_list,
                         [call(16, "/test", False), call(16, "/test", True)])

    def test_explicit_readonly_does_not_retry(self):
        ops = Mock()
        ops.attach_shm_pinned_ptr.side_effect = RuntimeError("refused")
        with self.assertRaises(RuntimeError):
            bench.allocate(ops, self.args(attach="readonly"), 16, "/test", False)
        ops.attach_shm_pinned_ptr.assert_called_once_with(16, "/test", False)

    def test_passive_detaches_without_unlinking_owner(self):
        ops = Mock()
        bench.release(ops, self.args(), 123, 16, "/test", False)
        ops.detach_shm_pinned_ptr.assert_called_once_with(123, 16)
        ops.free_shm_pinned_ptr.assert_not_called()

    def test_private_and_numa_pairs(self):
        ops = Mock()
        for mode in ("private", "numa"):
            bench.allocate(ops, self.args(mode), 16, "/test", True)
            bench.release(ops, self.args(mode), 123, 16, "/test", True)
        ops.alloc_pinned_ptr.assert_called_once_with(16, 0)
        ops.free_pinned_ptr.assert_called_once_with(123)
        ops.alloc_pinned_numa_ptr.assert_called_once_with(16, 1)
        ops.free_pinned_numa_ptr.assert_called_once_with(123, 16)

    def test_partial_failure_releases_successful_allocations(self):
        ops = Mock()
        ops.alloc_shm_pinned_ptr.side_effect = [123, RuntimeError("capacity")]
        fake_torch = SimpleNamespace(__version__="test", npu=Mock())
        modules = {"torch": fake_torch,
                   "torch_npu": SimpleNamespace(__version__="test"),
                   "lmcache_ascend": SimpleNamespace(c_ops=ops)}
        ops.__file__ = "test-extension"
        conn = Mock()
        with patch.dict(bench.sys.modules, modules), patch.dict(bench.os.environ), \
                patch.object(bench, "snapshot", return_value={}), \
                patch.object(bench, "emit"), self.assertRaises(SystemExit):
            bench.worker(self.args(), 0, [16, 16], ["/a", "/b"], True, conn)
        ops.free_shm_pinned_ptr.assert_called_once_with(123, 16, "/a")
        conn.send.assert_not_called()
        conn.close.assert_called_once()

    def test_mapping_failure_still_releases_allocation(self):
        ops = Mock()
        ops.__file__ = "test-extension"
        ops.alloc_shm_pinned_ptr.return_value = 123
        ops.get_device_ptr.return_value = 0
        modules = {"torch": SimpleNamespace(__version__="test", npu=Mock()),
                   "torch_npu": SimpleNamespace(__version__="test"),
                   "lmcache_ascend": SimpleNamespace(c_ops=ops)}
        with patch.dict(bench.sys.modules, modules), patch.dict(bench.os.environ), \
                patch.object(bench, "snapshot", return_value={}), \
                patch.object(bench, "emit"), self.assertRaises(SystemExit):
            bench.worker(self.args(), 0, [16], ["/a"], True, Mock())
        ops.free_shm_pinned_ptr.assert_called_once_with(123, 16, "/a")


if __name__ == "__main__":
    unittest.main()
