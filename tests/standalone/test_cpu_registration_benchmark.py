"""Host-only checks; no allocator calls or NPU initialization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, call, patch

SOURCE = Path(__file__).resolve().parents[2] / "benchmark/v1/kv_transfer/benchmark_cpu_registration.py"
spec = importlib.util.spec_from_file_location("registration_bench", SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class RegistrationBenchmarkTests(unittest.TestCase):
    def test_cgroup_v1_bind_mount_root(self):
        mounts = "31 20 0:27 /docker/abc /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory"
        rows = list(bench.memory_cgroups("9:memory:/docker/abc/child", mounts))
        self.assertEqual(rows, [(1, Path("/sys/fs/cgroup/memory"),
                                Path("/sys/fs/cgroup/memory/child"))])

    def test_cgroup_v2_namespace_root_and_escaped_mount(self):
        mounts = r"31 20 0:27 / /sys/fs/cgroup\040test rw - cgroup2 cgroup rw"
        rows = list(bench.memory_cgroups("0::/", mounts))
        self.assertEqual(rows, [(2, Path("/sys/fs/cgroup test"), Path("/sys/fs/cgroup test"))])

    def test_cgroup_unrelated_mount_does_not_read_wrong_limits(self):
        mounts = "31 20 0:27 /docker/other /cg rw - cgroup cgroup rw,memory"
        self.assertEqual(list(bench.memory_cgroups("9:memory:/docker/abc", mounts)), [])

    def test_cgroup_reads_visible_ancestor_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "child"
            child.mkdir()
            (root / "memory.limit_in_bytes").write_text("200")
            (child / "memory.limit_in_bytes").write_text("999")
            (child / "memory.failcnt").write_text("3")
            with patch.object(bench, "memory_cgroups", return_value=[(1, root, child)]):
                data = bench.cgroup_snapshot("", "")
            self.assertEqual(data[str(root)]["memory.limit_in_bytes"], "200")
            self.assertEqual(data[str(child)]["memory.failcnt"], "3")

    def test_shm_limit_stops_before_worker_creation_even_with_chunks(self):
        args = self.args()
        args.chunk_bytes = 40
        with patch.object(bench, "snapshot", return_value={"shm_available_bytes": 200}), \
                patch.object(bench.os, "sysconf", create=True, return_value=1), \
                patch.object(bench.mp, "get_context") as spawn, \
                patch.object(bench, "emit") as emit:
            self.assertFalse(bench.trial(args, 240))
        spawn.assert_not_called()
        self.assertEqual(emit.call_args.kwargs["failure_stage"], "shm_capacity_preflight")
        self.assertFalse(emit.call_args.kwargs["registration_attempted"])

    def args(self, mode="shared", attach="auto"):
        return SimpleNamespace(mode=mode, attach=attach, numa_node=1, interleave_nodes=[0, 1],
                               npu_free_gib=None, npu_preload_chunk_gib=1)

    def test_preload_retains_chunked_tensors_and_synchronizes(self):
        torch = Mock()
        torch.npu.mem_get_info.return_value = (11, 20)
        tensors = []
        self.assertEqual(bench.preload_npu(torch, 0, 4, 3, tensors), 7)
        self.assertEqual([c.args[0] for c in torch.empty.call_args_list], [3, 3, 1])
        self.assertEqual(len(tensors), 3)
        torch.npu.synchronize.assert_called_once()

    def test_preload_reports_insufficient_initial_headroom(self):
        torch = Mock()
        torch.npu.mem_get_info.return_value = (3, 20)
        with self.assertRaises(RuntimeError):
            bench.preload_npu(torch, 0, 4, 3, [])
        torch.empty.assert_not_called()

    def test_partial_preload_failure_preserves_existing_tensor(self):
        torch = Mock()
        torch.npu.mem_get_info.return_value = (11, 20)
        tensor = Mock()
        torch.empty.side_effect = [tensor, RuntimeError("OOM")]
        tensors = []
        with self.assertRaises(RuntimeError):
            bench.preload_npu(torch, 0, 4, 3, tensors)
        self.assertEqual(tensors, [tensor])

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
