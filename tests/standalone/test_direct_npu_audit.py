# SPDX-License-Identifier: Apache-2.0
"""Direct-import audit tests with delayed native copies, no accelerator."""

from contextlib import contextmanager
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).parents[2]
    common_path = root.parent / "LMCache/lmcache/v1/prefill_audit.py"
    spec = importlib.util.spec_from_file_location(
        "lmcache.v1.prefill_audit", common_path
    )
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, common)
    spec.loader.exec_module(common)
    path = root / "lmcache_ascend/v1/direct_npu_audit.py"
    spec = importlib.util.spec_from_file_location("direct_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    records = []
    module.audit_record = lambda kind, **fields: records.append(
        dict(kind=kind, **fields)
    )

    class Stream:
        def __init__(self):
            self.tasks, self.completed = [], 0

        def finish(self):
            for task in self.tasks[self.completed :]:
                task()
                self.completed += 1

    class Event:
        def record(self, stream):
            self.stream, self.target = stream, len(stream.tasks)

        def query(self):
            return self.stream.completed >= self.target

        def synchronize(self):
            raise AssertionError("readback must never wait on host")

    stream = Stream()

    @contextmanager
    def use_stream(value):
        assert value is stream
        yield

    api = SimpleNamespace(
        empty=lambda size, **kwargs: torch.full((size,), 255, dtype=torch.uint8),
        uint8=torch.uint8,
        npu=SimpleNamespace(Event=Event, stream=use_stream),
    )
    copies = []

    def copy(rows, to_cpu):
        assert to_cpu
        assert all(dst % 64 == 0 and src % 64 == 0 for dst, src, _ in rows)
        copies.extend(rows)
        stream.tasks.append(
            lambda: [ctypes.memmove(dst, src, size) for dst, src, size in rows]
        )

    reader = module.DirectNpuAudit(api, copy)
    return common, reader, stream, records, copies


def test_readback_is_bounded_delayed_and_matches_cpu_samples(runtime):
    common, reader, stream, records, copies = runtime
    buffers = [torch.arange(256, dtype=torch.uint8) for _ in range(100)]
    keys = [f"key-{i}" for i in range(len(buffers))]
    pointers = [[buf.data_ptr(), buf.data_ptr() + 7] for buf in buffers]
    sizes = [[7, 249] for _ in buffers]
    reader.submit("request", keys, pointers, sizes, stream, buffers)
    assert sum(row[2] for row in copies) <= 3072
    assert len(reader.pending[0].keys) <= 3
    reader.poll()
    assert not records
    assert bool((reader.pending[0].buffer == 255).all())
    stream.finish()
    reader.poll()
    record = records[-1]
    assert record["stage"] == "get_npu"
    selected = common.sampled_key_indices(keys, limit=3)
    assert record["samples"] == common.cpu_payload_samples(
        [pointers[i] for i in selected],
        [sizes[i] for i in selected],
    )
    assert not reader.pending


def test_receive_corruption_changes_the_readback(runtime):
    common, reader, stream, records, _ = runtime
    source = torch.zeros(256, dtype=torch.uint8)
    pointers, sizes = [[source.data_ptr()]], [[256]]
    expected = common.cpu_payload_samples(pointers, sizes)
    reader.submit("request", ["key"], pointers, sizes, stream, [source])
    source[0] = 7
    stream.finish()
    reader.poll()
    assert records[-1]["samples"] != expected


def test_partial_submission_error_keeps_host_storage_alive(runtime):
    _, reader, stream, records, _ = runtime
    source = torch.zeros(64, dtype=torch.uint8)
    original = reader.copy

    def fail_after_submit(rows, to_cpu):
        original(rows, to_cpu)
        raise RuntimeError("partial native submission")

    reader.copy = fail_after_submit
    reader.submit("request", ["key"], [[source.data_ptr()]], [[64]], stream, [source])
    assert records[-1]["error"] == "direct_readback_submit_failed"
    assert len(reader.pending) == 1 and not reader.pending[0].armed
    stream.finish()
    reader.poll()
    assert reader.pending  # quarantined; never free a possible DMA destination


def test_alignment_never_reads_beyond_owned_storage(runtime):
    _, reader, stream, records, copies = runtime
    source = torch.zeros(8, dtype=torch.uint8)
    reader.submit("request", ["key"], [[source.data_ptr()]], [[8]], stream, [source])
    assert not copies
    assert records[-1]["error"] == "direct_readback_submit_failed"


def test_real_direct_import_hook_runs_before_readiness(tmp_path):
    root = Path(__file__).parents[2]
    script = r"""
import json,sys,types
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
from cpu_vllm_runtime import CPUHardware,prepare_paths
prepare_paths(Path.cwd())
hw=CPUHardware(Path.cwd()).install()
import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.prefill_audit import cpu_payload_samples
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector as GPU,
)
target=torch.zeros(512,dtype=torch.uint8)
hw.queue.memory(target,target)
gpu=types.SimpleNamespace(load_stream=hw.Stream())
for name in ('audit_direct_indexer_import','record_dense_load_readiness',
             'consume_dense_load_readiness'):
    setattr(gpu,name,types.MethodType(getattr(GPU,name),gpu))
gpu.plan_direct_page_destinations=lambda *args: (
    [[target.data_ptr()]],[[512]],(target,)
)
engine=object.__new__(AscendLMCacheEngine)
engine.gpu_connector=gpu
engine._force_layerwise_prefill_store=False
def load(*args):
    target.copy_(torch.arange(512,dtype=torch.int64).remainder(251).to(torch.uint8))
engine.storage_manager=types.SimpleNamespace(batched_get_external_pages=load)
key=CacheEngineKey('model',1,0,123,torch.float16,{},kv_group=1)
layers=[[item] for item in key.split_layers(2)]
kwargs={'kvcaches':[target], 'slot_mapping':torch.arange(128)}
def exercise(request):
    return engine._try_direct_indexer_page_load(
        req_id=request,keys_layer_major=layers,starts=[0],ends=[128],
        retrieve_kwargs=kwargs,
    )
assert exercise('request')
reader=gpu.direct_npu_audit
assert reader.pending
hw.queue.until(hw.queue.current.tail)
reader.poll()
assert not reader.pending
expected=cpu_payload_samples([[target.data_ptr()]],[[512]])
records=[json.loads(line) for path in Path(sys.argv[1]).glob('*.jsonl')
         for line in path.read_text().splitlines()]
assert any(row.get('stage')=='get_npu' and
           row['samples']==[list(pair) for pair in expected] for row in records)
def failed_audit(*args):
    raise ValueError('diagnostic failure')
gpu.audit_direct_indexer_import=failed_audit
assert exercise('request2')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=root,
        env=dict(
            os.environ,
            LMCACHE_PREFILL_AUDIT="1",
            LMCACHE_PREFILL_STORE_DEBUG="0",
            LMCACHE_PREFILL_AUDIT_DIR=str(tmp_path),
        ),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
