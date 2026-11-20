# SPDX-License-Identifier: Apache-2.0
"""Exercise real D2H event registration across chunks and mixed requests."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest


class Stream:
    def __init__(self):
        self.pending = []
        self.completed = 0

    def submit(self, callback):
        self.pending.append(callback)


class Event:
    def record(self, stream):
        self.stream = stream
        self.end = len(stream.pending)

    def synchronize(self):
        while self.stream.completed < self.end:
            self.stream.pending[self.stream.completed]()
            self.stream.completed += 1


@pytest.fixture
def connector():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "layerwise_prefill_store_fences",
            "_layerwise_prefill_transfer_state",
            "_layerwise_prefill_transfer_generation",
            "_check_layerwise_prefill_transfer_generation",
        )
    ]
    store = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "batched_from_gpu"
    )
    # Run the same block that registers completion after production D2H.
    registration = next(
        n
        for n in ast.walk(store)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "deferred_layerwise_put"
        and isinstance(n.body[0], ast.Assign)
        and ast.unparse(n.body[0].targets[0]) == "bank_done_event"
    )
    record = ast.parse(
        "def register(self, kv_group, bank, layer_id, bank_stream, req):\n"
        "    deferred_layerwise_put = prefill_dma = dense_direct = True\n"
        "    layerwise_prefill_generation = (\n"
        "        self._layerwise_prefill_transfer_generation(kv_group))\n"
        "    last_store_events = {}\n"
        "    kwargs = {'req_id': req}\n"
    ).body[0]
    record.body.append(registration)
    module = ast.parse("from __future__ import annotations")
    module.body.extend([*methods, record])
    scope = {"torch": NS(npu=NS(Event=Event))}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    obj = NS()
    for method in [*methods, record]:
        setattr(obj, method.name, MethodType(scope[method.name], obj))
    return obj


@pytest.mark.parametrize("banks", [2, 4, 8])
@pytest.mark.parametrize("group,layers", [(0, 79), (1, 22)])
def test_publication_waits_latest_save_on_fixed_banks_across_chunks(
    connector, banks, group, layers
):
    streams = [Stream() for _ in range(banks)]
    copied = set()
    for chunk in range(4):
        for layer in range(layers):
            bank = layer % banks
            streams[bank].submit(
                lambda layer=layer, chunk=chunk: copied.add((chunk, layer))
            )
            connector.register(group, bank, layer, streams[bank], "request")
        fences = connector.layerwise_prefill_store_fences(group)
        assert len(fences) == banks
        # This is what the Mooncake worker does before reading host memory.
        for fence in fences:
            fence.synchronize()
        assert {(chunk, layer) for layer in range(layers)} <= copied, (
            "Publication observed unfinished D2H after a chunk change"
        )


@pytest.mark.parametrize("banks", [2, 4, 8])
def test_mixed_request_publication_fences_every_save_on_shared_bank_streams(
    connector, banks
):
    streams = [Stream() for _ in range(banks)]
    copied = set()
    # First a different batch leaves high layer ordinals in the registry.
    for layer in range(79):
        connector.register(0, layer % banks, layer, streams[layer % banks], "previous")
    # Requests share bank streams, but retain separate KV storage.
    for layer in range(79):
        for req in ("A", "B", "C"):
            bank = layer % banks
            streams[bank].submit(lambda req=req, layer=layer: copied.add((req, layer)))
            connector.register(0, bank, layer, streams[bank], req)
    for fence in connector.layerwise_prefill_store_fences(0):
        fence.synchronize()
    assert copied == {(req, layer) for req in ("A", "B", "C") for layer in range(79)}


def test_old_generation_and_other_group_are_not_publication_fences(connector):
    stream = Stream()
    connector.register(0, 0, 78, stream, "old")
    connector.register(1, 0, 21, stream, "indexer")
    connector._layerwise_prefill_transfer_generations[0] += 1
    assert connector.layerwise_prefill_store_fences(0) == ()
    assert len(connector.layerwise_prefill_store_fences(1)) == 1


def test_no_store_has_no_publication_fences(connector):
    assert connector.layerwise_prefill_store_fences(0) == ()
