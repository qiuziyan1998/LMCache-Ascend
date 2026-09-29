# SPDX-License-Identifier: Apache-2.0
"""Exercise real prepared bank packet methods with CPU address/event doubles."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def methods():
    path = Path(__file__).parents[2] / 'lmcache_ascend/v1/npu_connector/npu_connectors.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VLLMPagedMemLayerwiseNPUConnector')
    names = {'_prepare_prefill_c8_packets', '_submit_prefill_c8_packet'}
    module = ast.parse('from __future__ import annotations')
    module.body.append(ast.ClassDef(name='Connector', bases=[], keywords=[],
        body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names],
        decorator_list=[]))
    return module


@pytest.mark.parametrize('different_capacity', [False, True])
@pytest.mark.parametrize('pages', [False, True])
@pytest.mark.parametrize('bank', [0, 1])
@pytest.mark.parametrize('direction', [False, True])
def test_mixed_packets_preserve_physical_tail_scale_offset_and_bank(bank, direction, different_capacity, pages, monkeypatch):
    events = []
    class Stream:
        def __init__(self, name): self.name = name
        def wait_stream(self, stream): events.append(('wait_stream', self.name, stream.name))
        def wait_event(self, event): events.append(('wait_event', self.name, event.stream.name))
    class Event:
        def record(self, stream): self.stream = stream
    compute, prep = Stream('compute'), Stream('prep')
    banks = [Stream('bank0'), Stream('bank1')]
    monkeypatch.setattr(torch, 'npu', NS(current_stream=lambda: compute, Event=Event), raising=False)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda self, stream: events.append(('retain', stream.name)))
    backing = torch.full((12, 128, 1, 128), -1, dtype=torch.bfloat16)
    key = backing.view(torch.int8).reshape(24, 128, 1, 128)
    scales = torch.full((24, 128, 1, 1), -7, dtype=torch.float16)
    caches = [(backing,), (key, scales)]
    policy = NS(mixed=True, is_c8=lambda layer: layer == 1)
    layout = NS(layer_token_bytes=(256, 130), layer_slot_factors=(1, 2), kv_device='cpu')
    # First packet has a full physical capacity of 4 but only 2 valid rows.
    layer_capacities = [(4, 3), (5 if different_capacity else 4, 3)]
    packets = [[torch.zeros(capacity * width, dtype=torch.uint8) for capacity in capacities]
               for width, capacities in zip(layout.layer_token_bytes, layer_capacities)]
    class Page:
        def __init__(self, tensor): self.tensor = tensor
        def layer_size_bytes(self, layer): return self.tensor.numel()
    objects = [[Page(tensor) if pages else NS(tensor=tensor)
                for tensor in row] for row in packets]
    calls = []
    ops = NS(IndexerC8State=lambda *args: args,
             indexer_c8_transfer_prepared=lambda *args, **kwargs: calls.append((args, kwargs)))
    def layer_tensor(obj, layer):
        assert not isinstance(obj, Page), 'Page size must not construct a tensor view'
        return obj.tensor
    scope = dict(torch=torch, lmc_ops=ops, LayerPageMemoryObj=Page,
        _layer_source_memory_objs=lambda objs, layer: objs,
        _layer_memory_tensor=layer_tensor)
    exec(compile(ast.fix_missing_locations(methods()), '<actual connector methods>', 'exec'), scope)
    obj = scope['Connector']()
    obj.indexer_c8_layout = policy
    obj.indexer_hbm_block_map = None
    obj._stream_context_or_null = lambda stream: nullcontext()
    obj._layerwise_prefill_dma_stream = lambda group, bank: banks[bank]
    obj._resolve_registered_cpu_source_device_ptr = lambda memory_obj, **kw: memory_obj.tensor.data_ptr()
    prepared = obj._prepare_prefill_c8_packets(layout=layout, kvcaches=caches,
        memory_objs=objects, block_ids_by_bank=((1, 4), (7, 10)), block_size=128,
        starts=(0, 128), ends=(2, 131), transfer_stream=prep)
    assert events == [('wait_stream', 'prep', 'compute'),
                      ('wait_event', 'bank0', 'prep'), ('wait_event', 'bank1', 'prep')]
    for layer in range(2):
        state, pointers, offsets, counts, maps, capacity = prepared[layer]
        first_capacity = layer_capacities[layer][0]
        assert counts.tolist() == [first_capacity, 3]
        assert offsets.tolist() == [0, first_capacity]
        assert capacity == first_capacity
        first, second = ((1, 4), (7, 10))[bank]
        assert maps[bank].tolist() == ([first*128, first*128+1]
            + [-1] * (first_capacity - 2)
            + [second*128, second*128+1, second*128+2])
        assert state[2] == (1 if layer == 0 else 2)
        obj._submit_prefill_c8_packet(prepared, layer, bank, banks[bank], direction)
        args, kwargs = calls[-1]
        assert args[0] is state and args[4] is maps[bank]
        assert args[-1] is direction and kwargs == {'fixed_chunks': False}
    for operand in (2, 3, 4):
        assert (prepared[0][operand] is prepared[1][operand]) is not different_capacity
    assert prepared[0][1] is not prepared[1][1]  # per-owner pointers stay distinct
    # Submissions retain all four operands and never take a later compute snapshot.
    assert len(events) == 11
    assert all(event == ('retain', banks[bank].name) for event in events[3:])
    # CPU address proof for factor 2: key rows occupy first half of a BF16 block.
    before = backing.view(torch.uint8).clone()
    key[2*first, :2].fill_(23)
    assert torch.equal(backing.view(torch.uint8)[first, 64:], before[first, 64:])
    assert torch.equal(backing.view(torch.uint8)[7 if bank == 0 else 1], before[7 if bank == 0 else 1])


def test_uniform_banked_c8_requires_explicit_capacity():
    scope = {}
    exec(compile(ast.fix_missing_locations(methods()), '<actual connector methods>', 'exec'), scope)
    obj = scope['Connector']()
    obj.indexer_c8_layout = NS(mixed=False)
    with pytest.raises(ValueError, match='explicit logical capacity'):
        obj._prepare_prefill_c8_packets(layout=None, kvcaches=None, memory_objs=None,
            block_ids_by_bank=None, block_size=128, starts=None, ends=None, transfer_stream=None)
