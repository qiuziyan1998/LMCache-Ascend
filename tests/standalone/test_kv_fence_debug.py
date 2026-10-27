# SPDX-License-Identifier: Apache-2.0
"""Boundary isolation without payload copies or device-dependent imports."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


_PATH = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/kv_fence_debug.py"
_SPEC = importlib.util.spec_from_file_location("kv_fence_debug_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
KVFenceDebug = _MODULE.KVFenceDebug


def request(name, offset=0, cached=1024, resident=0, can_load=True, sparse=False):
    return NS(
        req_id=name,
        layerwise_prefill_bank_offset=offset,
        token_ids=range(max(cached, 1) + 1),
        is_sparse_decode=sparse,
        load_spec=NS(
            can_load=can_load, lmcache_cached_tokens=cached, vllm_cached_tokens=resident
        ),
    )


def debug(mode="check", role="P", **env):
    output = []
    instance = KVFenceDebug(
        role, 3, {"LMCACHE_PREFILL_FENCE_DEBUG": mode, **env}, output.append
    )
    return instance, output


class Fence:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1

    def query(self):
        pytest.fail("No event query is allowed")


def test_disabled_factory_does_not_enable_or_validate_other_debug(monkeypatch):
    monkeypatch.setenv("LMCACHE_PREFILL_FENCE_DEBUG", "0")
    monkeypatch.setenv("LMCACHE_PREFILL_STORE_DEBUG", "1")
    assert KVFenceDebug.from_environment("P", 0) is None


@pytest.mark.parametrize(
    "env",
    [
        {"LMCACHE_PREFILL_STORE_DEBUG": "1"},
        {"LMCACHE_PREFILL_AUDIT": "1"},
        {"LMCACHE_PREFILL_FENCE_GROUPS": "2"},
        {"LMCACHE_PREFILL_FENCE_LAYERS": "39:39"},
        {"LMCACHE_PREFILL_FENCE_LAYERS": "-1:39"},
        {"LMCACHE_PREFILL_FENCE_LAYERS": "39"},
    ],
)
def test_reject_ambiguous_experiments(env):
    with pytest.raises(ValueError):
        debug(**env)


def test_check_has_no_device_operations():
    d, output = debug()
    d.begin_step([request("a"), request("b")], (1, 1), 4, 1024)
    a, b = Fence(), Fence()
    d.loaded("a", 0, 0, 0, a)
    d.loaded("b", 0, 0, 0, b)
    d.before_read(0, 0, 0, b)
    for stage in ("p_load", "p_use", "p_save", "d_load"):
        d.wait(stage, 0, 0, b)
    assert a.waits == b.waits == 0
    assert len(output) == 1  # Only startup; no per-token/layer spam.


def test_fence_only_does_not_change_protocol_or_build_expected_metadata():
    d, _ = debug("p_use")
    d.begin_step(None, None, 0, 0)
    d.loaded("a", 0, 0, 0, None)
    d.before_read(0, 0, 0, None)
    assert not d.expected and not d.submitted and not d.begun
    event = Fence()
    d.wait("p_use", 0, 0, event)
    assert event.waits == 1


def test_missing_second_request_on_same_bank_is_rejected_before_compute():
    d, output = debug()
    d.begin_step([request("a"), request("b")], (1, 1), 4, 1024)
    a = Fence()
    d.loaded("a", 0, 0, 0, a)
    with pytest.raises(RuntimeError, match="request_load_mismatch"):
        d.before_read(0, 0, 0, a)
    assert output[-1]["missing"] == ["b"]
    assert a.waits == 0


def test_shared_load_stream_last_event_covers_different_banks():
    d, _ = debug()
    d.begin_step([request("a"), request("b", offset=1)], (1, 0), 4, 1024)
    a, b = Fence(), Fence()
    d.loaded("a", 0, 0, 0, a, shared_stream=True)
    d.loaded("b", 0, 0, 1, b, shared_stream=True)
    d.before_read(0, 0, 0, b)
    d.before_read(0, 0, 1, None)  # The one shared event was consumed above.
    assert a.waits == b.waits == 0


def test_shared_load_stream_still_requires_its_final_event():
    d, _ = debug()
    d.begin_step([request("a"), request("b", offset=1)], (1, 0), 4, 1024)
    a, b = Fence(), Fence()
    d.loaded("a", 0, 0, 0, a, shared_stream=True)
    d.loaded("b", 0, 0, 1, b, shared_stream=True)
    with pytest.raises(RuntimeError, match="wrong_load_event"):
        d.before_read(0, 0, 0, a)


def test_raw_bank_and_shared_bank_require_their_respective_events():
    d, _ = debug()
    d.begin_step([request("a"), request("b", offset=1)], (1, 0), 4, 1024)
    a, b = Fence(), Fence()
    d.loaded("a", 0, 0, 0, a)
    d.loaded("b", 0, 0, 1, b, shared_stream=True)
    d.before_read(0, 0, 0, a)
    with pytest.raises(RuntimeError, match="wrong_load_event"):
        d.before_read(0, 0, 1, None)
    d.before_read(0, 0, 1, b)


@pytest.mark.parametrize("returned", [None, Fence()])
def test_missing_or_old_event_cannot_silently_use_save_fallback(returned):
    d, output = debug()
    d.begin_step([request("a")], (1, 1), 4, 1024)
    d.loaded("a", 0, 0, 0, Fence())
    with pytest.raises(RuntimeError, match="wrong_load_event"):
        d.before_read(0, 0, 0, returned)
    assert output[-1]["requests"] == ["a"]


def test_real_miss_and_resident_prefix_do_not_require_load():
    d, _ = debug()
    d.begin_step(
        [
            request("cold", cached=0),
            request("resident", resident=1024),
            request("skip", can_load=False),
            request("decode", sparse=True),
        ],
        (79, 22),
        4,
        1024,
    )
    d.before_read(0, 0, 0, None)
    assert not d.expected


def test_partial_resident_chunk_requires_load_and_mixed_phases_are_independent():
    d, _ = debug()
    d.begin_step([request("a", 0, resident=511), request("b", 3)], (5, 2), 4, 1024)
    a, b = Fence(), Fence()
    d.loaded("a", 0, 4, 0, a)
    d.loaded("b", 0, 4, 3, b)
    d.before_read(0, 4, 0, a)
    d.before_read(0, 4, 3, b)
    d.before_read(0, 4, 0, None)  # Repeated same-stream join is legitimate.
    with pytest.raises(RuntimeError, match="load_after_consume"):
        d.loaded("a", 0, 4, 0, a)


def test_new_step_cannot_reuse_old_completion():
    d, _ = debug()
    for _ in range(2):
        d.begin_step([request("a")], (1, 1), 4, 1024)
        assert not d.latest and not d.consumed
        with pytest.raises(RuntimeError, match="request_load_mismatch"):
            d.before_read(0, 0, 0, None)
        event = Fence()
        d.loaded("a", 0, 0, 0, event)
        d.before_read(0, 0, 0, event)


def test_role_stage_group_and_exclusive_layer_filter():
    d, output = debug(
        "all", LMCACHE_PREFILL_FENCE_GROUPS="1", LMCACHE_PREFILL_FENCE_LAYERS="10:12"
    )
    event = Fence()
    for stage in ("p_load", "p_use", "p_save", "d_load"):
        for group in (0, 1):
            for layer in (9, 10, 11, 12):
                d.wait(stage, group, layer, event)
    assert event.waits == 6
    assert len(output) == 4
    decoder, _ = debug("all", role="D")
    decoder.wait("p_load", 0, 0, event)
    decoder.wait("d_load", 0, 0, event)
    assert event.waits == 7


def test_device_failure_identifies_exact_boundary():
    d, output = debug("p_save")

    def failed():
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device_fence_failed"):
        d.wait("p_save", 1, 21, NS(synchronize=failed), req="r", bank=3)
    assert output[-1]["req"] == "r"
    assert output[-1]["bank"] == 3
    assert output[-1]["stage"] == "p_save"


@pytest.mark.parametrize("stage", ["p_load", "p_use", "p_save"])
def test_fence_completes_only_selected_snapshot_not_later_work(stage):
    from test_layerwise_prefill_dma import _AsyncNPU, _AsyncStream

    stream = _AsyncStream()
    actions = []
    stream.enqueue(lambda: actions.append("selected operation"))
    snapshot = _AsyncNPU.Event()
    snapshot.record(stream)
    stream.enqueue(lambda: actions.append("later layer"))
    snapshot.synchronize = snapshot.tail.run
    d, _ = debug(stage)
    d.wait(stage, 0, 0, snapshot)
    assert actions == ["selected operation"]


@pytest.mark.parametrize("group", [0, 1, "c8"])
@pytest.mark.parametrize("banks", [2, 4, 8])
def test_real_bank_callbacks_keep_protocol_and_payload_order(group, banks):
    from test_layerwise_prefill_dma import _async_bank_connector

    api = _async_bank_connector(group, 1, bank_count=banks, layers=banks + 2)
    d, _ = debug()
    api.connector.fence_debug = d
    physical_group = 1 if group == "c8" else group
    counts = (banks + 2, banks + 2)
    for chunk in range(3):
        d.begin_step([request("request", 1)], counts, banks, 4)
        generator, load = api.start(chunk)
        for layer in range(banks):
            load(layer)
        for layer in range(banks + 2):
            api.read(chunk, layer)
            api.save(layer)
            if layer + banks < banks + 2:
                load(layer + banks)
        generator.close()
    api.npu.compute.tail.run()
    assert api.observed == [
        (chunk, layer) for chunk in range(3) for layer in range(banks + 2)
    ]
    assert not any(key[0] == physical_group for key in d.latest)


def test_real_wait_rejects_dropped_record_instead_of_using_old_save():
    from test_layerwise_prefill_dma import _async_bank_connector

    api = _async_bank_connector(0, 0)
    d, output = debug()
    api.connector.fence_debug = d
    d.begin_step([request("request")], (4, 4), 2, 4)
    generator, load = api.start(0)
    load(0)
    api.connector._layerwise_prefill_transfer_state()[3].pop((0, 0, 0))
    with pytest.raises(RuntimeError, match="wrong_load_event"):
        api.read(0, 0)
    assert not api.observed
    assert output[-1]["requests"] == ["request"]
    generator.close()
