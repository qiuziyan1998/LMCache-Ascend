# SPDX-License-Identifier: Apache-2.0
"""Worker-side cleanup for requests reported finished by the scheduler."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, call

# Third Party
import pytest

pytest.importorskip("lmcache")
pytest.importorskip("vllm")

adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")


def _make_adapter(*, store_async: bool, requests=None):
    engine = MagicMock()
    metadata = adapter_mod.LMCacheConnectorMetadata(requests=requests or [])
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = "kv_both"
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    adapter._wait_for_save_done = True
    adapter._finished_req_ids_waiting_for_save = set()
    adapter._late_finished_sending = set()
    adapter._release_finished_worker_requests = MagicMock()
    return adapter, engine


def test_sync_get_finished_releases_worker_request_state() -> None:
    adapter, engine = _make_adapter(store_async=False)
    engine.get_finished_stores.return_value = None

    result = adapter.get_finished({"req-1"})

    assert result == (None, None)
    engine.get_finished_stores.assert_called_once_with({"req-1"})
    adapter._release_finished_worker_requests.assert_called_once_with({"req-1"})


def test_async_get_finished_releases_only_completed_stores() -> None:
    adapter, engine = _make_adapter(store_async=True)
    engine.get_finished_stores.side_effect = [set(), {"req-1"}]

    assert adapter.get_finished({"req-1"}) == (None, None)
    adapter._release_finished_worker_requests.assert_not_called()

    assert adapter.get_finished(set()) == ({"req-1"}, None)
    adapter._release_finished_worker_requests.assert_called_once_with({"req-1"})


def test_finish_before_wait_releases_after_save_replay() -> None:
    request = SimpleNamespace(
        req_id="req-1",
        token_ids=[1, 2, 3, 4],
        save_spec=SimpleNamespace(can_save=True, skip_leading_tokens=0),
    )
    adapter, engine = _make_adapter(store_async=False, requests=[request])
    adapter._wait_for_save_done = False
    engine.get_finished_stores.return_value = None

    assert adapter.get_finished({"req-1"}) == (None, None)
    assert adapter._finished_req_ids_waiting_for_save == {"req-1"}
    adapter._release_finished_worker_requests.assert_not_called()

    adapter._replay_finished_stores_after_save()

    assert adapter._finished_req_ids_waiting_for_save == set()
    assert engine.get_finished_stores.call_args_list == [call(set()), call({"req-1"})]
    adapter._release_finished_worker_requests.assert_called_once_with({"req-1"})

    adapter._replay_finished_stores_after_save()
    adapter._release_finished_worker_requests.assert_called_once_with({"req-1"})
