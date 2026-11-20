# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prepare_layerwise_prefill_group0_skip."""

# Standard
from types import SimpleNamespace
import unittest

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
)


class TestPrepareLayerwisePrefillGroup0Skip(unittest.TestCase):
    def test_prepare_layerwise_prefill_group0_skip_avoids_staging(self) -> None:
        conn = VLLMPagedMemLayerwiseNPUConnector.__new__(
            VLLMPagedMemLayerwiseNPUConnector
        )
        conn.kvcaches = {"group0": ["k0"], "group1": ["k1"]}  # type: ignore[assignment]
        conn._layerwise_prefill_dma = True

        init_ptr_calls: list[dict] = []
        lazy_init_calls: list[tuple[object, int, bool]] = []
        bank_counts: dict[int, int] = {1: 4}
        group1_staging = object()
        staging_by_group: dict[int, object] = {1: group1_staging}

        conn.initialize_kvcaches_ptr = (  # type: ignore[method-assign]
            lambda **kw: init_ptr_calls.append(kw)
        )

        def _fake_lazy_init(
            kvcaches: object, *, kv_group: int, init_staging: bool
        ) -> SimpleNamespace:
            lazy_init_calls.append((kvcaches, kv_group, init_staging))
            if init_staging:
                staging_by_group[kv_group] = object()
            return SimpleNamespace(kv_group=kv_group, init_staging=init_staging)

        conn._lazy_initialize_buffer_with_staging = _fake_lazy_init  # type: ignore[method-assign]
        conn._layerwise_prefill_transfer_state = lambda: (  # type: ignore[method-assign]
            bank_counts,
            {},
            {},
            {},
        )

        conn.prepare_layerwise_prefill_group0_skip(
            layerwise_prefill_bank_count=3,
            marker="forwarded",
        )

        self.assertEqual(len(init_ptr_calls), 1)
        self.assertEqual(init_ptr_calls[0]["layerwise_prefill_bank_count"], 3)
        self.assertEqual(lazy_init_calls, [(conn.kvcaches, 0, False)])
        self.assertNotIn(0, staging_by_group)
        self.assertIs(staging_by_group[1], group1_staging)
        self.assertEqual(bank_counts[0], 3)
        self.assertEqual(bank_counts[1], 4)

        # Fallback bank count = 2 when _layerwise_prefill_dma is False
        conn._layerwise_prefill_dma = False
        bank_counts.pop(0)
        snap = {"custom": ["kv"]}
        conn.prepare_layerwise_prefill_group0_skip(
            kvcaches=snap,
            layerwise_prefill_bank_count=4,
        )
        self.assertEqual(lazy_init_calls[-1], (snap, 0, False))
        self.assertEqual(bank_counts[0], 2)
        self.assertNotIn(0, staging_by_group)


if __name__ == "__main__":
    unittest.main()
