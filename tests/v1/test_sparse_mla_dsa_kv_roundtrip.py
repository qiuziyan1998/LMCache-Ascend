# SPDX-License-Identifier: Apache-2.0
"""Store/retrieve roundtrip via sparse_mla_dsa_batched_direct_kv_transfer."""
from __future__ import annotations

# Standard
import sys
from pathlib import Path
from unittest.mock import patch

# Third Party
import pytest
import torch

_BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "benchmark" / "v1" / "kv_transfer"
)
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

from kv_cache_fixtures import (  # noqa: E402
    check_paged_kv_cache_equal,
    generate_dsa_kv_cache,
    generate_mla_kv_cache,
)
from load_benchmark_utils import (  # noqa: E402
    KV_FORMAT_DSA,
    KV_FORMAT_MLA,
    build_load_benchmark_harness,
    compute_chunk_partition,
    run_all_direct_fast_layers,
    run_all_direct_layers,
    verify_load_benchmark_harness,
)

# First Party
from lmcache.v1.kv_checksum_diag import (  # noqa: E402
    collect_kv_checksum_samples,
    log_sparse_scatter_entry,
    reset_kv_checksum_diag_state,
    sample_layer_ids,
    sample_token_indices,
    slot_kv_fingerprints,
)


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _check_kwargs(
    fmt: str,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dsa_head_dim: int,
) -> dict:
    return dict(
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
    )


class TestSparseScatterEntryLogging:
    """log_sparse_scatter_entry is wired at scatter kernel entry."""

    def test_log_sparse_scatter_entry_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LMCACHE_DIAG_KV_CHECKSUM", "1")
        reset_kv_checksum_diag_state()
        with patch("lmcache.v1.kv_checksum_diag.log_kv_checksum_diag") as mock_log:
            log_sparse_scatter_entry(
                req_id="req-abc",
                layer_id=3,
                worker_id=2,
                num_sparse=2048,
                total_tokens=18879,
                chunk_size=256,
                chunk_ptrs=74,
                use_fast_path=True,
            )
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert "sparse_scatter_entry" in call_args[0]
            assert "sparse_mla_dsa_batched_direct_kv_transfer" in call_args[0]
            assert call_args[1] == "req-abc"
            assert call_args[2] == 3

    def test_log_sparse_scatter_entry_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LMCACHE_DIAG_KV_CHECKSUM", raising=False)
        with patch("lmcache.v1.kv_checksum_diag.log_kv_checksum_diag") as mock_log:
            log_sparse_scatter_entry(
                req_id="req-x",
                layer_id=0,
                worker_id=0,
                num_sparse=1,
                total_tokens=1,
                chunk_size=256,
                chunk_ptrs=1,
                use_fast_path=False,
            )
            mock_log.assert_not_called()


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
class TestSparseMlaDsaStoreRetrieveRoundtrip:
    """GPU compute KV (src) vs sparse_mla_dsa scatter (dst) checksum on NPU."""

    @pytest.mark.parametrize("fmt", ["mla", "dsa"])
    @pytest.mark.parametrize("num_selected", [256, 2048])
    def test_sparse_direct_matches_src_checksum(
        self, fmt: str, num_selected: int
    ) -> None:
        num_tokens = 1024 if num_selected <= 1024 else 4096
        chunk_size = 256
        num_layers = 2
        num_blocks = 2048
        block_size = 16
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        common = dict(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )
        src = (
            generate_mla_kv_cache(**common)
            if fmt == "mla"
            else generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
        )
        harness = build_load_benchmark_harness(
            fmt=fmt,
            src_kv_cache=src,
            num_tokens=num_tokens,
            chunk_size=chunk_size,
            num_layers=num_layers,
            num_selected=num_selected,
            seed=42,
        )
        try:
            check_kw = _check_kwargs(
                fmt, num_kv_heads, kv_lora_rank, qk_rope_head_dim, dsa_head_dim
            )
            verify_load_benchmark_harness(
                harness,
                check_paged_kv_cache_equal,
                **check_kw,
            )
            self._assert_checksum_roundtrip(harness, check_kw["kv_format"])
        finally:
            harness.close()

    def test_direct_and_fast_paths_agree_on_checksum(self) -> None:
        num_tokens = 512
        chunk_size = 256
        num_layers = 2
        num_selected = 512
        common = dict(
            num_blocks=512,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=128,
            block_size=16,
            dtype=torch.bfloat16,
        )
        src = generate_mla_kv_cache(**common)
        harness = build_load_benchmark_harness(
            fmt="mla",
            src_kv_cache=src,
            num_tokens=num_tokens,
            chunk_size=chunk_size,
            num_layers=num_layers,
            num_selected=num_selected,
            seed=7,
        )
        try:
            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            token_indices = sample_token_indices(num_tokens)
            layer_ids = sample_layer_ids(num_layers)
            slots = harness.slot_mapping_packed.detach().cpu().tolist()
            direct_samples = collect_kv_checksum_samples(
                harness.dst_direct,
                slots,
                token_indices,
                layer_ids,
                kv_format=KV_FORMAT_MLA,
            )
            fast_samples = collect_kv_checksum_samples(
                harness.dst_direct_fast,
                slots,
                token_indices,
                layer_ids,
                kv_format=KV_FORMAT_MLA,
            )
            for left, right in zip(direct_samples, fast_samples, strict=True):
                assert left.k_fp == right.k_fp
                assert left.v_fp == right.v_fp
        finally:
            harness.close()

    @staticmethod
    def _assert_checksum_roundtrip(harness, kv_format: int) -> None:
        """Experiment B style: src (compute) vs dst_direct (scatter) fingerprints."""
        token_indices = sample_token_indices(harness.num_tokens)
        layer_ids = sample_layer_ids(harness.num_layers)
        slots = harness.slot_mapping_packed.detach().cpu().tolist()

        for layer_id in layer_ids:
            for token_idx in token_indices:
                slot = int(slots[token_idx])
                src_fps = slot_kv_fingerprints(
                    harness.src_kv_cache[layer_id], slot, kv_format=kv_format
                )
                dst_fps = slot_kv_fingerprints(
                    harness.dst_direct[layer_id], slot, kv_format=kv_format
                )
                assert src_fps["k"] == dst_fps["k"], (
                    f"L{layer_id}T{token_idx} K mismatch after scatter"
                )
                assert src_fps["v"] == dst_fps["v"], (
                    f"L{layer_id}T{token_idx} V mismatch after scatter"
                )
                if kv_format == KV_FORMAT_DSA:
                    assert src_fps.get("dsa") == dst_fps.get("dsa")


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
def test_chunk_partition_matches_long_prompt() -> None:
    """Sanity: 18879-token partition matches production chunk layout."""
    part = compute_chunk_partition(18879, 256)
    assert sum(part.chunk_sizes) == 18879
    assert part.num_chunks == 74
