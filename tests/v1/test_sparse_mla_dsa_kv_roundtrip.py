# SPDX-License-Identifier: Apache-2.0
"""Store/retrieve roundtrip via sparse_mla_dsa_batched_direct_kv_transfer."""
from __future__ import annotations

# Standard
import sys
import traceback
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


def _checksum_token_indices(selected_idx_npu, num_tokens: int) -> list[int]:
    """Token indices that were actually scattered AND in the diagnostic sample set."""
    selected = set(selected_idx_npu.detach().cpu().tolist())
    sampled = sample_token_indices(num_tokens)
    indices = [t for t in sampled if t in selected]
    if indices:
        return indices
    return sorted(selected)[: min(8, len(selected))]


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
    """GPU compute KV (src) vs sparse_mla_dsa scatter (dst) checksum on NPU.

    All NPU-tensor-bearing logic is inlined and wrapped in try/except so pytest
    never has to repr an NPU tensor (which segfaults on Ascend) when reporting
    failures. Assertions are converted to plain-text AssertionError.
    """

    @pytest.mark.parametrize("fmt", ["mla", "dsa"])
    @pytest.mark.parametrize("num_selected", [256, 2048])
    def test_sparse_direct_matches_src_checksum(
        self, fmt: str, num_selected: int
    ) -> None:
        num_tokens = 1024 if num_selected <= 1024 else 4096
        num_layers = 2
        num_blocks = 512
        block_size = 16
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        kv_format = KV_FORMAT_MLA if fmt == "mla" else KV_FORMAT_DSA
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
        harness = None
        try:
            src = (
                generate_mla_kv_cache(**common)
                if fmt == "mla"
                else generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
            )
            harness = build_load_benchmark_harness(
                fmt=fmt,
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=num_layers,
                num_selected=num_selected,
                seed=42,
            )
            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            token_indices = _checksum_token_indices(
                harness.selected_token_idx, harness.num_tokens
            )
            layer_ids = sample_layer_ids(harness.num_layers)
            slot_mapping_full = harness.slot_mapping_full.detach().cpu()
            mismatches: list[str] = []
            for layer_id in layer_ids:
                for token_idx in token_indices:
                    slot = int(slot_mapping_full[token_idx].item())
                    src_fps = slot_kv_fingerprints(
                        harness.src_kv_cache[layer_id], slot, kv_format=kv_format
                    )
                    dst_fps = slot_kv_fingerprints(
                        harness.dst_direct[layer_id], slot, kv_format=kv_format
                    )
                    if src_fps["k"] != dst_fps["k"]:
                        mismatches.append(
                            f"L{layer_id}T{token_idx}slot{slot} K "
                            f"{src_fps['k']}!={dst_fps['k']}"
                        )
                    if src_fps["v"] != dst_fps["v"]:
                        mismatches.append(
                            f"L{layer_id}T{token_idx}slot{slot} V "
                            f"{src_fps['v']}!={dst_fps['v']}"
                        )
                    if kv_format == KV_FORMAT_DSA and src_fps.get("dsa") != dst_fps.get(
                        "dsa"
                    ):
                        mismatches.append(
                            f"L{layer_id}T{token_idx}slot{slot} DSA mismatch"
                        )
            assert not mismatches, (
                f"sparse scatter vs src checksum mismatch "
                f"fmt={fmt} num_selected={num_selected}: {mismatches[:5]}"
            )
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed fmt={fmt} num_selected={num_selected}: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass

    def test_direct_and_fast_paths_agree_on_checksum(self) -> None:
        num_tokens = 512
        num_layers = 2
        num_selected = 512
        common = dict(
            num_blocks=256,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=128,
            block_size=16,
            dtype=torch.bfloat16,
        )
        harness = None
        try:
            src = generate_mla_kv_cache(**common)
            harness = build_load_benchmark_harness(
                fmt="mla",
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=256,
                num_layers=num_layers,
                num_selected=num_selected,
                seed=7,
            )
            run_all_direct_layers(harness)
            run_all_direct_fast_layers(harness)
            torch.npu.synchronize()

            token_indices = sample_token_indices(num_tokens)
            layer_ids = sample_layer_ids(num_layers)
            slot_mapping = harness.slot_mapping_full.detach().cpu().tolist()
            direct_samples = collect_kv_checksum_samples(
                harness.dst_direct,
                slot_mapping,
                token_indices,
                layer_ids,
                kv_format=KV_FORMAT_MLA,
            )
            fast_samples = collect_kv_checksum_samples(
                harness.dst_direct_fast,
                slot_mapping,
                token_indices,
                layer_ids,
                kv_format=KV_FORMAT_MLA,
            )
            mismatches: list[str] = []
            for left, right in zip(direct_samples, fast_samples, strict=True):
                if left.k_fp != right.k_fp:
                    mismatches.append(
                        f"L{left.layer_id}T{left.token_idx} K "
                        f"{left.k_fp}!={right.k_fp}"
                    )
                if left.v_fp != right.v_fp:
                    mismatches.append(
                        f"L{left.layer_id}T{left.token_idx} V "
                        f"{left.v_fp}!={right.v_fp}"
                    )
            assert not mismatches, f"direct vs fast disagree: {mismatches[:5]}"
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"test failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ) from None
        finally:
            if harness is not None:
                try:
                    harness.close()
                except Exception:  # noqa: BLE001
                    pass


@pytest.mark.skipif(not _npu_available(), reason="NPU required")
def test_chunk_partition_matches_long_prompt() -> None:
    """Sanity: 18879-token partition matches production chunk layout."""
    part = compute_chunk_partition(18879, 256)
    assert sum(part.chunk_sizes) == 18879
    assert part.num_chunks == 74
