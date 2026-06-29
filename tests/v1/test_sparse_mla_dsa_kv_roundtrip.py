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
    log_sparse_scatter_entry,
    reset_kv_checksum_diag_state,
    sample_layer_ids,
    sample_token_indices,
)


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _sample_packed_indices(num_packed: int, max_samples: int = 32) -> list[int]:
    if num_packed <= max_samples:
        return list(range(num_packed))
    step = max(1, num_packed // max_samples)
    return list(range(0, num_packed, step))[:max_samples]


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
    """sparse_mla_dsa_batched_direct_kv_transfer store/retrieve roundtrip.

    Comparison mirrors kv_cache_fixtures.check_paged_kv_cache_equal but is
    inlined and done on CPU so that:
      1. MLA/DSA layout is reshaped correctly to [-1, num_heads, kv_lora_rank]
         (NOT [-1, kv_lora_rank] which would read a single head's row).
      2. pytest never reprs an NPU tensor (segfaults on Ascend) -- the only
         frame with NPU tensors is the test method body, whose args are
         (self, fmt, num_selected): plain str/int.
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

            slots_npu = harness.slot_mapping_packed
            num_packed = int(slots_npu.numel())
            slots = slots_npu.detach().cpu().tolist()
            packed_idx_to_check = _sample_packed_indices(num_packed)
            layer_ids = sample_layer_ids(harness.num_layers)

            mismatches: list[str] = []
            for layer_id in layer_ids:
                src_layer = harness.src_kv_cache[layer_id]
                dst_dir = harness.dst_direct[layer_id]
                dst_fast = harness.dst_direct_fast[layer_id]

                src_k = src_layer[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_dir_k = dst_dir[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_fast_k = dst_fast[0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                src_v = src_layer[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_dir_v = dst_dir[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_fast_v = dst_fast[1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()

                src_dsa = dst_dir_dsa = dst_fast_dsa = None
                if kv_format == KV_FORMAT_DSA:
                    src_dsa = src_layer[2].reshape(-1, dsa_head_dim).detach().cpu()
                    dst_dir_dsa = dst_dir[2].reshape(-1, dsa_head_dim).detach().cpu()
                    dst_fast_dsa = dst_fast[2].reshape(-1, dsa_head_dim).detach().cpu()

                for pi in packed_idx_to_check:
                    slot = int(slots[pi])
                    if not torch.equal(src_k[slot], dst_dir_k[slot]):
                        mismatches.append(
                            f"direct L{layer_id}P{pi}slot{slot} K mismatch"
                        )
                    if not torch.equal(src_v[slot], dst_dir_v[slot]):
                        mismatches.append(
                            f"direct L{layer_id}P{pi}slot{slot} V mismatch"
                        )
                    if not torch.equal(src_k[slot], dst_fast_k[slot]):
                        mismatches.append(
                            f"fast L{layer_id}P{pi}slot{slot} K mismatch"
                        )
                    if not torch.equal(src_v[slot], dst_fast_v[slot]):
                        mismatches.append(
                            f"fast L{layer_id}P{pi}slot{slot} V mismatch"
                        )
                    if src_dsa is not None:
                        if not torch.equal(src_dsa[slot], dst_dir_dsa[slot]):
                            mismatches.append(
                                f"direct L{layer_id}P{pi}slot{slot} DSA mismatch"
                            )
                        if not torch.equal(src_dsa[slot], dst_fast_dsa[slot]):
                            mismatches.append(
                                f"fast L{layer_id}P{pi}slot{slot} DSA mismatch"
                            )
            assert not mismatches, (
                f"sparse scatter roundtrip mismatch fmt={fmt} "
                f"num_selected={num_selected} (showing first 8 of "
                f"{len(mismatches)}): {mismatches[:8]}"
            )
        except AssertionError:
            raise
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

    def test_direct_and_fast_paths_agree(self) -> None:
        num_tokens = 512
        num_layers = 2
        num_selected = 512
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        common = dict(
            num_blocks=256,
            device="npu",
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
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

            slots = harness.slot_mapping_packed.detach().cpu().tolist()
            packed_idx_to_check = _sample_packed_indices(len(slots))
            layer_ids = sample_layer_ids(num_layers)
            mismatches: list[str] = []
            for layer_id in layer_ids:
                dst_dir_k = harness.dst_direct[layer_id][0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_fast_k = harness.dst_direct_fast[layer_id][0].reshape(
                    -1, num_kv_heads, kv_lora_rank
                ).detach().cpu()
                dst_dir_v = harness.dst_direct[layer_id][1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                dst_fast_v = harness.dst_direct_fast[layer_id][1].reshape(
                    -1, num_kv_heads, qk_rope_head_dim
                ).detach().cpu()
                for pi in packed_idx_to_check:
                    slot = int(slots[pi])
                    if not torch.equal(dst_dir_k[slot], dst_fast_k[slot]):
                        mismatches.append(f"L{layer_id}P{pi}slot{slot} K direct/fast differ")
                    if not torch.equal(dst_dir_v[slot], dst_fast_v[slot]):
                        mismatches.append(f"L{layer_id}P{pi}slot{slot} V direct/fast differ")
            assert not mismatches, (
                f"direct vs fast disagree (first 8 of {len(mismatches)}): "
                f"{mismatches[:8]}"
            )
        except AssertionError:
            raise
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
class TestDsaPopulateVsScatterIsolation:
    """Isolate whether DSA is lost in src->cpu (populate) or cpu->dst (scatter).

    Uses a single-chunk DSA layout so localTokenIdx == globalTokenIdx, making
    the CPU chunk DSA plane offset trivially auditable.

    CPU chunk (non-interleaved) layout for DSA, per chunk of N tokens:
        [ K plane: N*kH | V plane: N*vH | DSA plane: N*dsaH ]
    DSA for local token t lives at:
        offset = N*(kH+vH) + t*dsaH, length dsaH
    """

    def test_dsa_populate_and_scatter_isolation(self) -> None:
        num_tokens = 256
        chunk_size = 256  # single chunk
        num_layers = 1
        num_blocks = 512
        block_size = 16
        num_kv_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 128
        dsa_head_dim = 128
        kH = num_kv_heads * kv_lora_rank  # 4096
        vH = num_kv_heads * qk_rope_head_dim  # 1024
        dsaH = dsa_head_dim  # 128
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
            src = generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
            harness = build_load_benchmark_harness(
                fmt="dsa",
                src_kv_cache=src,
                num_tokens=num_tokens,
                chunk_size=chunk_size,
                num_layers=num_layers,
                num_selected=num_tokens,  # dense select: all tokens
                seed=42,
            )
            # populate already ran inside build_load_benchmark_harness.
            slot_mapping_full = harness.slot_mapping_full.detach().cpu().tolist()
            chunk0 = harness.stacked_by_layer[0][0].detach().cpu()
            num_lmc = chunk_size  # single full chunk

            # ---- Stage 1: src DSA vs CPU chunk DSA plane (populate check) ----
            src_dsa = harness.src_kv_cache[0][2].reshape(-1, dsaH).detach().cpu()
            dsa_plane_start = num_lmc * (kH + vH)
            populate_mismatches: list[str] = []
            for t in range(num_tokens):
                cpu_dsa_t = chunk0[dsa_plane_start + t * dsaH : dsa_plane_start + (t + 1) * dsaH]
                slot = int(slot_mapping_full[t])
                if not torch.equal(src_dsa[slot], cpu_dsa_t):
                    populate_mismatches.append(f"token{t}slot{slot}")

            # ---- Stage 2: scatter cpu->dst_direct DSA ----
            run_all_direct_layers(harness)
            torch.npu.synchronize()
            slots_packed = harness.slot_mapping_packed.detach().cpu().tolist()
            selected = harness.selected_token_idx.detach().cpu().tolist()
            dst_dsa = harness.dst_direct[0][2].reshape(-1, dsaH).detach().cpu()
            scatter_mismatches: list[str] = []
            for pi, gtok in enumerate(selected):
                local_t = gtok % chunk_size
                cpu_dsa_t = chunk0[dsa_plane_start + local_t * dsaH : dsa_plane_start + (local_t + 1) * dsaH]
                slot = int(slots_packed[pi])
                if not torch.equal(dst_dsa[slot], cpu_dsa_t):
                    scatter_mismatches.append(f"packed{pi}gtok{gtok}slot{slot}")

            populate_ok = not populate_mismatches
            scatter_ok = not scatter_mismatches
            summary = (
                f"populate(src->cpu)={'OK' if populate_ok else 'BROKEN'} "
                f"({len(populate_mismatches)}/{num_tokens} mismatched, "
                f"first 5: {populate_mismatches[:5]}); "
                f"scatter(cpu->dst)={'OK' if scatter_ok else 'BROKEN'} "
                f"({len(scatter_mismatches)}/{len(selected)} mismatched, "
                f"first 5: {scatter_mismatches[:5]})"
            )
            # Don't hard-fail: report which stage is broken. But assert so the
            # test surfaces in CI with the summary.
            if not (populate_ok and scatter_ok):
                raise AssertionError(f"DSA isolation: {summary}")
            # If both OK but the full roundtrip test still fails, the issue is
            # elsewhere (layout interpretation in the roundtrip checker).
        except AssertionError:
            raise
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
