# SPDX-License-Identifier: Apache-2.0
"""Correctness + small-scale timing for direct vs staging load kernels."""
from __future__ import annotations

# Standard
import sys
from pathlib import Path

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
    generate_dsa_index_kv_cache,
    generate_dsa_kv_cache,
    generate_mla_kv_cache,
)
from load_benchmark_utils import (  # noqa: E402
    bandwidth_gb_per_s,
    benchmark_load_harness,
    benchmark_store_harness,
    build_load_benchmark_harness,
    compute_direct_host_bytes,
    kv_format_from_name,
    verify_load_benchmark_harness,
    verify_store_benchmark_harness,
)


def _check_kwargs(fmt: str, num_kv_heads: int, kv_lora_rank: int, qk_rope_head_dim: int, dsa_head_dim: int):
    return dict(
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=kv_format_from_name(fmt),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
    )


def _run_case(
    *,
    fmt: str,
    num_tokens: int,
    chunk_size: int,
    num_layers: int,
    num_selected: int,
    verify_timing: bool,
) -> None:
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
    if fmt in ("mla", "mla_latent"):
        src = generate_mla_kv_cache(**common)
    elif fmt == "dsa":
        src = generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
    else:
        src = generate_dsa_index_kv_cache(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            dsa_head_dim=dsa_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )

    harness = build_load_benchmark_harness(
        fmt=fmt,
        src_kv_cache=src,
        num_tokens=num_tokens,
        chunk_size=chunk_size,
        num_layers=num_layers,
        num_selected=num_selected,
        seed=0,
    )
    try:
        verify_load_benchmark_harness(
            harness,
            check_paged_kv_cache_equal,
            **_check_kwargs(fmt, num_kv_heads, kv_lora_rank, qk_rope_head_dim, dsa_head_dim),
        )
        if verify_timing:
            case = benchmark_load_harness(harness, warmup=1, iters=3)
            assert case.staging_median_ms > 0
            assert case.direct_median_ms > 0
    finally:
        harness.close()


def _run_store_case(
    *,
    fmt: str,
    num_tokens: int,
    chunk_size: int,
    num_layers: int,
    verify_timing: bool,
) -> None:
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
    if fmt in ("mla", "mla_latent"):
        src = generate_mla_kv_cache(**common)
    elif fmt == "dsa":
        src = generate_dsa_kv_cache(dsa_head_dim=dsa_head_dim, **common)
    else:
        src = generate_dsa_index_kv_cache(
            num_blocks=num_blocks,
            device="npu",
            num_layers=num_layers,
            dsa_head_dim=dsa_head_dim,
            block_size=block_size,
            dtype=torch.bfloat16,
        )

    harness = build_load_benchmark_harness(
        fmt=fmt,
        src_kv_cache=src,
        num_tokens=num_tokens,
        chunk_size=chunk_size,
        num_layers=num_layers,
        num_selected=num_tokens,
        seed=0,
    )
    try:
        verify_store_benchmark_harness(harness)
        if verify_timing:
            case = benchmark_store_harness(harness, warmup=1, iters=3)
            assert case.staging_median_ms > 0
            assert case.direct_median_ms > 0
    finally:
        harness.close()


@pytest.mark.parametrize("fmt", ["mla", "dsa", "mla_latent", "dsa_index"])
@pytest.mark.parametrize("num_tokens", [512, 1024])
@pytest.mark.parametrize("num_layers", [1, 4])
def test_direct_vs_staging_dense(fmt: str, num_tokens: int, num_layers: int) -> None:
    _run_case(
        fmt=fmt,
        num_tokens=num_tokens,
        chunk_size=256,
        num_layers=num_layers,
        num_selected=num_tokens,
        verify_timing=False,
    )


@pytest.mark.parametrize("num_selected", [256, 512])
def test_direct_vs_staging_sparse(num_selected: int) -> None:
    _run_case(
        fmt="mla",
        num_tokens=1024,
        chunk_size=256,
        num_layers=1,
        num_selected=num_selected,
        verify_timing=False,
    )


@pytest.mark.parametrize("fmt", ["mla_latent", "dsa_index"])
def test_direct_vs_staging_store_dense_split_formats(fmt: str) -> None:
    _run_store_case(
        fmt=fmt,
        num_tokens=512,
        chunk_size=256,
        num_layers=1,
        verify_timing=False,
    )


def test_direct_vs_staging_timing_smoke() -> None:
    _run_case(
        fmt="mla",
        num_tokens=1024,
        chunk_size=256,
        num_layers=4,
        num_selected=1024,
        verify_timing=True,
    )


def test_compute_direct_host_bytes_mla_2048() -> None:
    nbytes = compute_direct_host_bytes(
        num_selected=2048,
        num_layers=2,
        plane_elems=5120,
        element_size=2,
    )
    assert nbytes == 41_943_040
    assert 22.5 <= bandwidth_gb_per_s(nbytes, 1.81) <= 24.0


def test_bandwidth_gb_per_s_zero_guard() -> None:
    assert bandwidth_gb_per_s(1024, 0.0) == 0.0
    assert bandwidth_gb_per_s(0, 1.0) == 0.0
