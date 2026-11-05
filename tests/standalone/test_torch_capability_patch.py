# SPDX-License-Identifier: Apache-2.0
"""Exercise compiler probes after LMCache's CUDA-to-NPU import remapping."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch._dynamo.device_interface import CudaInterface
from torch.utils import _triton


@pytest.fixture
def remapped_torch(monkeypatch):
    # transfer_to_npu binds CUDA names to NPU callables; changing the NPU
    # attribute later does not replace the callable already bound under CUDA.
    unsupported = lambda *args, **kwargs: None
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        get_device_capability=unsupported), raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unsupported)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(CudaInterface, "is_available",
                        staticmethod(lambda: torch.cuda.is_available()))
    monkeypatch.setattr(CudaInterface.Worker, "get_device_properties",
                        staticmethod(lambda: SimpleNamespace(major=None)))
    contrib = ModuleType("torch_npu.contrib")
    contrib.transfer_to_npu = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch_npu.contrib", contrib)
    monkeypatch.setattr(_triton, "has_triton_package", lambda: True)
    _triton._device_supports_tma.cache_clear()
    _triton.has_triton.cache_clear()
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/__init__.py"
    module = ast.Module(body=[node for node in ast.parse(path.read_text()).body
                             if isinstance(node, ast.FunctionDef)
                             and node.name == "_patch_torch_capability"],
                        type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    yield namespace["_patch_torch_capability"]
    _triton._device_supports_tma.cache_clear()
    _triton.has_triton.cache_clear()


def test_tma_probe_survives_connector_import_before_compilation(remapped_torch):
    remapped_torch()
    assert torch.cuda.get_device_capability() == (0, 0)
    assert torch.npu.get_device_capability() == (0, 0)
    assert not _triton._device_supports_tma()
    # LMCache's existing device dispatch remains NPU-enabled.
    assert torch.cuda.is_available()


def test_triton_probe_skips_cuda_but_preserves_other_backends(
    remapped_torch, monkeypatch,
):
    from torch._dynamo import device_interface

    visited = []
    def get_interface(device):
        visited.append(device)
        return CudaInterface if device == "cuda" else SimpleNamespace(
            is_available=lambda: device == "xpu")
    monkeypatch.setattr(device_interface, "get_interface_for_device", get_interface)
    remapped_torch()
    assert _triton.has_triton()
    assert visited == ["cuda", "xpu"]


def test_patch_is_repeatable_and_does_not_replace_triton_dispatch(remapped_torch):
    triton_probe = _triton.has_triton
    remapped_torch()
    remapped_torch()
    assert not CudaInterface.is_available()
    assert _triton.has_triton is triton_probe
