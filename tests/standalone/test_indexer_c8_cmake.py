# SPDX-License-Identifier: Apache-2.0
"""Configure the real kernel build graph against a compiler-free CANN shim."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("mindspore", [False, True])
@pytest.mark.parametrize("explicit_arch", [False, True])
def test_kernel_toolkit_is_initialized_once(tmp_path, mindspore, explicit_arch):
    cmake = os.environ.get("CMAKE_TEST_EXECUTABLE") or shutil.which("cmake")
    ninja = os.environ.get("NINJA_TEST_EXECUTABLE") or shutil.which("ninja")
    vendor = Path(
        os.environ.get("KVCACHE_OPS_TEST_SOURCE", ROOT / "third_party/kvcache-ops")
    )
    if not cmake or not ninja or not (vendor / "CMakeLists.txt").exists():
        pytest.skip(
            "CMake, Ninja and the initialized kvcache-ops submodule are required"
        )

    def source(relative):
        revision = os.environ.get("C8_TEST_SOURCE_REVISION")
        if revision:
            return subprocess.check_output(
                ["git", "show", f"{revision}:{relative}"], cwd=ROOT
            ).decode()
        return (ROOT / relative).read_text(encoding="utf-8")

    def write(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # Only dependency/toolchain operations are stubbed. Root routing, toolkit
    # initialization, kernel source discovery and target names execute in CMake.
    shim = """
set(CMAKE_SYSTEM_PROCESSOR x86_64)
function(find_package)
endfunction()
function(target_link_options)
endfunction()
function(target_link_directories)
endfunction()
function(target_link_libraries)
  set_property(GLOBAL APPEND PROPERTY test_links "${ARGV}")
endfunction()
function(target_include_directories)
endfunction()
function(install)
  set_property(GLOBAL APPEND PROPERTY test_install "${ARGV}")
endfunction()
"""
    root_text = source("CMakeLists.txt").replace(
        "project(c_ops)", "project(c_ops LANGUAGES NONE)\n" + shim
    )
    root_text += """
get_target_property(sources cache_kernels test_sources)
get_property(links GLOBAL PROPERTY test_links)
get_property(installed GLOBAL PROPERTY test_install)
file(WRITE "${CMAKE_BINARY_DIR}/graph.txt" "${sources}\n${links}\n${installed}\n")
"""
    write("CMakeLists.txt", root_text)
    write("csrc/CMakeLists.txt", "add_library(c_ops INTERFACE)\n")
    write("csrc/mindspore/CMakeLists.txt", "add_library(c_ops INTERFACE)\n")
    write("csrc/graph/CMakeLists.txt", source("csrc/graph/CMakeLists.txt"))
    write("csrc/graph/sparse_graph_kernel.cpp", "// configuration test\n")
    write("csrc/indexer_c8/kernel.cpp", "// configuration test\n")
    for name in ("CMakeLists.txt", "npu_lib.cmake"):
        write(
            f"third_party/kvcache-ops/{name}",
            (vendor / name).read_text(encoding="utf-8"),
        )
    write(
        "third_party/kvcache-ops/ascendc_with_def.cmake",
        """
# CANN 8.5.1 creates this target unconditionally. Re-including must fail.
add_library(host_intf_pub INTERFACE)
function(ascendc_library target kind)
  add_library(${target} INTERFACE)
  set_property(TARGET ${target} PROPERTY test_sources "${ARGN}")
endfunction()
function(ascendc_library_with_def target kind)
  ascendc_library(${target} ${kind} ${ARGN})
endfunction()
function(ascendc_compile_definitions)
endfunction()
""",
    )
    buckets = ("", "cachegen/", "fused_rope/", "single_layer/", "multi_layer/")
    for bucket in buckets:
        write(
            f"third_party/kvcache-ops/kernels/{bucket}existing.cpp",
            "// configuration test\n",
        )
    env = os.environ.copy()
    env.pop("USE_MINDSPORE", None)
    if mindspore:
        env["USE_MINDSPORE"] = "1"
    command = [
        cmake,
        "-S",
        str(tmp_path),
        "-B",
        str(tmp_path / "build"),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
    ]
    if explicit_arch:
        command.append("-DASCEND_AICORE_ARCH=220")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "build/graph.txt").read_text().splitlines()
    sources = lines[0].split(";")
    assert len(sources) == (5 if mindspore else 7)
    for bucket in buckets:
        assert any(path.endswith(f"kernels/{bucket}existing.cpp") for path in sources)
    assert (
        any(path.endswith("csrc/indexer_c8/kernel.cpp") for path in sources)
        is not mindspore
    )
    assert any(path.endswith("csrc/graph/sparse_graph_kernel.cpp") for path in sources) is not mindspore
    assert "cache_kernels" in lines[1].split(";")
    assert "cache_kernels" in lines[2].split(";")
    assert "indexer_c8_kernels" not in "\n".join(lines)
