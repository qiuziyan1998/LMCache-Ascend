# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_SOURCE = (
    _REPO_ROOT
    / "third_party"
    / "kvcache-ops"
    / "kernels"
    / "single_layer"
    / "single_layer_sparse_k_transfer.cpp"
)


def _kernel_declaration_macro() -> str:
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"#define SPARSE_K_TRANSFER_KERNEL_DECLARE\(TYPE, SLOT_TYPE\)"
        r"(?P<body>.*?)\n\nSPARSE_K_TRANSFER_KERNEL_DECLARE\(half, int32_t\);",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "sparse K kernel declaration macro is missing"
    return match.group("body")


def _parenthesized_arguments(source: str, marker: str) -> list[str]:
    marker_end = source.index(marker) + len(marker)
    start = source.index("(", marker_end)
    depth = 0
    for end in range(start, len(source)):
        if source[end] == "(":
            depth += 1
        elif source[end] == ")":
            depth -= 1
            if depth == 0:
                return [
                    argument.strip()
                    for argument in source[start + 1 : end].split(",")
                ]
    raise AssertionError(f"unclosed argument list after {marker}")


def test_generated_launch_pointer_arguments_use_gm_addr() -> None:
    """Protect the spelling consumed by CANN's launch-wrapper generator."""

    declaration = _kernel_declaration_macro()
    pointer_names = (
        "chunkPtrs",
        "vllmKPtr",
        "slotMappingPtr",
        "selectedTokenIdxPtr",
        "selectedTokenCountsPtr",
    )
    for name in pointer_names:
        assert re.search(rf"\bGM_ADDR\s+{name}\b", declaration)
        assert not re.search(
            rf"__gm__\s+uint8_t\s*\*\s*{name}\b", declaration
        )


def test_device_entry_and_generated_launch_argument_order_match() -> None:
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    declaration = _kernel_declaration_macro()
    device_arguments = _parenthesized_arguments(
        declaration, "sparse_k_transfer_##TYPE##_##SLOT_TYPE"
    )
    device_names = [
        re.search(r"([A-Za-z_]\w*)\s*\\?\s*$", argument).group(1)
        for argument in device_arguments
    ]

    launch_match = re.search(
        r"#define SPARSE_K_TRANSFER_LAUNCH\(TYPE, SLOT_TYPE\)"
        r"(?P<body>.*?)\n\n"
        r"template <typename T, typename SlotT>",
        source,
        flags=re.DOTALL,
    )
    assert launch_match is not None, "sparse K host launch macro is missing"
    launch_arguments = _parenthesized_arguments(
        launch_match.group("body"), ">>"
    )
    launch_names = [
        re.sub(r"\\|\s", "", argument) for argument in launch_arguments
    ]

    expected_device = [
        "chunkPtrs",
        "vllmKPtr",
        "slotMappingPtr",
        "selectedTokenIdxPtr",
        "selectedTokenCountsPtr",
        "vllmKBufferSize",
        "kHiddenDims",
        "numTokens",
        "numChunks",
        "chunkSize",
        "chunkShift",
        "chunkMask",
        "totalTokens",
        "tileTokens",
        "pipeline",
        "workAssignment",
        "rowWidth",
        "requestCount",
        "selectedCountStride",
    ]
    expected_launch = [
        "chunkPtrsPtr",
        "vllmKPtr",
        "slotMappingPtr",
        "selectedTokenIdxPtr",
        "selectedTokenCountsPtr",
        *expected_device[5:],
    ]
    assert device_names == expected_device
    assert launch_names == expected_launch


def test_all_supported_type_pairs_have_device_and_host_instantiations() -> None:
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    for scalar_type in ("half", "int8_t", "bfloat16_t"):
        for slot_type in ("int32_t", "int64_t"):
            device = (
                f"SPARSE_K_TRANSFER_KERNEL_DECLARE"
                f"({scalar_type}, {slot_type});"
            )
            host = (
                f"SPARSE_K_TRANSFER_HOST_DECLARE"
                f"({scalar_type}, {slot_type});"
            )
            assert source.count(device) == 1
            assert source.count(host) == 1
