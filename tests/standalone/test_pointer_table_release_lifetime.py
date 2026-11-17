# SPDX-License-Identifier: Apache-2.0
"""Guard the ``_layerwise_pointer_tables`` registry release path.

The registry maps either ``id(pointer_list)`` or
``(id(pointer_list), layer_id)`` to one NPU pointer table. Releasing one
request's entries used to iterate ``tuple(tables)``, which is O(total live and
stale entries in the process) and raises ``RuntimeError`` when another request
appends concurrently.

These tests load the production function from source with ``ast`` (no NPU
imports) and assert the release only touches the caller's own keys.
"""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace

CONNECTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
)


class _NoIterDict(dict):
    """Registry stand-in that fails if the implementation scans it."""

    def __iter__(self):
        raise AssertionError("release must not iterate the registry")

    def items(self):
        raise AssertionError("release must not iterate the registry")

    def keys(self):
        raise AssertionError("release must not iterate the registry")

    def values(self):
        raise AssertionError("release must not iterate the registry")


def _load_release():
    """Compile ``release_sparse_chunk_ptr_cache`` without importing torch."""
    source = CONNECTOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "release_sparse_chunk_ptr_cache"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, func], type_ignores=[])
    )
    namespace = {"torch": None, "Optional": object, "List": object}
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace["release_sparse_chunk_ptr_cache"]


def test_release_removes_both_key_shapes_and_preserves_other_requests():
    release = _load_release()
    target = [None] * 4
    identity = id(target)
    tables = {
        identity: {"table": "group"},
        (identity, 0): {"table": "layer0"},
        (identity, 1): {"table": "layer1"},
        (0xDEAD, 0): {"table": "other-request"},
        0xBEEF: {"table": "other-group"},
    }
    deferred = {identity: target}
    owner = SimpleNamespace(
        _layerwise_pointer_tables=tables,
        _deferred_sparse_pointer_caches=deferred,
    )

    release(owner, target)

    assert identity not in tables
    assert (identity, 0) not in tables
    assert (identity, 1) not in tables
    assert (0xDEAD, 0) in tables
    assert 0xBEEF in tables
    assert deferred == {}


def test_release_does_not_scan_the_registry():
    release = _load_release()
    target = [None] * 2
    identity = id(target)
    tables = _NoIterDict()
    for index in range(100_000):
        tables[(index, 0)] = None
    tables[identity] = None
    tables[(identity, 0)] = None
    owner = SimpleNamespace(
        _layerwise_pointer_tables=tables,
        _deferred_sparse_pointer_caches={},
    )

    release(owner, target)

    assert identity not in tables
    assert (identity, 0) not in tables
    assert len(tables) == 100_000


def test_release_is_idempotent_and_tolerates_missing_state():
    release = _load_release()

    release(SimpleNamespace(), None)
    release(SimpleNamespace(), [None])

    target = [None]
    tables = {id(target): None}
    owner = SimpleNamespace(_layerwise_pointer_tables=tables)
    release(owner, target)
    release(owner, target)
    assert tables == {}


def test_release_handles_empty_pointer_list():
    release = _load_release()
    target = []
    tables = {id(target): {"table": "group"}}
    owner = SimpleNamespace(
        _layerwise_pointer_tables=tables,
        _deferred_sparse_pointer_caches={id(target): target},
    )

    release(owner, target)

    assert tables == {}


def _calls_in_finally(source_path, class_name, function_name):
    """Return the call names reached from any ``finally`` of a method."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    func = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    names = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for stmt in node.finalbody:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                target = child.func
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    names.add(target.attr)
    return names


def test_store_layer_finally_releases_pointer_registry():
    """Store results own a temporary identity the retrieve path never sees."""
    engine_path = (
        Path(__file__).resolve().parents[2]
        / "lmcache_ascend/v1/cache_engine.py"
    )

    names = _calls_in_finally(engine_path, "AscendLMCacheEngine", "store_layer")

    assert "release_sparse_chunk_ptr_cache" in names or "release_ptrs" in names


def test_batched_from_gpu_finally_releases_fallback_table():
    """The deferred non-raw-DMA fallback table is private to one transfer."""
    names = _calls_in_finally(
        CONNECTOR_PATH, "VLLMPagedMemLayerwiseNPUConnector", "batched_from_gpu"
    )

    assert "release_sparse_chunk_ptr_cache" in names
