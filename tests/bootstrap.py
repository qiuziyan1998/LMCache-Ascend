# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib
import importlib.util
import os
import subprocess
import sys

# First Party
import lmcache_ascend

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LMCACHEGITREPO = "https://github.com/LMCache/LMCache.git"
VERSION_TAG = lmcache_ascend.LMCACHE_UPSTREAM_TAG
TEST_ALIAS = "lmcache_tests"
LMCACHEPATH = os.environ.get("LMCACHEPATH", "")


def _run_git(cmd_list, cwd=None):
    try:
        subprocess.check_call(["git"] + cmd_list, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {' '.join(cmd_list)}")
        raise e


def get_current_git_tag(path):
    """Returns the current tag name if HEAD is exactly on a tag, else None."""
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _has_upstream_tests(root: str) -> bool:
    tests_utils = os.path.join(root, "tests", "v1", "utils.py")
    tests_conftest = os.path.join(root, "tests", "conftest.py")
    return os.path.exists(tests_utils) or os.path.exists(
        os.path.join(root, "tests", "__init__.py")
    ) or os.path.exists(tests_conftest)


def _resolve_lmcache_root() -> str | None:
    """Find an on-disk LMCache repo root that includes the upstream tests tree."""
    candidates: list[str] = []

    if LMCACHEPATH:
        candidates.append(LMCACHEPATH)

    try:
        # Third Party
        import lmcache

        pkg_dir = os.path.dirname(os.path.abspath(lmcache.__file__))
        candidates.append(os.path.dirname(pkg_dir))
    except ImportError:
        pass

    ascend_root = os.path.dirname(os.path.dirname(os.path.abspath(lmcache_ascend.__file__)))
    parent = os.path.dirname(ascend_root)
    candidates.extend(
        [
            os.path.join(ascend_root, "LMCache-NPU"),
            os.path.join(ascend_root, "LMCache"),
            os.path.join(parent, "LMCache-NPU"),
            os.path.join(parent, "LMCache"),
            "/workspace/LMCache-NPU",
            "/workspace/LMCache",
            "/workspace/qzy/LMCache-NPU",
            "/workspace/qzy/LMCache",
        ]
    )

    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if _has_upstream_tests(norm):
            return norm
    return None


def setup_lmcache_dependency():
    """Use an existing LMCache install/repo; clone only as a last resort."""
    global LMCACHEPATH

    resolved = _resolve_lmcache_root()
    if resolved:
        LMCACHEPATH = resolved
        current_tag = get_current_git_tag(LMCACHEPATH)
        if current_tag and current_tag != VERSION_TAG:
            print(
                f"ℹ️ Using LMCache at {LMCACHEPATH} "
                f"(tag {current_tag}, bootstrap expects {VERSION_TAG}; skipping git sync)"
            )
        else:
            print(f"✅ Using LMCache at {LMCACHEPATH}")
        return

    # `lmcache` installed without the upstream tests tree (e.g. pip / patched env).
    try:
        import lmcache  # noqa: F401

        print(
            "✅ `lmcache` is importable; upstream tests tree not found "
            "(skipping git clone). Set LMCACHEPATH to the LMCache repo root "
            "if you need lmcache_tests fixtures."
        )
        LMCACHEPATH = ""
        return
    except ImportError:
        pass

    clone_target = LMCACHEPATH or "/workspace/LMCache"
    print(f"📦 LMCache missing. Cloning {VERSION_TAG} into {clone_target}...")
    _run_git(
        [
            "clone",
            "--branch",
            VERSION_TAG,
            "--depth",
            "1",
            LMCACHEGITREPO,
            clone_target,
        ]
    )
    LMCACHEPATH = clone_target


def register_alias():
    """Injects the upstream tests into sys.modules as 'lmcache_tests'."""
    if not LMCACHEPATH or not _has_upstream_tests(LMCACHEPATH):
        return False

    if LMCACHEPATH not in sys.path:
        sys.path.insert(0, LMCACHEPATH)

    if TEST_ALIAS in sys.modules:
        return True

    tests_init_path = os.path.join(LMCACHEPATH, "tests", "__init__.py")
    if os.path.exists(tests_init_path):
        spec = importlib.util.spec_from_file_location(TEST_ALIAS, tests_init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[TEST_ALIAS] = module
            spec.loader.exec_module(module)
            print(f"✅ Registered module alias '{TEST_ALIAS}'")
            return True
        return False

    # LMCache-NPU / dev trees may omit tests/__init__.py; alias the `tests` package.
    tests_pkg = importlib.import_module("tests")
    sys.modules[TEST_ALIAS] = tests_pkg
    for name, mod in list(sys.modules.items()):
        if name == "tests" or name.startswith("tests."):
            sys.modules[name.replace("tests", TEST_ALIAS, 1)] = mod
    print(f"✅ Registered module alias '{TEST_ALIAS}' from installed `tests` package")
    return True


def prepare_environment():
    """Main entry point to prepare the test environment."""
    setup_lmcache_dependency()
    return register_alias()
