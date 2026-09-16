# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the real editable-install metadata hook."""

# Standard
from email.parser import Parser
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# Third Party
from packaging.version import Version
import pytest


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for prefix in ("SETUPTOOLS_SCM", "VCS_VERSIONING"):
        monkeypatch.delenv(f"{prefix}_PRETEND_VERSION", raising=False)
        monkeypatch.delenv(
            f"{prefix}_PRETEND_VERSION_FOR_LMCACHE_ASCEND", raising=False
        )
    root = Path(__file__).resolve().parents[2]
    for name in ("setup.py", "pyproject.toml"):
        shutil.copy2(root / name, tmp_path / name)
    (tmp_path / "README.md").write_text("Metadata regression test\n", encoding="utf-8")
    package = tmp_path / "lmcache_ascend"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Version Test")
    git(tmp_path, "config", "user.email", "version-test@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "tag.gpgsign", "false")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "no-hooks"))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def metadata_version(repo: Path) -> str:
    # Run the same backend hook that failed on the server. It must neither
    # import torch_npu nor invoke the native extension compiler at this stage.
    with tempfile.TemporaryDirectory() as metadata_dir:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from setuptools.build_meta import "
                "prepare_metadata_for_build_editable; "
                "prepare_metadata_for_build_editable(sys.argv[1])",
                metadata_dir,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        paths = list(Path(metadata_dir).glob("*.dist-info/METADATA"))
        assert len(paths) == 1
        metadata = Parser().parsestr(paths[0].read_text(encoding="utf-8"))
        assert metadata["Name"] == "lmcache-ascend"
        version = metadata["Version"]
        Version(version)
    assert (repo / "lmcache_ascend" / "_version.py").is_file()
    return version


@pytest.mark.parametrize("tag", ["v0.3.7", "0.3.7", "v0.3.7rc1", "0.3.7.post1"])
def test_release_version_is_preserved(repo: Path, tag: str) -> None:
    git(repo, "tag", tag)
    assert Version(metadata_version(repo)) == Version(tag)


def test_performance_tag_does_not_mask_release(repo: Path) -> None:
    git(repo, "tag", "v0.3.7")
    git(repo, "commit", "--allow-empty", "-m", "after release")
    before = metadata_version(repo)
    git(repo, "tag", "pd-tpot-96.8ms-baseline-20260910")
    assert metadata_version(repo) == before
    assert "+" not in before  # Preserve the existing no-local-version scheme.


@pytest.mark.parametrize(
    "tag", ["pd-tpot-96.8ms-baseline-20260910", "remote-fill-checkpoint-20260909"]
)
def test_non_release_tag_without_release_tag(repo: Path, tag: str) -> None:
    before = metadata_version(repo)
    git(repo, "tag", tag)
    assert metadata_version(repo) == before


def test_explicit_version_override(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git(repo, "tag", "pd-tpot-96.8ms-baseline-20260910")
    monkeypatch.setenv(
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND", "0.3.7+test"
    )
    assert metadata_version(repo) == "0.3.7+test"
