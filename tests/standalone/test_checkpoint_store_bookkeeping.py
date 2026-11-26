# SPDX-License-Identifier: Apache-2.0
"""Checkpoint retirement and spill snapshots keep only necessary bookkeeping."""

from threading import Event

import pytest

from test_checkpoint_writeback import deployment as checkpoint_deployment, wait
from test_preemption_checkpoint import api as checkpoint_api

deployment = checkpoint_deployment
api = checkpoint_api


class CountedManifests(dict):
    def __init__(self, entries):
        super().__init__(entries)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


@pytest.mark.parametrize("writeback", [False, True])
def test_exact_generation_forget_avoids_scanning_other_manifests(deployment, writeback):
    engine, create, module = deployment
    worker = create()
    local = worker.local if writeback else module.LocalCheckpointStore(engine)
    manifest = worker.local.manifest("r", 1)
    if not writeback:
        local.publish("r", 1, manifest)
    local.publish("r", 2, manifest)
    local.publish("other", 1, manifest)
    local.manifests = CountedManifests(local.manifests)
    expected = {("r", 2), ("other", 1)}
    try:
        local.forget("r", 1)
        local.forget("r", 1)  # Duplicate completion cannot release twice.
        local.forget("missing", 1)
        assert local.manifests.scans == 0
        assert set(local.manifests.keys()) == expected
        local.forget("r")
        assert local.manifests.scans == 1
        assert set(local.manifests.keys()) == {("other", 1)}
        if writeback:
            wait(lambda: all(count == 1 for _, count in local.charges.values()))
            assert set(local.records) == {("other", 1)}
            assert local.used > 0  # The other request retains shared pages.
    finally:
        local.forget("r")
        local.forget("other")
        if writeback:
            wait(lambda: local.used == 0)


def test_spill_snapshot_reuses_entries_but_isolates_lists_and_retains_pages(deployment):
    engine, create, _ = deployment
    worker = create()
    local = worker.local
    record = local.records["r", 1]
    originals = [list(group) for group in record.groups]
    refs = {id(page): page.refs for group in originals for _, page in group}
    selected, resume = Event(), Event()
    observed = {}
    spill = local._spill

    def inspect(record, groups):
        observed["recreated"] = sum(
            old is not new
            for original, snapshot in zip(originals, groups, strict=True)
            for old, new in zip(original, snapshot, strict=True)
        )
        observed["lists"] = all(
            original is not snapshot
            for original, snapshot in zip(record.groups, groups, strict=True)
        )
        observed["held"] = all(
            page.refs == refs[id(page)] + 1 for group in groups for _, page in group
        )
        marker = object()
        groups[0].append(marker)
        observed["isolated"] = marker not in record.groups[0]
        groups[0].pop()
        selected.set()
        assert resume.wait(5)
        return spill(record, groups)

    local._spill = inspect
    engine.free = 0
    local.wake.set()
    try:
        assert selected.wait(5)
        assert observed == dict(recreated=0, lists=True, held=True, isolated=True)
        local.forget("r", 1)
    finally:
        resume.set()
    wait(lambda: local.used == 0)
    assert all(page.refs == 1 for page in engine.backend.pages.values())
    assert not engine.puts
