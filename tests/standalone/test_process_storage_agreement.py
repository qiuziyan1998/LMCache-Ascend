# SPDX-License-Identifier: Apache-2.0
"""Production mailbox/agreement methods; CPU threads simulate two TP ranks."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier
from types import SimpleNamespace

# Third Party
import pytest

# Local
from test_checkpoint_prefix_agreement import classes


@pytest.mark.parametrize(
    "phase", ["storage_dense_import", "storage_source_import", "storage_store_import"]
)
@pytest.mark.parametrize("passive_ready", [True, False])
def test_storage_import_requires_both_tp_ranks(phase: str, passive_ready: bool) -> None:
    """A passive import failure must reach the leader through the actual reduction."""
    namespace = classes()
    engine_type = namespace["AscendLMCacheEngine"]
    messages: Queue = Queue()
    rendezvous = Barrier(2)
    flags: list[bool | None] = [None, None]
    participants: list[int] = []
    namespace["Base"]._receive_shared_envelope = lambda self: messages.get(timeout=2)
    engines = [engine_type(), engine_type()]
    for rank, engine in enumerate(engines):
        engine.metadata = SimpleNamespace(is_first_rank=lambda rank=rank: rank == 0)
        engine.shared_cpu_cache_generation = 1
        engine.shared_cpu_cache_strict = True
        engine.config = SimpleNamespace(blocking_timeout_secs=2)
        engine.enable_checkpoint_prefix_agreement()

        def agree(ready: bool, rank: int = rank) -> bool:
            participants.append(rank)
            flags[rank] = ready
            rendezvous.wait(timeout=1)
            result = all(flags)
            rendezvous.wait(timeout=1)
            return result

        engine.collective_all_true_fn = agree
    leader, passive = engines
    leader._broadcast_shared_envelope = messages.put
    with ThreadPoolExecutor(2) as workers:
        leading = workers.submit(
            leader._ordered_checkpoint_agreement, "request", phase, 7, 0, True
        )
        following = workers.submit(
            passive._ordered_checkpoint_agreement,
            "request",
            phase,
            7,
            0,
            passive_ready,
        )
        assert leading.result(3) is passive_ready
        assert following.result(3) is passive_ready
    assert sorted(participants) == [0, 1]
    assert not passive._checkpoint_prefix_results
