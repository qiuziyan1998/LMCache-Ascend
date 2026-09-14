# Incremental local checkpoint: independent audit, 2026-09-13

Scope: the implementation of `incremental_local_checkpoint_design.md`
on `fix/decoder-resume-dp-sync`. The four base commits are recorded in that plan.
The production diff remains 251 insertions / 25 deletions in four Python files.
This audit added tests and documentation; no additional production defect was
proved, so no further production changes were made.

## Findings and verification

| Area | Code and result |
| --- | --- |
| Derived implementation enters | Ascend `handle_preemptions` passes active worker state before dropping it. The dynamic Ascend connector and MultiConnector tests exercise that actual dispatch and both values of the decode-window qualifier. Only the existing first-rank checkpoint writer captures replicated KV. |
| Ordinary decode cost | The only changed existing Ascend adapter method is `handle_preemptions`. Existing LocalCPU methods, vLLM, and vLLM-Ascend are unchanged. New recency calls are in checkpoint publication/normalization, not model layers or ordinary decode. No new polling loop, graph gate, configuration read per token or device timing was introduced. |
| Both groups and repeated preemption | The active Group-0 source supplies canonical hashes. Its current nonresident frontier controls legal HBM reads. Group 1 reuses the available contiguous CPU prefix and repairs the remaining suffix from its dense resident HBM. Missing Group-0 data below that frontier caps usable coverage. Three-generation byte tests cover advancing frontiers and replacement partial chunks. |
| CPU pressure and fragmentation | Existing paired fragment allocation, one bounded reclaim attempt and shortened final fragment are retained. A saved old partial wins when a replacement cannot extend it. Combined-hole tests exercise simultaneous latent loss, index eviction and one-token allocation limits. Failure to bridge the original prompt boundary correctly returns no generated checkpoint. |
| LRU and active ownership | `LayerPageMemoryObj` inherits the locked tensor-object reference accounting; `can_evict` requires no pins and only the cache reference. Reuse acquires references before validation/capture. Publication, cancellation and deadline cleanup retire them explicitly. Waiting offers keep descriptors rather than allocator references. LRU refresh uses a nonblocking lock and does not mutate pins, references or `keys_in_request`. |
| Hashes and partial replacement | Normalization preserves accepted-history validation, uses only full-chunk hash anchors and hashes the changing suffix through the production token-database API. Group-1 key construction retains its own schema/dtype namespace. Missing seed information falls back to one full hash pass. Cropped descriptors retain the physical page length for plane offsets. Tests compare canonical keys and bytes at shorter seals and aligned/partial boundaries. |
| Stream order and HBM reuse | New capture uses the existing contiguous prepared-group kernel: source work precedes store-stream capture; one final event covers nonempty group submissions. Source blocks cannot be reused before capture returns. Fully reused checkpoints submit no native work or fence. The native binding still releases the GIL after converting Python arguments. |
| Unknown native completion | A failed capture fence quarantines newly allocated and borrowed sources. Cancellation does not erase this quarantine. The new test checks that borrowed source references survive the terminal failure; normal cleanup is only performed in test teardown where no device operation occurred. |
| Asynchronous publication and GC disabled | Cancellation during blocked publication retains borrowed pages until the writer finishes. The completion path removes cancelled manifests and releases every counted owner. Mutable worker-state lists can be cleared after capture without changing copied descriptors/hash metadata. Existing tests and the new deadline test verify explicit cleanup without requiring cyclic GC. |
| Restore/admission and fallback | Existing all-worker generation matching, readiness checks, `WAITING_FOR_REMOTE_KVS`, local restore-miss coordination and bounded shorter retry are unchanged. Borrowing an old prefix does not itself admit a request or bypass HBM allocation/readiness. A checkpoint remains advisory until actual restore acquisition re-proves its data. |
| RemoteFill and early reservations | In `remote_fill/state.py`, native-armed windows cannot finish; only terminal `READY_HIDDEN` reservations reach lifecycle commit. Ascend `remote_fill.py::commit_pages` publishes them through the atomic LocalCPU commit APIs. A new hidden reservation is not a readable canonical LocalCPU page, so checkpoint reuse cannot consume it as a hit. Existing ready pages are not overwritten by that reservation. |
| Local/remote Mooncake placement | The original prompt retains its persistent two-group lookup proof. Generated checkpoint data remains LocalCPU-only. This optimization adds no Mooncake requests or placement change. Existing original-boundary loading and mixed original-prefix/local-tail restore remain in place; neither remote placement nor an early Group-1 reservation authorizes reuse of missing generated KV. |
| Prefiller and pending stores | Prefiller paths are unchanged. The existing preemption callback drains pending/direct stores for its victims before capture, and preserves producer events, RemoteFill publication and shared transport ordering. The optimization does not replace their fences or claim those waits have disappeared. |
| CUDA/ACL graphs and MTP | No scheduler, model, graph eligibility, sampling or speculative-token semantics changed. Only accepted/computed positions enter the sealed checkpoint; fewer captured bytes do not justify graph replay before normal readiness. The Group-0 remap frontier still advances only through successful restore. |

The lock-order review found no new nested manifest/LocalCPU lock acquisition:
manifest publication releases its lock before recency refresh; normalization
acquires page owners before updating the manifest and releases that lock before
cache alias operations. The LRU hint can be skipped if its lock is busy. Existing
allocation and page-acquisition locks remain; this does not claim that preemption
is wait-free under contention.

## Additional independent regression matrix

`tests/standalone/test_incremental_checkpoint_audit.py` adds 26 cases:

- 24 combinations of missing nonresident latent pages, index holes/cached islands,
  and normal versus one-token fragment capacity.
- Deadline expiry with borrowed references outstanding.
- Unknown native capture completion with borrowed references outstanding.

Successful combinations must restore exact bytes for both groups, never read
null latent HBM slots below the current frontier, and release all temporary
references. Under a one-token allocation limit, an index hole beginning at token
4 cannot bridge an original prompt ending at token 6 within the existing bounded
fragment policy. That expected refusal is tested explicitly, not changed into
an unsafe success or an unbounded allocation loop.

The prior incremental suite also covers old-partial fallback without DMA, no-op
capture without an empty cache-lock probe, canonical key equivalence, source
generation/identity rejection, indexer readiness, first/middle/tail index holes,
LRU ordering, immutable metadata snapshots and cancellation during publication.

## Results

Command run in each repository:

```text
python -m pytest --confcutdir=tests/standalone tests/standalone -q -o log_cli=false --tb=short -p no:cacheprovider
```

| Repository | Result |
| --- | ---: |
| LMCache-NPU | 41 passed |
| LMCache-Ascend | 243 passed |
| vLLM | 16 passed |
| vLLM-Ascend | 46 passed |
| Total | 346 passed |

The new test files pass the repository's Ruff rules. Focused Python error checks
and whitespace checks pass for the production changes. Pre-existing lint findings
documented in the implementation plan were not treated as new functional bugs.

## Remaining hardware evidence

No NPU/CANN runtime is available locally. These results establish host-side
contracts and planned byte equality through explicit fake transfer boundaries;
they do not prove every accelerator interaction or negligible throughput loss.

Qualification still requires matched no-preemption controls and repeated forced
preemption with retained/evicted index pages. Compare output/KV correctness,
capture preparation/wait, publication/restore, resumed steady decode, graph/native
counts and unrelated requests' TPOT/TTFT. Long Group-1 holes can recover little of
the intended saving, and a larger restored latent frontier can affect later sparse
retrieval. Existing waits and device resource contention remain measurable costs.

No new native build is required by this incremental change. Deploy the matching
LMCache-NPU and LMCache-Ascend Python modifications with the existing checkpoint
setting; periodic decode-window saving stays disabled for incremental capture reuse.
