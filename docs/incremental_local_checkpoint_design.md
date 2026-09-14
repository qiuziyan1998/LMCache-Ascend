# Incremental local checkpoint saving across repeated preemption

Status: implemented and CPU-audited on 2026-09-13;
NPU correctness/performance qualification remains pending. See section 11.
Branch inspected: `fix/decoder-resume-dp-sync`, 2026-09-12.
The preceding recovery fixes passed 281 CPU tests. The initial implementation
passed 320; the subsequent independent audit expanded this to 346. This does not
replace NPU qualification. See `incremental_checkpoint_audit_20260913.md`.

| Repository | Inspected commit |
| --- | --- |
| LMCache-NPU | `7bd9d686c537fda192ab84a1aa059efd1a5eb70b` |
| LMCache-Ascend | `1002120d3d423a1e8efec1e7c7170486b37a45c2` |
| vLLM | `7d1eb8892680ec32ffd2f2d597ed4171b00750f1` |
| vLLM-Ascend | `58cbdbbc15a9fd1afcc31223b9fe018105dd0f1f` |

## 1. Objective and limits

Reduce allocation, metadata preparation and D2H work when the same request is
preempted repeatedly. Reuse valid completed chunks, rebuild the changing partial
chunk, and capture the new suffix. Generated KV remains temporary, DP/process-local
LocalCPU data. The immutable original prompt keeps its existing persistent source.

The target is smaller preemption pauses and less disruption to other requests.
This does not promise negligible overhead or higher throughput without an NPU
comparison. An early eviction hole can remove most of the incremental benefit.

The first implementation must retain:

- Existing accepted/computed-token sealing and generation checks.
- Existing contiguous prepared group capture, at most one submission per group
  and one final capture fence when there is actual D2H work.
- Existing partial allocation, bounded reclaim, deadline and quarantine behavior.
- The new coordinated shorter-prefix retry when local checkpoint data disappears.
- Existing MC2 limits, MTP semantics, graph gates, DP communication and HBM admission.

Do not add a dedicated CPU pool, periodic saving, per-token polling, generated-tail
Mooncake I/O, per-page transfer workers, or a new C++ kernel. Do not change vLLM core.
Periodic decode-window saving remains a separate feature and is disabled in the
target deployment.

## 2. Code anchors and current gaps

Line numbers refer to the commits above. Use function names after edits.

| Repository / file | Anchor | Relevance |
| --- | --- | --- |
| LMCache-NPU `lmcache/integration/vllm/vllm_v1_adapter.py` | `RequestTracker.update`, around 935 | Resets old-table nonresidency on preempted resume. Preserve this correction. |
| Same | `_add_completed_cold_resume`, around 11700 | Installs the new successful restore frontier, which can exceed the prompt length. |
| Same | `_build_preemption_controls`, around 11754 | Records original prompt end, block tables and current latent resident boundary. |
| Same | `_start_load_kv`, around 7400 | Releases Group-1 CPU lease after completed checkpoint resume; active Group-0 sources remain owned. |
| Same | `_init_decode_window_save_start` / `_add_decode_window_save_metas`, around 10781 / 10869 | Existing aligned append frontier and completion discipline; do not enable or mutate this persistent-save path. |
| LMCache-Ascend `lmcache_ascend/integration/vllm/vllm_v1_adapter.py` | `handle_preemptions`, 1379 | Cancels old offers, captures using active worker state, then drops that state. |
| LMCache-Ascend `lmcache_ascend/v1/preemption_checkpoint.py` | `CaptureJob`, 45; `capture`, 102 | Group 0 already references prior CPU data. Group 1 still captures from the original aligned prompt boundary each time. |
| Same | `_publish_local`, 306; `_release`, 421 | Private captured pages, accepted sealing, local manifest publication and owner retirement. |
| LMCache-Ascend `lmcache_ascend/v1/local_checkpoint.py` | `LocalCheckpoint`, 19; `acquire`, 49; `normalize`, 106 | Advisory offers, paired prefix acquisition, boundary assembly and canonical admission. Normalization currently hashes the full input separately for each group. |
| LMCache-Ascend `lmcache_ascend/v1/cache_engine.py` | `_direct_suffix_plans`, around 1455 | Existing pattern: hash Group 0 once and derive other group keys from hashes and lengths. Reuse the pattern, not its persistent-store state. |
| LMCache-NPU `lmcache/v1/token_database.py` | `process_tokens_from_prefix`, 541 | Public incremental hashing API requiring a validated, chunk-aligned seed. |
| LMCache-Ascend `lmcache_ascend/v1/npu_connector/npu_connectors.py` | `prepare_group_capture`, 4300; `enqueue_group_capture`, 4405 | Requires contiguous ranges and packed slot mapping; submits the existing prepared D2H operation. |
| LMCache-NPU `lmcache/v1/storage_backend/local_cpu_backend.py` | `batched_get_layer_page_prefix`, 772; `batched_submit_layer_pages`, 475 | Acquires references; compatible existing admissions are skipped. Neither refreshes reused pages' LRU priority. |
| Same | `touch_cache`, 369 | Uses shared `keys_in_request`; do not populate that list or introduce lookup pins solely to change checkpoint LRU order. |

## 3. Frontiers and ownership

Use half-open token intervals throughout:

| Symbol | Meaning |
| --- | --- |
| `C` | LMCache chunk size. |
| `P` | Immutable original prompt length. It is not advanced by checkpoint saving. |
| `B = floor(P/C)*C` | Start of the prompt's overlapping tail chunk. |
| `F` | Group-0 nonresident/remap frontier of the current block table. |
| `R0 = max(B,F)` | Existing `CaptureSpec.resident_start`; earliest Group-0 position this capture may read from HBM. |
| `E` | Snapshot/capture upper bound. It can include optimistic speculative positions. |
| `S` | Final accepted, computed seal end selected by existing sealing code; `S <= captured_end <= E`. |
| `Q1` | End of the contiguous Group-1 CPU prefix actually acquired at this preemption. |
| `H = floor(Q1/C)*C` | Group-1 capture start when replacing an open tail or repairing a hole. At minimum use `B`. |
| `Q` | Already reusable paired checkpoint frontier, including an existing partial tail when both groups cover it. |

After successful checkpoint restoration, `F` can extend into generated tokens.
Those latent pages participate in subsequent sparse retrieval. Group 0 below `F`
must not be read from nominal main-HBM slots: they may be null or scratch/remapped
storage. Group 1 remains densely resident in its valid indexer block table, so
evicted Group-1 CPU pages can be captured again before that table is released.

Do not derive `F` from `P`, an old generation's end, or the aligned release counter.
Use the current capture specification and validated worker state. Capturing or
publishing a new checkpoint does not advance `F`; only successful restoration does.

Active Group-0 sources are protected by ownership. Ordinary LRU should not evict
them. If a required Group-0 source is unavailable below `R0`, shorten the offer or
use existing recovery; never infer that its old HBM slot still contains valid KV.

### Attention semantics and the cost after resume

Advancing `F` must change the source of selected latent KV, not logical sequence
lengths, causal positions or the selected attention token set. In vLLM-Ascend,
`attention/utils.py::_dsa_remap_frontier` (413) supplies the remap frontier;
`attention/sfa_v1.py::_resolve_sparse_cached_tokens_by_request` (412) and
`_prepare_sfa_remap_boundary` (444) prepare its sparse-source interpretation.
The indexer calls around 2530–2570 use query/key sequence lengths and the indexer
block table. The remap path around 4793–4828 updates the split boundary while
retaining the original absolute selected indices. Preserve these responsibilities;
incremental checkpoint saving must not introduce a second frontier policy.

These code anchors explain the intended separation, but do not alone prove NPU
attention equivalence. Test the same accepted history and logical sequence length
with `F` near the prompt boundary and with `F` advanced into generated tokens.
Compare selected absolute indices, restored KV bytes and outputs before accepting
the change. Exercise both staged graph execution and the existing bounded fallback.

A larger `F` also increases the active Group-0 CPU working set and can move selected
generated-token reads onto the sparse CPU-to-NPU retrieval path. Measure steady
decoding after resume as well as capture/restore latency. Lower D2H volume alone
does not prove lower whole-run TPOT. Retain the existing successful-restore frontier
and graph eligibility rules; do not weaken either to improve a timing result.

### Example matching the observed long-request layout

Take `C=1024`, `P=131600`, `B=131072`, and a successfully restored frontier
`F=132233`. Assume the next captured and sealed end is `132500` for simplicity.

- Reuse Group-0 CPU sources through `132233`; capture Group 0 only over
  `[132233,132500)` — 267 tokens.
- If Group-1 CPU data survives, reuse its complete chunk `[131072,132096)`.
  Capture Group 1 over `[132096,132500)` — 404 tokens, replacing its open tail.
- Existing behavior captures Group 1 over `[131072,132500)` — 1428 tokens.
- If `[131072,132096)` was evicted from Group 1, capture from `131072` instead,
  repairing the hole and saving the new tail in one contiguous submission.
- If the old Group-1 partial page through `132233` survives but replacement
  allocation fails, retaining an offer through `132233` is better than dropping
  to `132096`. Its old partial key is valid for that shorter lookup.

For this model's BF16 79-layer layouts, logical bytes per token are approximately
`79*576*2` for Group 0 and `79*128*2` for Group 1. Use actual connector shapes/dtypes
in code; do not hardcode these example dimensions or confuse logical bytes with
allocator-aligned bytes.

## 4. Minimal state changes

Extend existing dataclasses rather than adding a new manager or background service.

- `CaptureJob`: keep current Group-0 `prefix_sources`; add Group-1 reused source
  descriptors and a list of temporarily retained CPU owners. Add optional immutable
  known-full-chunk descriptors and an aligned hash seed for key planning.
- `LocalCheckpoint`: carry the optional known-full-chunk descriptors and seed into
  the existing manifest. Defaults must preserve existing construction and fallback.
- Keep `CaptureSpec`, `SealSpec`, result/control wire formats and scheduler APIs
  unchanged unless implementation reveals a concrete missing contract.

Suggested metadata-only fields are `known_chunks: tuple[CheckpointPage, ...]` for
canonical full Group-0 tail chunks and `key_seed: tuple[int, int | bytes] | None`
for the complete prefix before them. These contain values, not device pointers or
allocator ownership. Do not retain the mutable `WorkerRetrieveState` object: its
lists are cleared when preemption drops the old state.

Temporary owner references last only through capture, sealing/publication or the
existing failure/deadline cleanup. Waiting offers must still retain no allocator
references or pins. Extend `_release(job)` to cover every acquired owner, including
the all-reused, cancelled and allocation-failure cases.

Suggested implementation boundaries:

| Location | Responsibility |
| --- | --- |
| `CheckpointWorker` private `_plan_prefix_reuse(spec, state, job)` | Snapshot canonical Group-0 metadata, acquire reusable sources, and return the Group-1 capture start, already reusable paired end, and safe capture limit. |
| `CaptureJob` | Own temporary page references and immutable metadata until the existing publication/cleanup point. |
| `AscendLMCacheEngine` small `checkpoint_token_plans(...)` helper, or equivalent private helper in `LocalCheckpointStore` | Build both canonical group key plans once; use only public token-database APIs. |
| `LocalCheckpointStore.normalize` | Consume those plans while retaining its existing byte assembly, admission and miss handling. |
| `LocalCPUBackend.try_touch_layer_pages(...)` | Best-effort checkpoint-event recency update; no access to backend-private fields from Ascend callers. |

## 5. Capture planning and hole repair

Implement one private reuse-planning helper inside `CheckpointWorker`, using public
engine/backend APIs. Keep storage-layout validation in the existing engine helpers.

1. In `handle_preemptions`, pass the active state before `_drop_worker_retrieve_state`.
   The old manifest has already been cancelled there: do not obtain reuse metadata
   from the previous manifest or move cancellation/binding of unrelated connectors.
2. Qualify incremental reuse only for the existing local checkpoint profile, with
   periodic decode-window save disabled. The adapter already has the resolved
   `_decode_window_save_window_size`; use it at this event call site, without
   reparsing environment variables or adding a normal decode check.
3. Validate request identity and shared/pointer generation against the active state.
   Reuse the existing validated-source contract: `shared_request_active`, present
   latent source, current shared generation, and actual contiguous key/range
   metadata. Missing seed information disables incremental hashing. Stale identity,
   generation or invalid layout must not be accepted as cache data.
   For an already resumed sparse request, check `indexer_npu_resident` and absence
   of pending indexer materialization before using its HBM as a repair source.
4. Snapshot Group-0 physical keys using `without_layer()`, plus absolute ranges.
   Retain needed generated-tail CPU sources for the short capture/publication job.
   This extends their existing ownership rather than copying their bytes. Include
   any boundary source required for replacement; do not retain the whole immutable
   prompt merely to save its tail.
5. Derive Group-1 keys from the same known Group-0 hashes and exact lengths with
   `token_database.process_tokens(hashes=..., offsets=..., kv_group=1, ...)`.
   This preserves dtype, payload-layout, valid-token and index-schema tags without
   rehashing tokens. Its returned positions are relative: retain the original
   absolute ranges when pairing the resulting keys with descriptors.
6. Use `batched_get_layer_page_prefix` to acquire the contiguous Group-1 CPU prefix
   under one lock. Validate each returned page's group layout and token coverage.
   An ordinary miss stops reuse. Do not probe every layer or search Mooncake for
   generated data. Keep a valid final partial page as an optional fallback source.
7. Set Group-1 capture start to the first missing complete chunk or the open-tail
   boundary `H`. Group-0 capture start remains `R0`. The allocation loop can begin
   at `H` instead of `B`, continuing to use `max(cursor,R0)` for Group 0.
8. Validate block IDs only for nonempty ranges actually read from HBM. Preserve
   positive-block and source-extent checks; never expand Group-0 reads below `R0`.
9. Keep the existing bounded paired allocation/reclaim logic. If replacement cannot
   extend past an already reusable partial frontier `Q`, retain `Q`, discard any
   redundant newly allocated overlap and submit no redundant D2H. Do not let a
   smaller allocation shrink an already valid saved prefix.
10. Prepare all necessary allocations before device work. Each group's new intervals
    remain contiguous and use one existing prepared group submission. If there is
    no new/repair work, skip preparation, enqueue and capture fence entirely.

The first version intentionally does not skip cached islands after the first
Group-1 hole. It recaptures the contiguous suffix. An early hole can therefore
approach current capture cost, but never justifies relaxing the kernel's contiguous
interval checks or reading nonresident Group-0 positions.

Before advertising `captured_end`, verify that retained CPU coverage plus newly
captured coverage is contiguous in both groups from the original prompt boundary.
An unrepairable Group-0 gap caps this end. If no generated prefix survives, use the
existing no-checkpoint recovery outcome without copying an unusable suffix.

Control outline (pseudocode, not a replacement for existing validation):

```text
reuse = plan_prefix_reuse(current_spec, active_state)
limit = min(E, reuse.safe_group0_limit)
old_end = min(reuse.paired_cpu_end, limit)

if old_end == limit and old_end > P:
    capture_nothing; offer old_end
else:
    allocate existing contiguous suffixes up to limit
    if selected_end <= old_end:
        release redundant new allocations; capture_nothing; offer old_end
    else:
        prepare/enqueue only nonempty groups
        wait for the existing final capture event
        offer selected_end

seal later using accepted history
select either the retained old partial or its replacement; never both overlapping
publish the new advisory local manifest
release temporary job owners through the existing completion path
```

The reusable frontier is a data-coverage result, not merely the previous offer's
end. Group-1 key probes are limited to ranges with validated Group-0 hash metadata.
If active metadata ends before some old cached island, do not guess that island's
key from the last partial hash. Capture the required resident suffix instead.

## 6. Sealing, partial replacement and incremental keys

Keep canonical normalization at its existing restore stage in this implementation.
Moving normalization or original-prefix fetching into the publication executor
would be another lifecycle change and is not required to reduce blocking capture.

In `_publish_local`:

- Merge reused descriptors with new private fragments under the accepted `S`.
- Complete reused chunks keep their existing keys and objects.
- If new capture extends the old partial page, select the replacement fragments
  for that interval; never publish overlapping old/new Group-1 coverage.
- If sealing leaves no accepted extension beyond `Q`, reuse the old partial page
  and discard redundant speculative capture. The old partial key still describes
  a valid shorter checkpoint even though it cannot describe the longer sequence.
- Clip descriptor coverage to `S`, while keeping physical page lengths intact for
  correct plane offsets. Never publish rejected speculative or uncomputed KV.
- Do not call persistent-store APIs or update persistent RemoteFill outcomes.

In `normalize`, replace full-history hashing for both groups with one key-plan helper:

1. Preserve the existing generation/history validation and paired acquisition.
2. Reuse validated full Group-0 chunk keys for unchanged ranges beginning at `B`.
3. For the requested restore end `L`, choose the last usable complete boundary
   `A <= floor(L/C)*C` from known metadata. A shorter retry can require an earlier
   anchor than the previous checkpoint's last complete chunk.
4. Seed `process_tokens_from_prefix` with the hash for `[0,A)`, then hash only
   `[A,L)`. An old partial hash is never a seed for extending that same chunk.
5. Derive Group-1 keys from the resulting hashes and lengths, following
   `_direct_suffix_plans`; do not reuse that function's mutable persistent-store
   state. Validate equal absolute ranges and chunk hashes across groups.
6. If no valid seed exists, use the current full-hash path. This is a performance
   fallback, not permission to infer or fabricate a seed.
7. Reuse current exact-page admission and `_assemble` for boundary/cropped pages.
   Retain actual canonical winners before device restoration and preserve the
   admission-miss/alias cleanup introduced in the recovery fixes.

Populate the seed while the active worker state still exists: find the canonical
full chunk ending exactly at `B`, and copy its hash value. Snapshot known full
chunks from `B` onward from the same validated state. For a restore truncated
within those known chunks, select an earlier complete anchor from that snapshot.
If `B=0` or no valid preceding key is available, retain the existing initial-hash
path. Do not keep references to lists that `_drop_worker_retrieve_state` clears.

Using an active-state seed relies on vLLM's immutable accepted-prefix contract.
Do not broaden the fast path to a history-rewriting or unvalidated source path.
Existing history validation must remain; a questionable seed falls back to full
key construction rather than accepting an unproved association between bytes and
tokens.

Normal token lookup may still hash its query. This optimization removes redundant
checkpoint key construction; it does not claim that the entire control protocol
has constant cost. Accepted-history validation and metadata scans remain necessary.

## 7. LRU ordering without lifetime changes

Reusing old pages without refreshing their recency can put them ahead of newly
saved tails in the eviction queue. Re-submitting compatible pages does not fix
this: `batched_submit_layer_pages` skips those mappings.

Add a small LocalCPU public helper, for example
`try_touch_layer_pages(keys: Sequence[CacheEngineKey]) -> bool`:

- Use the existing LRU policy and an explicit ordered key list.
- Prefer a nonblocking cache-lock attempt; skip this best-effort priority update
  if busy. It must not determine whether checkpoint bytes are valid.
- Touch present layer-page entries through `cache_policy.update_on_hit`, in the
  caller's order. Do not alter pins, object references or `keys_in_request`.
- Leave other cache policies and existing `touch_cache()` behavior unchanged.

At local checkpoint publication, order the offer's keys by logical chunk start,
latest first and earliest last, across both groups. Include reused keys, not only
new private fragments. Refresh canonical keys similarly after normalization.
Do not force-remove an obsolete canonical partial key: another request or a
shorter restore may still use it. Only retire this generation's references and
private aliases through existing ownership rules.

This is a checkpoint-event priority hint, not a new eviction policy or a pin held
while waiting for HBM. LRU remains free to evict idle offers afterward.

## 8. Failure matrix

| Condition | Required behavior |
| --- | --- |
| Group-1 closed chunk evicted after resume | Stop reuse there; repair from valid dense indexer HBM and capture the suffix. |
| Cached islands after that hole | Recapture in the same contiguous run; no per-island workers or kernel changes. |
| Group-1 old partial survives, replacement allocation fails | Retain the old paired partial frontier if valid; do not reduce it to a full-chunk boundary unnecessarily. |
| Group-0 page unavailable below its current nonresident boundary | Do not read main HBM there. Cap coverage/use existing shorter-prefix recovery. |
| CPU allocation/reclaim refuses | Preserve valid existing and newly completed paired coverage; bounded retry/shrinking only. |
| Eviction after lookup but before restore acquisition | Existing `CheckpointRestoreMiss` and all-worker retry protocol. |
| Eviction attempt while a capture/restore owns the page | Existing reference/pin rules prevent physical reclamation. |
| No new accepted KV | Reuse the prior valid offer; avoid D2H when no repair is needed. |
| Rejected speculative tail | Clip at sealing; never reuse an optimistic hash as an accepted key. |
| Cancellation/deadline/failed allocation | Release every temporary borrowed/new owner; no offer resurrection. |
| Unknown native completion | Existing quarantine/restart behavior; no unsafe release or retry-as-miss. |
| Invalid source identity/layout/generation | Keep existing fail-closed diagnostics; do not treat corruption as an ordinary hit. |
| Periodic decode-window saving enabled | Preserve that feature's existing path and counters; incremental local-only optimization stays unqualified initially. |

## 9. Incremental implementation sequence

Each step must retain a runnable branch and can be committed after its checks pass.

1. **Add failing tests for repeated capture and eviction.** Extend existing CPU
   fixtures using production key classes. Count allocation/capture intervals,
   references and submitted groups; verify data bytes against non-incremental
   capture. Do not start by modifying graph or scheduler code.
2. **Implement reuse planning and temporary ownership.** Change `CaptureJob`,
   `CheckpointWorker.capture`, `_publish_local`, `_release`, and the Ascend adapter
   call site. Include first-hole repair, preserved old partial fallback, and
   zero-work handling. Keep the current key hashing initially.
3. **Add incremental key metadata/planning.** Change `LocalCheckpoint` and
   `normalize`, reusing public token-database APIs. Prove exact key equality with
   the full-hash reference before enabling this optimization.
4. **Add scoped LRU refresh.** Add the small LocalCPU helper and event-only callers.
   Verify priority without changing active ownership or unrelated lookup state.
5. **Perform a separate audit and run all existing recovery tests.** Check rollback
   frontiers, generation matching, typed misses, cancellation, GC-disabled cleanup,
   derived dispatch and unchanged normal model paths.
6. **NPU qualification before judging performance.** Compare with the committed
   baseline above under the same inputs, GC, perf level, graph sizes and concurrency.

Expected production edit scope:

| Area | Estimated added/modified lines |
| --- | ---: |
| Capture reuse, repair and owner lifecycle | 50–80 |
| Incremental canonical-key planning | 30–45 |
| Scoped LRU helper and calls | 15–25 |
| Qualification/call-site plumbing | 5–10 |

Aim for roughly 100–150 production lines, with a review checkpoint around 160.
These are estimates, not a reason to compress validation or skip ownership rules.
Allow approximately 150–250 test lines by extending existing fixtures. If the
implementation needs a new transfer protocol or broad scheduler changes, narrow
the scope and revise the estimate before proceeding.

## 10. Validation and performance acceptance

CPU tests must cover:

- At least three capture/resume generations with advancing `F`, stable full keys,
  changing partial keys and exact reconstructed bytes for both groups.
- A shorter successful restore followed by another preemption; no old-table
  frontier may reappear.
- Local Group-1 holes at the first, middle and final chunk, with later cached
  islands, and the all-missing case.
- Whole reuse, partial-only reuse and no-new-KV cases; no redundant D2H/fence.
- Aligned/misaligned prompt and restore ends, a partial-only tail, a chunk-closing
  extension, MTP clipping and a seal/retry shorter than the saved hash anchor.
- Incremental/full key equality including payload, index-schema and valid-token
  tags; no anchoring on an old partial hash.
- Allocation failure that preserves the old partial fallback, and a valid partial
  repair that extends coverage through a previously missing Group-1 chunk.
- Nonresident Group-0 source loss without illegal HBM reads.
- Eviction eligibility during and after borrowed ownership, shared canonical
  pages, cancellation, deadlines, duplicate replies and GC disabled.
- LRU refresh touching only explicit checkpoint keys, skipping a busy lock and
  leaving unrelated `keys_in_request`, pins and other policies unchanged.
- Existing contiguous-kernel validation, unknown-DMA quarantine, all-worker miss
  retry, and actual Ascend/dynamic/multi-connector dispatch.

Use the workspace's existing standalone command in each repository:

```text
python -m pytest --confcutdir=tests/standalone tests/standalone -q --tb=short
```

On NPU, run both a no-preemption control and forced repeated preemption of the
same long request. Exercise retained CPU pages and deliberate Group-1 eviction
after successful resume. Validate deterministic output/KV data against a controlled
reference before relying on aggregate benchmark success counts.

Measure capture preparation/fence time, bytes and allocations per group, reusable
frontier, actual retry frontiers, graph/native step counts, and both the preempted
request's pause and unaffected requests' TPOT/p99 latency. Start with existing
timings and counters from test instrumentation; add at most one gated scalar plan
event if needed for hardware attribution, not broad per-layer logging.

Acceptance criteria:

- With retained closed chunks, new capture volume is bounded by the new suffix
  plus the changing Group-1 partial overlap; unchanged complete chunks incur no
  payload copy or allocation.
- Holes produce valid repaired/shortened coverage, not incorrect success or an
  unbounded retry loop.
- Captured and restored token data, accepted output history and native TP/DP/MTP
  semantics remain correct.
- No new per-token/model-step work or native code is introduced. No-preemption
  throughput must remain within measured run-to-run variance.
- Repair under early eviction may approach baseline capture volume. Report that
  limitation explicitly rather than claiming all cases scale only with new tokens.

## 11. Implementation and separate audit, 2026-09-13

The implementation is on `fix/decoder-resume-dp-sync`. The inspected commits above
remain its base.

Production changes are limited to four Python files:

- Ascend `preemption_checkpoint.py`: borrow validated active Group-0 sources,
  derive and acquire the contiguous Group-1 CPU prefix, and retain those owners
  until capture/publication retirement. Capture starts at the first index hole or
  changing partial boundary. Group-0 reads never extend below the current resident
  frontier. If that nonresident source is missing, coverage is capped before the
  hole. Existing paired allocation, partial progress, deadline and fencing remain.
- Ascend `local_checkpoint.py`: immutable full-chunk keys and aligned seed travel
  in the local manifest; normalization hashes only the unproved suffix and derives
  Group-1 keys through the existing token-database API. Missing seed information
  uses one full Group-0 hash pass. Sealing retains a valid old partial instead of
  replacing it with an incomplete or entirely speculative extension.
- Ascend adapter `handle_preemptions`: passes the already-resolved decode-window
  setting. Incremental capture reuse is qualified only for prepared active sources
  with periodic decode-window saving disabled. The existing capture path remains
  available otherwise. No new runtime configuration is introduced.
- LMCache-NPU `LocalCPUBackend.try_touch_layer_pages`: explicit-key LRU refresh
  with a nonblocking lock, no pins or references, and no changes to the ordinary
  lookup-touch list. Both private publication and canonical normalization refresh
  reused leading pages after later pages.

The formatted production diff is **251 insertions / 25 deletions** (net 226).
This exceeds the initial 100–150-line estimate: the review retained explicit
generation/coverage checks, per-group ownership, safe partial replacement and
readable formatting. It introduces no new manager, transfer protocol, scheduler
change, model-step callback or native kernel. vLLM and vLLM-Ascend are unchanged.

Verification completed:

| Repository | Standalone CPU tests |
| --- | ---: |
| LMCache-NPU | 41 passed |
| LMCache-Ascend | 217 passed |
| vLLM | 16 passed |
| vLLM-Ascend | 46 passed |
| Total | 320 passed |

The new matrix uses production chunk/key methods and counted fake CPU allocations
to check byte equality across both groups, three checkpoint generations, aligned
and partial boundaries, index holes and later islands, allocation fragmentation,
short seals, absent seeds, stale identity/generation, unavailable nonresident
latent data, cancelled publication and cleanup with cyclic GC disabled. Existing
tests continue to cover all-worker restore misses and unknown-transfer fencing.
Dynamic Ascend/MultiConnector tests verify the new qualifier and restoration of
ordinary dispatch. The explicit LRU tests check ownership and nonblocking behavior.

Concrete capture checks include Group-1 `[4,19)` becoming `[12,19)` after a
13-token restore (chunk size 4, original prompt length 6), with Group-0 capture
still starting at 13. Fully reused or non-extending captures issue no D2H command,
no completion fence and no empty page admission. First preemption with no reusable
generated pages does not acquire the cache lock for empty prefix probes.

The separate audit compared method ASTs against the baseline: all 62 pre-existing
LocalCPU methods are unchanged; the only altered method among the Ascend adapter's
39 methods is `handle_preemptions`. New logic is confined to checkpoint capture,
publication and normalization. No extra work is added to ordinary decode methods.
Focused Python error checks and `git diff --check` pass. Full lint still reports
pre-existing synchronous-lambda B023 and exception-chaining B904 findings in the
checkpoint modules; this change does not alter those existing control paths.

Deployment needs matching LMCache-NPU and LMCache-Ascend Python changes. It needs
no additional C++ rebuild beyond the prepared checkpoint kernel already required
by the baseline. Keep the existing checkpoint enablement and disable periodic
decode-window saving for incremental capture reuse.

NPU byte/output equivalence, real stream ordering under pressure, the resumed
request's steady decode cost and measured throughput/TTFT gains are still open.
Use the hardware matrix in section 10. CPU tests establish reduced planned capture
work and lifecycle invariants; they do not demonstrate an NPU speedup.

The subsequent independent audit is recorded in
`incremental_checkpoint_audit_20260913.md`. It adds 26 combined-pressure and
ownership tests (346 total), with no additional production change.
