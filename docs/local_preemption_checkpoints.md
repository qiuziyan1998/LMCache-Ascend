# Partial local preemption checkpoints

Branch: `fix/decoder-resume-dp-sync`. This replaces the fixed staging slab and
generated-tail Mooncake publication path. It does not change MC2 recovery limits,
model graph eligibility or the ordinary persistent prefix-cache contract.

## Evidence and objective

The September 11 logs showed a 186,384,384-byte contiguous Group-0 allocation
failing with 310,484,992 aggregate free bytes. Reclamation took under 1 ms but
could report `not_needed` despite the allocation failure. Recovery batches then
scheduled 32 positions across four requests, executing in approximately 0.8 s
while emitting about six new tokens per second. Aggregate free bytes cannot
prove that one large allocation fits.

The objective is to retain as much paired, contiguous accepted tail KV as a
bounded allocation attempt obtains. Full restoration can return to ordinary
graph/MTP execution; partial restoration reduces native recovery work. Neither
universal checkpoint success nor negligible throughput cost is guaranteed.

## Data flow

1. Scheduler preemption records original block tables, the immutable original
   prompt end and the nonresident latent frontier. No fixed output-window limit
   rejects long tails. Explicit skip-save/role/layout restrictions still apply.
2. The Ascend preemption hook runs before block reuse and before dropping worker
   retrieve state. Its Group-0 key/range descriptors preserve earlier local-only
   tail sources during repeated preemption without pinning waiting offers.
3. Capture allocates from the existing registered LocalCPU heap in intervals no
   larger than an LMCache chunk. Both groups for an interval must fit before its
   frontier advances. One bounded eviction attempt and shrinking of one final
   interval handle pressure; completed earlier intervals are retained.
4. All selected allocations complete before metadata upload or D2H. Python
   prepares one `[layers, chunks]` pointer matrix per group. The existing native
   group binding submits all chunks and releases the GIL. One final event fences
   capture before HBM reuse. No payload `.cpu()`, `.item()` or per-chunk device
   synchronization is added.
5. After earlier asynchronous outputs are consumed, sealing clips to accepted,
   computed history. The last sampled token and rejected speculative positions
   are not advertised. Private page keys isolate physical captured data from
   ordinary prefix hits; the manifest exposes only its valid logical ranges.
6. Local publication transfers references into the existing LRU cache. There is
   no dedicated idle slab pool and no generated-tail Mooncake get/put. The
   manifest retains keys and accepted history, not allocator ownership.
7. A resumed lookup carries `lmcache.local_checkpoint_generation` internally.
   Only that lookup may combine the independently proved persistent original
   prefix with a fresh paired LocalCPU tail probe. Other lookups call the
   original implementation. Probe references are released before HBM admission.
8. After HBM admission, TP0 reacquires and validates source pages. It reuses
   complete pages directly and assembles only boundary/cropped pages in CPU
   memory. A boundary read uses the original stored prefix length, not a remap
   frontier shortened by one token. Missing boundary data or insufficient
   workspace is a bounded failure, not fabricated coverage.
9. TP0 broadcasts a generation-scoped control acknowledgement before normal
   shared retrieval. Group 0 uses the existing sparse CPU materialization path.
   Group 1 reads full original-prefix chunks through the persistent direct-HBM
   loader, and only boundary/tail chunks through the existing dense CPU loader.
   Their readiness is joined by the existing cold-load coordinator.
10. After all-worker receive completion and the resumed dispatch, Group-1 CPU
    lease ownership retires. Group-0 pages still used by sparse decoding remain
    protected until the existing request-source lifecycle releases them.

## Memory and failure rules

- `can_evict` remains authoritative. Pins or borrowed references from any active
  request/transfer make a candidate ineligible. There is no force-unpin path.
- Eviction work uses the existing 4096-entry budget and nonblocking cache-lock
  attempt. Captured pages and restore leases are excluded by ownership.
- Saving does not require all requested output to fit. Losing one group/page
  shortens the usable common frontier; later pages cannot bridge a hole.
- A smaller final interval is best effort, not a global heap-packing algorithm.
- Boundary assembly may require another chunk-sized allocation. It has one
  bounded reclaim/retry and can still refuse under fragmentation or pressure.
- Moving a raw checkpoint page to a canonical restore key must remove the
  checkpoint-owned alias. Two cache references to one page would otherwise keep
  its reference count above the eviction threshold indefinitely.
- Published waiting checkpoints may be evicted. A fresh restore failure clears
  the old proof and permits one strictly shorter attempt, then ordinary fallback.
  Repeated reports for the same failed read do not reset or consume retries.
- Cancellation during publication cannot resurrect an offer. Generation and
  accepted-history checks prevent old offers from serving a different attempt.
- Unknown device/native completion is not an ordinary miss. Retain owners and
  refuse unsafe allocator teardown; do not reuse potentially DMA-active memory.
- This is process/DP-local state, not durable storage or a migration protocol.

## Scope and normal decoding

Implementation is in the two LMCache repositories. `local_checkpoint.py` owns
offers, acquisition and CPU boundary assembly; `preemption_checkpoint.py` owns
capture and publication. The Ascend engine provides allocator/lookup/shared
transport bridges. The adapter extends only checkpoint controls, resumed lookup,
cold-load dispatch and existing completion/preemption branches.

No model/token/scheduler/graph kernel was changed. The original NPU dense group
store remains unchanged. New source validation and lookup work runs on admission,
not ordinary per-token sparse decoding. Existing event-installed checkpoint
polling is removed once capture/publication jobs retire, even while evictable
offers remain. Completion cleanup executes inside the existing nonempty
finished-request loop.

The local H2D restore retains the existing guard against runtime graph capture
overlapping CPU-load stream work. It does not weaken that guard to claim faster
decoding. Transfer cost, boundary assembly, shared-cache competition and any
guard-induced delay require NPU measurement.

## Deployment and validation

Use the existing decoder-only `decode_preemption_checkpoint: true`. Periodic
decode save is not required. No additional pool size or policy knob is added.
Update both LMCache Python repositories and restart decoder workers. The native
extension built for the preceding checkpoint implementation is reused.

Focused CPU tests exercise actual control, dispatch and allocation/submission
methods with device/storage boundaries mocked. They cover partial capture,
multiple pointer columns, one fence, speculative trimming, original/remap prefix
boundaries, two-group holes, eviction races, repeated preemption, cancellation,
cache-alias retirement, mixed-source masks and ordinary dispatch.

The implementation and separate audit passed 135 focused CPU tests: 35 in
LMCache-NPU, 58 in LMCache-Ascend, 16 in vLLM and 26 in vLLM-Ascend. Parsing,
targeted Ruff checks and whitespace checks passed. Ten existing allocator,
ordinary transfer, sparse-decode and Ascend dispatch methods are AST-identical
to their committed versions. vLLM and vLLM-Ascend have no production changes.

The separate audit corrected an immutable-envelope construction error, preserved
the exact original prefix when its remap frontier is one token earlier, and
verified that cache aliases cannot retain unowned pages permanently. Ownership
of the CPU index tail is released only after the existing all-worker completion
barrier, rather than on TP0's individual transfer completion.

NPU qualification must compare restored KV/next-token output with an uninterrupted
reference; force preemption with long outputs, near-full HBM, CPU contention,
multiple TP ranks, cancellation and missing original-prefix storage. Measure
capture stall, saved/required frontier, restore latency, native recovery passes,
graph re-entry, MTP acceptance and aggregate throughput.

With `PD_SERVING_PERF=1`, inspect `decoder_preemption_checkpoint` and
`checkpoint_capacity_reclaim`. `ready` is an evictable local offer, not proof of
restoration. Confirm the subsequent `scheduler_lookup` paired frontier,
`worker_load_complete`, and graph route. `local_publish_ms` replaces the former
tail persistence timing; existing lookup/load diagnostics retain their meaning.
