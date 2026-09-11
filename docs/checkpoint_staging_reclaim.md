# Eviction-assisted checkpoint staging

The 2026-09-11 run entered checkpoint capture but all eight reported attempts
failed with `Checkpoint CPU staging allocation refused`. The previous lazy pool
could not reclaim unused cache storage. This change adds one bounded reclamation
attempt without reserving peak checkpoint capacity at startup.

## Admission and ordering

1. Validate both resident ranges/block maps and prepare their CPU slot lists.
2. Reuse idle staging or try ordinary eviction-free CPU allocation for both groups.
3. If allocation fails, compute aligned physical bytes only for the groups that
   still need allocations. Exclude already allocated and reusable buffers.
4. Reclaim once, outside the private staging-buffer lock. Retry allocation once.
   Concurrent allocation or fragmentation can still make this fail safely.
5. Only after both groups own their CPU destinations, prepare/enqueue D2H and
   retain the existing final completion fence before HBM reuse.

The old partial Group-0 prefix still uses the existing exact LocalCPU lookup,
retaining a hit before further allocation. A miss uses Mooncake. Its background
temporary allocation also allows one bounded reclaim/retry after HBM release.
Pinned or borrowed prefix pages are ineligible for reclamation. If the old source
is unavailable, persistence fails rather than assembling incomplete KV.

## Bounded reclamation

`LocalCPUBackend.reclaim_evictable_capacity` accepts an optional
`max_scan_entries`; existing callers omitting it retain their behavior. Checkpoint
admission supplies 4096 and a `checkpoint_capacity_reclaim` diagnostic cause.

The bounded mode:

- Uses the existing default LRU order and page eviction bookkeeping.
- Does not wait for `cpu_lock`; contention produces refusal.
- Limits candidate traversal and inspected entries. Expanded legacy siblings
  count against the budget, even when stored far apart in layer-major order.
- Requires every legacy sibling to be evictable; a pinned or externally referenced
  layer protects the group. Private staging/transfer owners never become victims.
- Selects sufficient capacity before removing anything. An insufficient bounded
  candidate set leaves the cache unchanged.
- Releases removed objects outside the cache lock and checks actual free capacity.
  Reclaimed bytes are not a reservation and do not guarantee a contiguous allocation.

Only the default LRU policy is qualified for this bounded selection mode. Other
policies retain normal allocation and safe refusal; their victim ordering is not
silently replaced. RemoteFill's existing unbounded-candidate mode is unchanged.
No new serving configuration knob or connector wire field is added.

This bounds work and retries, not absolute wall-clock latency. Allocator operations,
reference retirement and Python scheduling can still add latency. No CPU sleep,
device synchronization or I/O is introduced by reclamation itself.

## Memory retention and normal decoding

Active slabs remain owned until capture/persistence reaches a known terminal state.
Cancellation and unknown-native-completion quarantine retain the same fences.
Retirement retains at most one reusable idle slab per group; excess slabs are
returned to the original allocator instead of retaining peak concurrency.

Each slab still uses the existing capacity of one resolved decode-save window
plus one LMCache chunk, across all layers. There is no eager reservation, separate
CPU allocator, new model-KV HBM pool or normal-decode scan. The checkpoint reclaim
calls occur only after an actual staging allocation failure.

## Validation and deployment

Focused tests cover pinned/borrowed pages, legacy sibling protection and ordering,
the candidate limit, contention, insufficient capacity, allocation races,
per-group alignment, both-group admission, one retry, no device work on failure,
old-prefix fallback and excess-buffer retirement. CPU tests mock storage/device
boundaries; NPU latency and checkpoint success rate require deployment testing.

Review result: 127 focused CPU tests pass across the four fix worktrees
(vLLM 16, vLLM-Ascend 26, LMCache-NPU 30, LMCache-Ascend 55). The review added
a failing-before case for layer-major legacy keys whose siblings lie far apart;
the final scan charges those siblings against its budget without requiring
contiguous LRU positions. Byte plans exclude reusable groups and round each
physical allocation independently. Invalid capture sources allocate/evict nothing.

Update LMCache-NPU and LMCache-Ascend together. Existing
`decode_preemption_checkpoint: true` enables this behavior. No native rebuild or
vLLM/vLLM-Ascend change is needed for this follow-up.

With `PD_SERVING_PERF=1`, inspect capacity reclamation and checkpoint results:

```bash
grep -nE 'checkpoint_capacity_reclaim|decoder_preemption_checkpoint' D_dp*_port*.log | tail -n 60
```

A `reclaimed` outcome should be followed by capture/persistence evidence, not
treated as proof that checkpoint storage or restoration completed.
