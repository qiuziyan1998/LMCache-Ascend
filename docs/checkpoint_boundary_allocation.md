# Checkpoint boundary allocation and ownership

Implemented on `fix/decoder-resume-dp-sync`, 2026-09-14, on top of LMCache-NPU
`d17e75a5` and LMCache-Ascend `91264a5`.

## Confirmed problems

A restore can own valid captured fragments but fail to allocate the larger
canonical boundary page. The previous capacity reclaimer skipped eviction when
aggregate free bytes exceeded the request, even though the real allocator had
already refused the request because no suitable span was available.

Normalization also retained its temporary original-boundary source through the
remaining groups and device restore after its CPU assembly copy had completed.
Counted-ownership tests reproduced that unnecessary retention.

## Changes

- `LocalCPUBackend.reclaim_evictable_capacity` accepts an optional
  `allocation_failed` flag, defaulting to false. The checkpoint-specific caller
  sets it only from the existing allocation-failure paths.
- After failure, the reclaimer scans the existing bounded LRU window even when
  total free bytes suffice. It prefers one eligible merged page large enough to
  satisfy the requested allocation. Otherwise it retains a bounded candidate
  batch whose freed spans may coalesce with existing free space.
- Pinned or externally owned objects remain ineligible. Legacy multi-layer
  entries retain their existing all-sibling ownership checks. Removal is selected
  under the CPU-cache lock; allocator references are released after that lock.
- The reclaimer remains nonblocking for the bounded cache-lock attempt and never
  allocates or sleeps. The actual allocator retry remains authoritative. The
  previous one-reclaim budget in capture/normalization is unchanged; a failed
  retry still uses existing shorter-prefix recovery. `retry_required` replaces
  misleading `not_needed` diagnostics when an actual failure preceded the check.
- Once CPU boundary assembly completes, normalization releases only the temporary
  original-boundary reference it acquired. The destination is already owned at
  that point. Exceptions before completion retain the existing cleanup path.

## Sparse pointer and frontier safety

The new canonical last-chunk pages are **not** the released temporaries. They
remain in the returned restore-owner list, including the actual canonical winners
retained after cache admission. Existing worker-state construction builds and
publishes its prepared sparse pointer tables from these owned pages. Existing
all-worker completion and active Group-0 ownership remain unchanged.

This change does not advance a frontier, overwrite an old content-hash key, unpin
another request's source, modify a live pointer tensor, or release active sparse
sources. Original pages shared by another request retain that request's ownership.

Ordinary allocations do not opt into forced reclamation. The default capacity
reclaim behavior and LRU selection remain covered by the existing tests. No model,
scheduler, graph, native kernel or configuration knob changes are required.

## Validation

The initial implementation passed 359 standalone CPU tests. A separate audit
expanded coverage to 365: LMCache-NPU 52, LMCache-Ascend 251, vLLM 16, and
vLLM-Ascend 46. Focused Python error checks and whitespace checks pass.

The regression suite includes a fragmented pool exercised through production
`AddressManager` allocation/coalescing methods, with a test sorted-container
substitute: 80 total free bytes cannot initially satisfy one 60-byte allocation;
after reclaiming an idle 70-byte page, the actual allocation succeeds. Additional
tests cover a busy cache lock, bounded scans, protected pages, insufficient
capacity, legacy owners, other policies, and allocation failure after reclamation.

Lifetime tests release and overwrite the temporary original-boundary span, then
check exact replacement bytes, their pointer-table entries and retained ownership.
They also check shared original sources, failed assembly, and the existing retry
limit without publishing false success.

Production scope is 43 insertions / 10 deletions across three Python files.
No additional native rebuild is needed; deploy matching LMCache-NPU and
LMCache-Ascend Python revisions. NPU correctness under contention and actual
TTFT/throughput improvement still require hardware testing. Reclamation is bounded,
so it does not guarantee successful allocation under every fragmentation or race.

## Separate audit

No additional production defect was proved. Six additional cases verify:

- The actual Ascend byte-planning helper reaches the modified LocalCPU reclaim
  methods for both groups, including the previously skipped sufficient-total-free
  case. The dynamic Ascend/MultiConnector dispatch tests also remain passing.
- A concurrent allocation consuming reclaimed capacity cannot produce a false
  capacity-success result. Actual allocation remains the final decision.
- The candidate inspection limit holds in failed-allocation mode, including
  protected entries; removed owners are freed outside the CPU-cache lock.
- A Group-1 workspace failure after Group-0 assembly does not double-release the
  retired temporary. A subsequent shorter restore preserves the replacement
  canonical pages, their byte contents and ownership.

Method-body comparison confirms that only the two LocalCPU reclamation methods,
the Ascend reclaim helper and checkpoint normalization changed. The other 61
LocalCPU methods and 144 Ascend engine methods are unchanged. No model or
ordinary decode callback, pointer-table builder, frontier update, graph gate,
native synchronization or persistent-store route was modified by this follow-up.
