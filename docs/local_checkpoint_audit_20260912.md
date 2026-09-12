# Local checkpoint audit — 2026-09-12

Audited `56830458` (LMCache-NPU) and `c90622d1` (LMCache-Ascend) on
`fix/decoder-resume-dp-sync`. Changes remain isolated from the performance branch.
The audit used production Python methods with controlled storage/device boundaries;
it does not certify NPU kernels, HCCL behavior or throughput on hardware.

## Confirmed defects and corrections

| Defect | Correction and regression evidence |
| --- | --- |
| A persistent Group-1 prefix read can fail on one TP rank while peers enter the shared CPU tail collective. | All ranks report prefix-stage success before entering the tail stage. Four-rank asymmetric-failure tests verify that no rank enters the tail after a failed prefix. |
| A new checkpoint reduction can interleave with unrelated broadcasts or an existing RemoteFill reduction on the same TP CPU group. | An ordered marker runs the reduction inside the existing receive-active interval. Once armed by actual preemption, RemoteFill materialization uses that same order. Tests deliberately reverse local arrival order. |
| The new agreement could time out earlier than the existing bounded native read/drain operation. | It waits for that operation's terminal report; it does not introduce a second, shorter timeout or extend native deadlines. |
| TP0's normalized-source references were released after its own load, before all peer readers necessarily finished. | Restore-attempt references survive cancellation/local failure until the scheduler's all-worker receive acknowledgement. Generation-scoped release controls retire them; stale/duplicate releases cannot affect another attempt. |
| A release acknowledgement could remain queued when the engine became idle. | A genuine idle connector-control hook schedules a no-forward step. Active scheduling short-circuits before that hook, and no fake request or finished-request ID is created. |
| LocalCPU admission can retain a compatible existing canonical page rather than the submitted duplicate. Restore previously retained only the duplicate. | Acquire and validate the actual installed canonical pages before handoff. A test using the production LocalCPU admission method verifies that the existing source is not evictable during restore. |
| A token database omitting the normalized partial tail could return an incomplete successful handoff. | Validate contiguous normalized coverage through the exact requested end before cache admission. |
| A retry calculated from the original offer could repeat an already-shortened failed frontier. | Calculate the strictly shorter retry from the actual attempted frontier; retain the existing one-retry limit. |
| A duplicate partial-capture notification reported the original requested end. | Report the actual captured end from the existing job. |
| An unknown boundary-page DMA completion quarantined CPU memory but did not latch the decoder's existing restart guard. | Use the same fatal restart latch as unknown Group-1 DMA, preserving owners and preventing ordinary recovery from treating it as a normal miss. |

## Scope and invariants reviewed

- **Two groups and source selection:** persistent original prefix, local generated
  tail, exact boundary assembly and speculative clipping remain distinct. Both
  groups must cover the usable frontier. Normal lookup rules are unchanged.
- **Active ownership:** checkpoint offers remain evictable while waiting. Actual
  restore sources and running sparse Group-0 sources cannot be reclaimed. The
  added restore reference is held through all-worker completion, not indefinitely
  for every waiting checkpoint.
- **HBM admission:** vLLM still allocates fresh destination blocks and retains them
  through the existing receive/send lifecycle. The audit does not force graph
  admission, alter MC2 capacity or invent additional model tokens.
- **Ordering:** the prefix agreement is TP-local and runs in the background load
  stage. Marker reception is serialized with ordinary shared envelopes; the main
  decode loop does not gain DP metadata agreement.
- **RemoteFill overlap:** its existing materialization reduction retains its result
  and diagnostic event. After checkpoint transport is armed, its reduction cannot
  overtake or be overtaken by a checkpoint collective. Early Group-1 reservation,
  placement policy, prefiller async storage and native transfers are not changed.
- **Failure and cancellation:** restore releases are scheduled even after the
  request's ordinary state was removed. Before acknowledgement, cancel cannot
  free sources used by a peer. Unknown DMA remains fatal, and shutdown refuses
  to free unacknowledged restore sources.
- **Idle cleanup:** `has_requests` retains its existing active short circuit.
  Only when ordinary work is absent does it query the connector control queue.
  The dynamic LMCache connector and Ascend multi-connector delegate this check.
- **GC disabled:** the event-armed transport captures a weak engine receiver.
  Request-owned references are explicitly released, and tests verify that the
  transport wrapper does not create an engine ownership cycle.
- **Derived dispatch:** tests execute the deployed Ascend/dynamic/multi-connector
  declarations, including release-only frames and restoration of idle methods.

## Performance interpretation

The fixes prevent deadlock, avoid repeating a failed restore frontier, and keep
valid cache sources from being evicted prematurely. They do not prove a TTFT or
throughput improvement on NPU.

There is real recovery-side work: a checkpoint-stage TP agreement and marker,
canonical-source reference acquisition, and acknowledgement delivery. Once the
transport is armed, ordinary shared-metadata reception recognizes the marker;
RemoteFill materialization uses an ordered marker for its existing reduction.
This work is not added to every model-token step. Prepared sparse decoding,
ordinary dense group stores, NPU kernels and graph gates remain unchanged.

Deploy the matching Python revisions in **all four repositories** because idle
control dispatch crosses the scheduler and connector wrappers. No new knob or
C++ rebuild is required beyond the already-built checkpoint extension.

Hardware qualification must cover an asymmetric prefix-read failure, delayed TP
readers, overlapping RemoteFill and checkpoint activity, cancellation with no
remaining requests, near-full CPU/HBM capacity and GC disabled. Measure graph
re-entry and unaffected requests' TPOT, in addition to checkpoint success.

## Validation completed

154 focused CPU tests passed: LMCache-NPU 37, LMCache-Ascend 72, vLLM 19 and
vLLM-Ascend 26. Parsing, targeted Ruff and diff checks passed. Fifteen existing
scheduling, graph-route, allocation and transfer methods are AST-identical to
their committed versions. The active scheduling test fails if connector-control
readiness is queried while ordinary requests exist.

The concurrency tests use actual mailbox and agreement methods with a controlled
two-rank transport, plus four-rank asymmetric prefix outcomes. They cover a marker
received by an unrelated thread, reverse RemoteFill/checkpoint arrival order,
failure propagation, and the external read's existing completion deadline. These
are reproducible Python contract tests, not substitutes for NPU qualification.

## Follow-up: decoder errors at 11:58 and 12:00

The 11:58 trace reaches Group-0 materialization, then attempts 158 legacy
per-layer reads (two chunks across 79 layers) with zero bytes read. Production
key classes reproduce a checkpoint handoff defect: normalization admitted merged
pages under `LayerCacheEngineKey(layer_id=0)`, whereas page readers request the
unequal, layer-independent `CacheEngineKey`. Checkpoint publication and original
boundary lookup now use the same physical-page key contract as ordinary loaders.
The earlier simplified test key returned itself from `split_layers`, hiding this
distinction; the new regression tests execute the production key classes.

A second reproducer covers a remote original prefix followed by a local-only
checkpoint tail. The merged-page resolver previously sent that tail to the
legacy remote suffix path after fetching the remote prefix. It now acquires any
following LocalCPU page prefix before legacy fallback. Existing reference and
pin cleanup applies to both sources. Complete local and complete remote page
hits retain their original lookup counts; warm prepared decoding is untouched.

The 12:00 trace is different: normalization fails to allocate a boundary/cropped
page, including its single post-reclaim retry. Capture success does not reserve
this additional assembly workspace. Correcting the original-prefix lookup avoids
a redundant staging allocation when that exact page is already cached, but the
trace does not establish whether the failed assembly allocation lacked total
space or contiguous space. No change claims to eliminate genuine CPU capacity
refusals. A failure test verifies bounded retry, preserved source bytes and
reference cleanup without reclaiming active sources.

`tests/standalone/test_checkpoint_page_keys.py` covers both KV groups, full and
partial pages, mixed local/remote placement, unavailable tails and allocation
refusal. Device operations remain controlled test boundaries; NPU qualification
is still required.

## Related-path follow-up

- Repeated preemption copied layer-specific Group-0 keys from
  `WorkerRetrieveState.cached_keys[0]` into physical-page descriptors. Convert
  these to their layer-independent keys during capture. A two-generation test
  using production key classes previously lost the generated prefix; it now
  verifies both groups' restored bytes, including the cropped tail.
- The base LMCache merged-page resolver had the same remote-prefix/local-tail
  assumption as the Ascend override. Apply the same local-tail acquisition to
  both and run the same source-selection and ownership tests against both
  implementations. This does not change ordinary lookup authority or persist
  generated tokens.
- Restore normalization rebuilt boundaries before checking whether an exact
  canonical page already existed. Reuse that page through the existing checked
  checkpoint lookup; defer the original partial-page read until assembly is
  needed. The reproducer restores successfully with allocation unavailable and
  forbids an unnecessary original-boundary read. Genuine allocation failure
  retains its bounded reclaim/retry and cleanup behavior.

The affected standalone suites pass 136 tests (LMCache-NPU 37,
LMCache-Ascend 99). Changes remain in those two repositories. vLLM scheduling,
vLLM-Ascend graph routing, ordinary adapter dispatch, prepared sparse decoding
and direct-HBM transfer methods are unchanged. No per-token check, collective,
device fence or dedicated memory pool is added.

## Scheduler ownership migration

Checkpoint proof invalidation and idle release dispatch now live in
`vllm_ascend/core/recompute_scheduler.py`. `RecomputeScheduler` overrides
`_preempt_request` and `has_requests`; the existing `AsyncRecomputeScheduler`
MRO inherits both. The request proof is initialized lazily during preemption,
before the base implementation releases blocks. No `Request` subclass, global
monkey patch or per-request initialization hook is needed.

vLLM's `SchedulerInterface` is restored to `dsa-two-groups`. Its remaining
production delta is the generic receive/send lifetime correction: 18 added and
5 removed lines across `scheduler.py` and `request.py`. The corresponding
16 lifetime tests stay in vLLM. The idle-control tests move to vLLM-Ascend and
cover both scheduler classes, active/finished short-circuiting, actual connector
delegation, first and repeated preemption, forced prefix reset and invalid
preemption. The vLLM-Ascend standalone suite passes 46 tests.

This migration adds no model-step callback or synchronization. The checkpoint
feature continues to require the Ascend recompute scheduler; other schedulers
retain vLLM's generic transfer-lifetime behavior. The earlier sections describe
the original audit state; this section supersedes their placement of the idle
hook in the base scheduler.
