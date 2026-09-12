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
