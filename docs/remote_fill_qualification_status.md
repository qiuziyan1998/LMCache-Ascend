# Direct Remote LMCache qualification status

This file records implementation status for the evidence-gated qualification
plan. It is not a hardware qualification result. The serving feature remains
default-off and the NEGOTIATE/OPEN/RESERVE/ARM/REPORT/FINISH/ABORT/STATUS
protocol is unchanged.

## Implemented

- Exact four-repository manifest generation, dirty-tree rejection, runtime,
  artifact, topology, network, NUMA, clock, layout, model-bundle, and MTP
  inventory validation.
- V1--V8 evidence records. V1, V2, and V6 require passing hardware evidence
  before C1/release exit; unit evidence alone cannot close them.
- A deployment-adapter H0-A--H0-G orchestrator with exact byte/canary,
  partial-page, visibility/reuse, soak, production-window bandwidth, and
  dual-sink report validation. It cannot turn a mock into hardware evidence.
- Strict C1 scenario, integrity-comparison, TP8, DP2, MTP, eviction, and
  cross-DP reuse-decision report contracts plus a fixed production-adapter
  runner.
- A/B/C streaming client TTFT runner with warmups completed before measurement,
  canonical workload identity, exact manifest binding, paired comparisons,
  median/P95/MAD/bootstrap confidence intervals, and request throughput.
- Boot-scoped clock-domain and wall-clock fields across LMCache, vLLM,
  vLLM-Ascend, and the proxy; the proxy returns its trace ID to the client.
- Clock-safe stage tracing. Cross-host monotonic subtraction is rejected.
  Prefiller chunk timing, native overlap/tail, decoder cache-ready stages, and
  client TTFT remain separate to avoid double counting.
- Separate ARM, producer-event fence, source registration, native-slot wait,
  native execution, REPORT, FINISH control, decoder commit, and LocalCPU commit
  lock wait/hold timing. All are CPU-side observations under cold-perf logging;
  no device readback or new synchronization is introduced.
- Window-owned direct-source plans are retained across chunked-prefill callbacks
  until `_finish_save_batch` adopts the final request-matched producer fence.
  An early last-window callback cannot finalize RemoteFill, and a missing final
  fence still fails closed to mandatory persistence. Structured retain/release/
  reject counts distinguish ordering from absent-source failures without
  requiring a full-prefix slot mapping.
- P1 decision gates for disabled/fallback overhead, client TTFT confidence,
  decoder cache-ready reduction/interference, publication, retention review,
  full-window overlap, and conditional O1 eligibility. O2 always remains off
  until post-O1 hardware evidence exists.
- A staged P1 campaign runner with explicit cold-trial isolation evidence, exact A/B/C
  path contracts, and alternated order; Tier 1 is run before separately
  selected later tiers.
- A fail-closed paired-restart supervisor state machine that rejects incomplete
  process-group stop or reused engine/session/cache identities.
- Exact timeout selection and lost-reply replay tests for all eight control
  operations, plus shared physical completion for single, partial, and batched
  persistent writes.

## Hardware/deployment gates still pending

- A production H0 adapter beside the real launcher, because only that process
  can access the live P/D GlobalTE sessions, production source plans, hidden D
  allocations, and destination bytes. H0 must run on the isolated real hosts.
- Real configured AscendMultiConnector construction, full producer-stream
  fence coverage, TP8 passive-rank failure agreement, DP2 routing, MTP first
  resume, registration soak, and active-decoder interference campaigns.
- A production adapter binding the paired-restart state machine to the actual
  proxy and process supervisor. Until that adapter is qualified, hardware
  fault injection and production fault qualification are prohibited.
  Fault-free H0/C1 may stop and use the documented manual paired restart
  sequence.
- A clean committed four-repository snapshot and exact runtime inventory. A
  dirty development manifest may be generated only with `--allow-dirty` and is
  explicitly nonqualifying.

## Deliberately not implemented

- O1 unarmed lookahead, `ADVANCE`, and two simultaneous native writes. Their
  thresholds require current-protocol hardware evidence first.
- Any relaxation of ambiguous ARM/REPORT/native-timeout restart behavior.
- Any synthetic or mock claim of H0, C1, or production readiness.
