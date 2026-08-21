# Direct Remote LMCache qualification

This runbook applies the evidence-gated qualification plan. It does not claim
that native direct placement has passed H0, C1, or production qualification.
The production default remains disabled, and the conservative wire protocol is
unchanged.

## Freeze one exact run

Copy `benchmark/v1/kv_transfer/remote_fill_inventory.example.json`, replace
every placeholder with observed deployment data, and generate the manifest:

```bash
python benchmark/v1/kv_transfer/remote_fill_qualification.py \
  inventory.json qualification-manifest.json
```

The command rejects a dirty repository, a missing artifact identity, invalid
topology values, or an incomplete clock/network inventory. For an explicitly
nonqualifying development snapshot, `--allow-dirty` records the dirty state.

Maintain V1--V8 evidence in a copy of
`remote_fill_evidence.example.json`. Bind it to the exact manifest with:

```bash
python benchmark/v1/kv_transfer/remote_fill_qualification.py \
  inventory.json qualification-record.json --evidence evidence.json
```

Add `--require-hardware` only for a C1-exit or release record. It rejects every
remaining `PENDING_HARDWARE` item and requires V1, V2, and V6 to carry
`FIXED_AND_PASS` hardware evidence; unit-only evidence cannot close those gates.

## H0 gate

H0 must run in an isolated P/D deployment with the exact production GlobalTE,
source registration, source-page planner, decoder LocalCPU allocator, NIC,
MTU, routing, and NUMA policy. Set
`LMCACHE_REMOTE_FILL_H0_QUALIFICATION=mooncake-sync-write-visible-v1` only for
that isolated deployment.

The hardware driver must cover H0-A through H0-G from the governing plan and
retain raw results. It must validate destination bytes inside D before any test
allocation is released and must not publish synthetic pages into ordinary
LMCache. Stop the campaign on a byte/canary mismatch, late mutation, ambiguous
native terminal state, registration overlap, or resource leak. The activation
environment variable is operator intent; it is not evidence that H0 passed.
Validate the resulting report with `validate_h0_report`; its
`qualification_manifest_sha256` must equal `payload_sha256` of the exact
generated manifest.

`remote_fill_h0.py` owns the fixed case order, partial-token matrix, manifest
binding, fail-closed validation, and cleanup. Supply a deployment adapter as a
`module:factory`; that adapter is intentionally kept beside the real launcher
because it must reach the live P/D GlobalTE sessions and D-local hidden bytes:

```bash
python benchmark/v1/kv_transfer/remote_fill_h0.py \
  --adapter production_h0:create_adapter --adapter-config h0-config.json \
  --manifest qualification-manifest.json --output h0-report.json \
  --chunk-size 256 --window-tokens 4096 --soak-iterations 100
```

The adapter contract is `H0Adapter`. A mock adapter can test orchestration but
cannot satisfy a hardware evidence record; its report must never be promoted
to `FIXED_AND_PASS` or used to activate serving.

## Client-observed TTFT

Prepare JSONL prompts with exactly `case_id` and `prompt`, then run each clean
A/B/C deployment separately:

```bash
python benchmark/v1/kv_transfer/remote_fill_workload.py \
  --endpoint http://proxy:9000/v1/completions \
  --mode C --trial-id c-128k-01 --workload-id cold-128k-v1 \
  --qualification-manifest qualification-manifest.json --cache-state cold \
  --prompts prompts.jsonl --output c-128k-01.jsonl \
  --repetitions 10 --warmups 1 --concurrency 1 --max-tokens 1
```

The runner finishes every warmup before admitting measured work and binds each
row to the canonical workload plus exact qualification manifest. The proxy
returns its internal request identity in `X-Request-Id`; the runner records it
beside authoritative client TTFT. Cold-performance events now carry
host, wall time, and a boot-scoped clock domain. Use monotonic differences only
within one clock domain and use the manifest's measured PTP/NTP offset for
cross-host ordering. Never subtract P monotonic time from D monotonic time.

Content diagnostics and device readback must be disabled for P1. Use fresh
cache namespaces or an audited complete cache reset for every cold repetition.
Do not mix warmups, first misses, and later prefix hits in one distribution.
After all three runs, compare only pairwise-identical trials:

```bash
python benchmark/v1/kv_transfer/remote_fill_compare.py \
  --mode-a a.jsonl --mode-b b.jsonl --mode-c c.jsonl \
  --output abc-comparison.json
```

The comparison reports the disabled-path TTFT gate and the client-TTFT portion
of the experimental gate. It deliberately leaves `proceed_gate_complete=false`
until decoder critical-path reduction and active-load interference evidence are
joined from stage traces.

Build a request-local stage trace without cross-host monotonic subtraction:

```bash
python benchmark/v1/kv_transfer/remote_fill_trace.py --trace REQUEST_UUID \
  --log proxy=proxy.log --log prefiller=P.log --log decoder=D.log \
  --client c-128k-01.jsonl --output c-128k-01-trace.json
```

Per-window and commit timings are explanatory children of the prefiller round
trip. The trace labels them explicitly so they are not added to TTFT twice.
Source-event fence wait and GlobalTE source-registration time are measured
separately. TP0 prefiller chunk events provide summed model-forward CPU time,
the first-to-last prefill span, full-window native overlap, and the remaining
post-prefill native tail without cross-host clock subtraction.

Before declaring C1 complete, fill
`remote_fill_c1.example.json`, or run the fixed production-topology adapter:

```bash
python benchmark/v1/kv_transfer/remote_fill_c1.py \
  --adapter production_c1:create_adapter --adapter-config c1-config.json \
  --manifest qualification-manifest.json --output c1-report.json
```

The adapter must run beside the real TP8/DP2/MTP launcher. The runner fixes the
nine-scenario order, binds the report to the manifest, always closes the
adapter, and requires explicit use of production vLLM, AscendMultiConnector,
LMCache, Mooncake, shared LocalCPU, DP2 routing, and MTP. A mock validates only
orchestration. `validate_c1_report` requires scenario-specific proof, both TP8
passive-group failure campaigns, all four DP2 mappings, bounded diagnostics,
the complete comparison matrix, a cross-DP reuse decision, and zero values for
every integrity invariant. Fill
`remote_fill_p1.example.json` and feed the final P1 aggregate to
`evaluate_p1_report`; it applies the documented disabled/fallback,
experimental, production, publication, overlap, and O1 thresholds. It always
keeps O2 disabled until a separate post-O1 hardware result exists.

For the full staged P1 matrix, provide a deployment adapter that activates the
exact A/B/C mode and completely resets cache state before each mode. Every
result must attest its path contract: A=`existing_production_path`,
B=`new_code_feature_disabled`, and C=`conservative_remote_fill`. This prevents
a partially enabled mode from being accepted as a valid comparison. Cold
evidence must also attest either a full tier clear before every measured batch
or a unique cache namespace for every measured batch; resetting only once per
mode is not a cold campaign:

```bash
python benchmark/v1/kv_transfer/remote_fill_campaign.py \
  --adapter production_p1:create_adapter --adapter-config p1-config.json \
  --manifest qualification-manifest.json --output p1-campaign.json \
  --max-supported-tokens 131072 --tier 1
```

The campaign alternates A/B/C order, requires identical workload identity,
separate warmups, at least ten measured repetitions, per-batch cold-cache
isolation, disabled content diagnostics, and raw evidence for every run. It
covers the documented Tier 1--4 progression without introducing a full
Cartesian product. Tier 1 is
the default; run `--tier 2`, then `--tier 3`, and finally targeted `--tier 4`
only after reviewing the preceding report. Multiple `--tier` options may be
combined deliberately.

## Paired restart boundary

Before hardware fault injection, the deployment supervisor must treat the
selected source engine and destination engine—including all TP/DP workers—as
one restart unit. On unresolved ARM/REPORT recovery, native hard timeout,
worker crash during DMA, TransferEngine reset, or registered-pool loss:

1. remove the P/D pair from proxy placement;
2. stop admission to both engines;
3. terminate both complete process groups;
4. wait for every process to exit before memory can be reused;
5. start both engines with new engine epochs, sessions, registrations, and
   shared-cache generation;
6. rediscover placement before restoring traffic.

Dying only D, restarting one TP worker, or reusing an old capability is unsafe.
Until the supervisor automates this sequence, fault injection and production
fault qualification remain prohibited; fault-free H0/C1 runs may use the same
sequence manually after stopping the run.

`vllm_ascend.distributed.kv_transfer.remote_fill_restart` provides the narrow
`PairedRestartAdapter` and `restart_affected_pair()` state machine. A deployment
adapter must bind those calls to the real proxy and process supervisor. The
state machine never republishes placement or restores admission unless every P
and D process stopped and source/destination epochs, sessions, and shared-cache
generation all changed. Merely importing this module is not deployment
automation; fault qualification remains prohibited until the production
supervisor adapter is exercised and its restart record retained.

```bash
python -m vllm_ascend.distributed.kv_transfer.remote_fill_restart \
  --adapter production_supervisor:create_adapter \
  --adapter-config supervisor.json --pair-id p0-d1 \
  --timeout-seconds 300 --output paired-restart.json
```

## Optimization decision

Do not implement lookahead, `ADVANCE`, or two simultaneous native writes from
this runbook. First retain the clean current-protocol P1 result. One unarmed
lookahead becomes eligible only when its measured native gap and post-prefill
tail cross the thresholds in the governing plan. `ADVANCE` is considered only
after lookahead leaves REPORT+ARM as a material TTFT contributor.
