# Decoder recovery audit

Scope: the implementation in the four `fix/decoder-resume-dp-sync`
worktrees under `resume-fix-20260910`. The `perf/prefill-direct-minimal` work was
not read, edited, merged or cherry-picked.

## Confirmed defects corrected

| Area | Reproducer or code evidence | Correction |
| --- | --- | --- |
| Dynamic connector | Its inherited base preemption hook did nothing | Explicit implementation delegation and wrapper-chain regression test |
| Composite metadata | CPU-offload bind can enqueue stores; pre-binding all children invokes it twice | Scope checkpoint metadata to the checkpoint child; preserve plain hooks elsewhere |
| Stale replies | A rejected old result still cleared the current lookup | Invalidate lookup only on a matching, nonterminal state transition |
| Old attempts | A captured old generation could seal with a newer request history | Cancel mismatched generations before sealing |
| Restore failure | Invalid destination blocks left the checkpoint ready | Clear its readiness/proof and revert to ordinary prefix recovery |
| Missing acknowledgement | Only worker jobs had deadlines, so a missing capture could park indefinitely | Independent scheduler deadline and cancellation command |
| GC disabled | Failed future/traceback retained the job after retirement | Clear terminal failure tracebacks and detach completed futures |
| Quarantine | Cancelling a failed capture with no future removed its quarantine entry | Retain quarantined owners and make close refuse unsafe teardown |
| Queued execution | Another queued call could proceed after a failed preemption hook | Latch the runner before further block reuse |
| Shared cache generation | Completed cold-resume exemption bypassed existing shared-generation checks | Continue ordinary source validation after accepting the load generation |
| CPU allocation refusal | Group 0 metadata was uploaded before discovering Group 1 OOM | Admit both groups' CPU resources before device preparation |
| Storage policy | Checkpoints bypassed per-request skip-save and freeze | Respect both at admission/persistence boundaries |
| Expanding drafts | Production draft expansion turns a 32-token target batch into 33 tokens | Keep agreement for unqualified draft modes; validate actual extra-slot behavior |
| Startup | Manager catches engine post-init exceptions and degrades silently | Validate checkpoint configuration, wrapper and native binding before manager creation |
| Import order | A dynamic wrapper imported before the Ascend patch retained the base factory; CPU reproducer failed | Resolve the patched implementation at construction, with no decode-time work |
| Idle checkpoint handling | Metadata/control helpers scanned or polled with no active preemption | Activate control emission on scheduler events and worker handling on capture; restore original dispatch on retirement |
| Ordinary D2H | Shared checkpoint preparation introduced branches/helper calls into ordinary stores | Restore the original store method; prepare the bounded single-fragment checkpoint separately |
| Ordinary attention metadata | Extra frontier fields were copied/checked for every batch | Carry the frontier proof on cold-resume tuples only; preserve existing ordinary constructors and MTP copying |

Tests execute production control methods, route classification, dynamic wrapper
delegation and transfer orchestration with CPU storage/native boundaries mocked.
The audit includes payload plane/layer ordering, rejection-tail cropping,
generation transitions, cancellation, missing acknowledgements, GC-disabled
retirement, preserved ordinary-store behavior and no-sync submission boundaries.

First committed audit result: **67 passed** (vLLM-Ascend 26, LMCache-NPU 17,
LMCache-Ascend 24). Changed Python sources also passed parsing and static
undefined-name/syntax checks; all four working diffs passed whitespace checks.
These results describe the audit completed before committing the implementation.

The entry/idle audit additionally exercises actual Ascend dynamic-wrapper and
composite declarations, the async scheduler's schedule inheritance, capture before
state cleanup, both TP result aggregation orders, event-only seal emission,
post-retirement dispatch, and GC-disabled owner release. Native operations remain
mocked. Six ordinary methods/classes were compared structurally against baseline:
`batched_from_gpu_group`, `start_load_kv`, `build_connector_worker_meta`,
`get_finished_stores`, `_build_attention_metadata`, `AscendCommonAttentionMetadata`.
All six are identical. See the ordinary-path section of `preemption_checkpoint.md`
for the remaining preemption detection/control-result checks and the distinction
between unchanged steady decoding and intentionally bounded recovery/admission.

## Follow-up audit of the committed fix

Reviewed starting commits: vLLM `027d03361823`, vLLM-Ascend `14b53da19291`,
LMCache-NPU `30cb7ca7e10b`, LMCache-Ascend `9ae3b8f7c1df`.

The implementation follows the subsequently agreed two-phase plan recorded in
`preemption_checkpoint.md`: bounded MC2 recovery, then optional accepted-KV
checkpointing. The earlier external `RESUME_DP_COMMUNICATION_FIX_DESIGN.md`
predates checkpointing and also describes the CPU-agreement alternative. Its
recommendation to qualify the original 4096-token budget first is not the policy
implemented here. The bounded effective decoder budget is intentional and documented;
the configured buffer sizes and prefiller budget are not reduced.

### Concrete corrections

1. **Overlapping checkpoint controls and cold loads.** The one-shot metadata
   wrapper copied only `requests`, dropping `dsa_cold_compact_load_pending` from
   the ordinary builder. `_start_load_kv` needs that flag to submit cold loading
   before returning when attention metadata is absent. Preserve the ordinary
   metadata instance fields in the checkpoint envelope. The regression test
   failed before the change; it now also tests pickle transport and executes the
   actual no-forward load prefix to verify submission.
2. **Incomplete storage-key coverage.** With `save_unfull_chunk=false`, token
   processing can omit the final partial chunk. Persistence nevertheless reported
   the complete seal length as ready. Validate contiguous coverage through the
   exact accepted end for each group before submitting writes. Tests reproduced
   incorrect readiness with both a preceding full chunk and an entirely partial
   suffix. Both now fail safely without submitting an incomplete checkpoint.
3. **GIL during prepared native capture.** Python sequence conversion must hold
   the GIL, but the following native dense-group launch contains only C++ tensor
   metadata and an OpCommand with C++ kernel handlers. Release the GIL for that
   launch, reacquiring it before Python argument cleanup. This is a source-level
   correction to the new binding only; native compilation and the latency benefit
   cannot be tested in this Windows environment.

### Cross-path review matrix

| Area | Code checked and resulting contract |
| --- | --- |
| Derived classes | AsyncRecomputeScheduler inherits the bounded schedule. AscendMultiConnector delegates to the deployed Ascend dynamic wrapper; construction resolves the patched implementation. Capture precedes worker state drop and block zeroing. |
| Async ordering | Scheduler snapshots precede free; worker capture completes before HBM reuse. Captured acknowledgement travels behind older executor results, and only then can accepted token IDs seal the snapshot. Rejected/speculative and final uncomputed token KV are excluded. |
| Two-group layout | Separate block maps and explicit group layouts are retained. Native dense capture writes the existing latent/index formats. Fragment vectors preserve layer/plane/token order and require continuous coverage. LayerPageMemoryObj requires valid_tokens to match its physical layer shape. |
| LocalCPU prefix source | get_checkpoint_prefix uses the existing retained page lookup and validates valid count, layers, format and dtype. An exact hit keeps its reference through persistence. Missing/incompatible pages use a separate registered buffer and the external read path. New tests execute these actual methods for valid, invalid-format and missing pages. |
| Mooncake locality | StorageManager selects RemoteBackend, which delegates exact page reads/writes to Mooncake. The existing per-group preferred-segment selection is preserved through request configs. Local/remote network placement is a backend decision; a preference is not a locality guarantee. |
| Group-1 loading | Existing direct-HBM preflight, destination planner, scheduler admission slot and terminal DMA checks remain authoritative. Neither checkpoint capture nor the MC2 fix changes the reader or its TP startup agreement. |
| RemoteFill overlap | Existing RemoteFill capacity accounting, epochs, shared-cache generations, producer fences and terminal handling are unchanged. Checkpoint buffers use the same accounted CPU allocator without eviction. Checkpoint metadata now preserves concurrent cold-load dispatch. |
| Early Group-1 reservation | This baseline reserves final Group-1 HBM through scheduler allocation and then launches the existing cold loader. It does not add a new prefiller-end-triggered HBM reservation/load protocol. Integration with such a feature on another performance branch is not qualified by this audit. |
| Graph/MTP | Only a verified complete cold restore with the supported last-token/MTP layout can use the graph path. Cold proof survives existing metadata slicing/copying. A historical suffix stays native under the MC2 bound; increasing capture capacity or removing a gate is not used to hide missing KV. |
| DP communication | Startup validates the capacity and phase-aware selector. Target scheduling and graph padding fit the qualified bound; expanding draft/unsupported parallel layouts retain agreement. The existing DP engine dummy-batch path remains in place for peers without model work. Hardware deadlock freedom remains an NPU qualification item. |
| HBM pressure | No extra model-KV HBM pool is allocated. Old request blocks remain source-owned through capture completion. The preemption hook also preserves existing pending-store drains; these waits can be expensive and are not removed. |
| CPU pressure | Private reusable slabs are bounded by queue size and resolved window/chunk capacity. Allocation neither evicts nor busy-retries. Refusal gives a failed checkpoint and bounded recovery; it does not publish false readiness. |
| Threads, cancellation, GIL | Capture uses the worker thread; persistence uses one background executor. Cancellation suppresses readiness and retains in-flight DMA owners. Unknown completion is quarantined and requires restart. The prepared launch now releases the GIL; external Mooncake binary GIL behavior was not measurable here. |
| Minimal ordinary-path disturbance | These corrections only run in checkpoint control/persistence/capture. Ordinary store/load/attention routines remain baseline-equivalent. No new per-token checks, tensor readbacks, synchronization, CPU eviction loops or metadata scans were added by this follow-up. |

The checkpoint payload is bounded, but all checkpoint work is not constant-cost:
accepted history is copied/hashed, group plans are constructed, and restore may
read the full prefix. Python preparation, CPU allocator contention, Mooncake
placement/latency and the exposed capture fence can still reduce throughput during
recovery. This audit does not establish negligible preemption cost, optimal
Group-0 metadata construction, or a TTFT improvement. Those require measurements.

Focused follow-up result: **73 passed** (vLLM-Ascend 26, LMCache-NPU 18,
LMCache-Ascend 29). Existing cancellation, ownership, generation, failed-restore,
ordinary-path and MC2 policy tests remain included. No performance-branch code
was imported and no new configuration knob was introduced.

## Remaining deployment qualification

This Windows workspace cannot compile CANN/NPU extensions or execute HCCL. CPU
tests do not prove hardware deadlock freedom, output/logit equivalence or negligible
throughput impact. Rebuild the native extension and run the multi-DP forced-
preemption matrix in `preemption_checkpoint.md`, first with phase 2 disabled,
then enabled. Test idle peers, concurrent recoveries, MTP rejection, chunk edges,
immediate block reuse and storage failure. Keep existing source-ownership fences;
do not bypass them to improve a benchmark.
