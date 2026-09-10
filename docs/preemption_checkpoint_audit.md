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

Final focused result: **67 passed** (vLLM-Ascend 26, LMCache-NPU 17,
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

## Remaining deployment qualification

This Windows workspace cannot compile CANN/NPU extensions or execute HCCL. CPU
tests do not prove hardware deadlock freedom, output/logit equivalence or negligible
throughput impact. Rebuild the native extension and run the multi-DP forced-
preemption matrix in `preemption_checkpoint.md`, first with phase 2 disabled,
then enabled. Test idle peers, concurrent recoveries, MTP rejection, chunk edges,
immediate block reuse and storage failure. Keep existing source-ownership fences;
do not bypass them to improve a benchmark.
