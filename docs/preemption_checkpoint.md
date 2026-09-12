# Decoder recovery on the 96.8 ms baseline

This implementation belongs to `fix/decoder-resume-dp-sync`. It does not include
the unrelated optimizations from the integration branch. CPU contract tests pass;
native compilation, multi-DP collective compatibility and NPU performance still
require qualification on the deployment.

## Phase 1: bounded MC2 recovery

The Ascend recompute scheduler limits qualified MoE/EP decoder batches to the
worker's MC2 token capacity (32 for the baseline TP4, 16-sequence, one-MTP-token
configuration). This bounds model computation, not the number of KV tokens loaded.
Both running and resumed requests share the budget. Original prompt lengths,
lookahead allocation and intermediate-prefill output suppression are preserved.

The no-sync decision requires an actual bounded scheduler. Unsupported parallel
configurations and draft paths that expand target inputs retain DP metadata
agreement. The qualified speculative path is padded MTP without parallel drafting.
The Ascend worker invokes the
connector preemption hook before block zeroing, state replacement and new loads.

No new launch option is needed for phase 1. It applies to the existing
`recompute_scheduler_enable` decoder setup. It does not make every recovery pass
graph-compatible and can require multiple native passes when KV is missing.

## Phase 2: generated KV checkpoints

Add this **only to the decoder LMCache YAML**:

```yaml
decode_preemption_checkpoint: true
```

It defaults to false. The qualified path uses the existing `kv_both` decoder,
`pd_role: receiver`, `store_async: true`, two-group MLA/DSA shared CPU cache,
`enable_dsa_cold_compact_load: true`, and
`dsa_group1_load_mode: persistent_direct_hbm`. PP, PCP and DCP must be one.
The initial storage implementation uses the existing replicated MLA first-rank
writer for both groups; other writer layouts cannot publish checkpoints.
Use `LMCacheAscendConnectorV1Dynamic`, directly or inside `AscendMultiConnector`.
The dynamic wrapper delegates preemption explicitly; other composite children
receive their ordinary hook without early binding of next-step metadata.
The implementation factory is resolved at connector construction, so importing
the dynamic wrapper before Ascend installs its patch cannot retain the base
implementation accidentally.

These are the **baseline** option names. Do not substitute configuration names
introduced by the later performance branch.

Rebuild/reinstall the LMCache-Ascend native extension with the deployment's usual
build procedure. The new prepared group binding is required, and startup rejects
an older extension when checkpointing is enabled. Deploy matching changes from
all four repositories, including vLLM's internal checkpoint proof field.
Validation runs before manager/service initialization, so an incompatible
checkpoint configuration cannot silently become degraded recomputation.

### Ordering and memory ownership

1. Before scheduler preemption, copy both original block tables and the generation.
2. Before worker block reuse, capture resident generated KV with the existing dense
   group D2H kernels. Both groups enqueue before the final capture fence.
3. After this fence, HBM can be reused. No CPU checkpoint is yet lookup-authoritative.
4. After older async outputs are consumed, seal the exact accepted/computed end;
   the last sampled token and rejected speculative positions are excluded.
5. Seal the captured prefix into ordinary LocalCPU pages and publish an
   evictable, request/generation-scoped offer. Generated KV is not written to
   Mooncake. The original prompt retains its existing persistent source.
6. Resume probes the persistent prompt plus the available two-group local tail.
   After HBM admission, the worker revalidates and owns the local pages, assembles
   boundary pages only when necessary, and reuses the existing cold loaders.
7. A full restore with one real token remaining can use ordinary MTP graph
   admission. Partial coverage retains native recovery within the MC2 bound.

The local policy is described in [local checkpoint design and audit](local_preemption_checkpoints.md).
No dedicated tail pool is reserved. Capture allocates actual fragments no larger
than a chunk, reclaims unused cache entries once, and retains the completed
paired prefix if a later allocation fails. It may select one smaller final
fragment. The original fixed decode-save-window rejection is removed, so long
outputs are considered even with periodic decode save disabled.

Capture prepares all admitted chunks together: one native submission per group
and one final fence. The existing prepared native binding supports the chunk
pointer matrix, so this follow-up requires no C++ rebuild beyond the earlier
checkpoint extension. Deploy matching Python code in all four repositories for idle release-control dispatch.

Waiting offers hold keys, not pins. Group-0 pages adopted as active sparse-decode
sources remain protected by the running request. Group-1 CPU ownership retires
on the resumed dispatch after all-worker receive completion. No active source
is forcibly evicted. The existing cold CPU-load guard against concurrent runtime
graph capture remains enabled for checkpoint restores.

CPU allocation/refusal, stale pages or read failure preserve a safe fallback.
A failed restore can retry once at a strictly shorter chunk boundary; repeated
invalid-block reports cannot consume that allowance twice. Thereafter the request
uses ordinary prefix recovery. Unknown DMA completion retains source owners and
refuses unsafe allocator teardown.
Control-only batches can process seal, cancel and completion messages.
Checkpoint control envelopes preserve the ordinary cold-load trigger, including
when a new prefix load and a checkpoint message share a no-forward batch.
The scheduler also bounds waiting for a missing acknowledgement with
`blocking_timeout_secs`. A failed restore invalidates the checkpoint proof and
returns to the original prefix/recompute path. Old generations cannot seal using
new history, and stale replies cannot clear a newer lookup. Explicit request
`lmcache.skip_save` and engine freeze remain authoritative.

See [bounded staging reclamation](checkpoint_staging_reclaim.md) for the candidate
limit, cache-lock behavior, legacy-layer ownership protection and diagnostics.

Known terminal errors drop exception tracebacks so GC-disabled deployments do
not retain failed jobs. Quarantined native owners remain visible through cancel
and shutdown. A failed preemption hook latches the model runner against already
queued work before it can zero or reuse blocks. Failure replaces the execution
entry point; ordinary steps do not poll a fatal-state flag.

Ordinary group stores retain their existing blocking publication behavior. The
checkpoint path does not use their per-layer publication generator. Pressure-based
speculative pre-copy is intentionally not enabled; measure the exposed stall first.
The prepared native binding releases the GIL after converting Python arguments;
the capture completion fence still must finish before HBM reuse.

With `PD_SERVING_PERF` enabled, `decoder_preemption_checkpoint` reports generation,
status, end, local_publish_ms and refusal reason without reading device tensors.

### Ordinary decode path

The scheduler notifies checkpoint support from its existing victim-selection
branch, before freeing either block table. Only this event or a checkpoint reply
arms a one-shot connector-metadata builder. That builder restores the actual
derived class method after emission; pending/ready checkpoint records do not
cause a scan on every scheduler iteration. Ordinary scheduler and LMCache
metadata keep their baseline fields. Checkpoint controls use a metadata subclass.

The worker activates checkpoint seal/cancel/poll handling after an actual capture
arrives. It remains active while that preemption owns work, including control-only
batches, and restores ordinary methods after retirement. Weak receivers avoid
adding ownership cycles when cyclic GC is disabled. Merely enabling the option
does not cause checkpoint polling. Completed cold loads are promoted only in
the existing resumed-request branch, not checked for every running request.

MC2 eligibility is computed at startup from the fixed target/draft configurations
and allocation. The effective scheduler budget is set once. Ordinary steps use
the existing budget assignment and assertion rather than an additional min/check.
Graph/MTP metadata retains its ordinary structure: a cold resume alone constructs
a boolean tuple carrying verified frontiers, preserved by existing slicing/copying.

The ordinary dense group store, Ascend load entry, ordinary worker metadata
builder, finished-store routine, attention-metadata builder and common attention
metadata class were compared with the baseline and have identical syntax trees.
The draft metadata propagation file has no production diff.

This is not a claim of literally zero added Python operations across all events:
the Ascend runner restores native vLLM's preemption-ID test before block reuse,
and nonempty worker-result aggregation/dispatch can inspect checkpoint results.
Neither performs checkpoint I/O on an ordinary decode step. A bounded decoder
also intentionally changes admission/recomputation when a batch would exceed
MC2 capacity. Exact throughput equivalence requires an NPU comparison; no added
device operation, synchronization or checkpoint poll is expected in steady decode.

## Qualification

Run the CPU tests independently of the repository NPU bootstraps:

```bash
# vllm-ascend
python -m pytest tests/standalone -q
# LMCache-NPU and LMCache-Ascend (run in each repository)
python -m pytest --confcutdir=tests/standalone tests/standalone -q
```

These tests execute production control, route and submission functions with mocked
device/storage boundaries. They do not certify the native kernel or HCCL behavior.

On NPU, first test phase 1 with checkpointing disabled. Force preemption on one DP
while peers decode in graphs, then with idle peers and simultaneous preemptions.
Exercise padded counts around the actual MC2 bound and cache-load failures.

Then enable phase 2. Cover chunk ends K-1/K/K+1, MTP rejection, queued async outputs,
immediate block reuse, repeated preemption, storage eviction, one-group failure,
and cancellation during persistence and restore. Compare restored KV/next-token
logits and output accounting with an uninterrupted reference.

Measure warm throughput/TPOT, exposed capture wait, persistence and restore times,
checkpoint success rate, native recovery passes and MTP acceptance separately.
Use repeated identical workloads. Negligible amortized throughput loss is a target,
not a result established by the CPU tests.
