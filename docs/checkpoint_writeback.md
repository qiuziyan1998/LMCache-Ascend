# Decoder checkpoint write-back

This optional policy protects waiting checkpoints in the existing registered
LocalCPU heap, then copies them to Mooncake when retention or allocator pressure
rises. It extends the existing decoder preemption checkpoint path. It does not
require a separate storage process or a Mooncake/native extension rebuild.

## Configuration

Keep the shared-CPU two-group layout, asynchronous stores, cold compact loading,
and the ordinary/one-token-MTP checkpoint restrictions. Mooncake must use the
existing metadata-free page-first format. A disaggregated receiver continues to
use `pd_role: receiver` and `dsa_group1_load_mode: persistent_direct_hbm`.

A standalone `kv_both` server leaves `pd_role` unset (or `null`) and can retain
the default `dsa_group1_load_mode: p2p_preferred`. The CPU-loading alternatives
`persistent_serial` and `persistent_parallel_prefetch` are also supported.
Standalone restore requires `extra_config.mooncake_layer_merged_page_objects:
true` and `extra_config.save_chunk_meta: false`, in addition to the shared-CPU
page-first settings. It does not require `enable_remote_lmcache_store` or a
RemoteFill service. Keep the existing standalone prefill store configuration.

Standalone checkpoint restore prepares both groups through the existing
shared-CPU loader, after checkpoint normalization. The receiver's direct-HBM
prompt-index load is unchanged. Standalone startup still budgets HBM for normal
prefill; checkpoint support does not enable the receiver-only compact startup
memory budget or establish standalone 1M-context capacity.

The feature is off by default. Example initial budgets for a **200 GiB CPU heap**:

```yaml
decode_preemption_checkpoint: true
decode_checkpoint_writeback: true
decode_checkpoint_retention_max_bytes: 4294967296  # 4 GiB
decode_checkpoint_free_reserve_bytes: 8589934592   # 8 GiB
```

Budgets are per TP instance/storage-owning rank, not per TP worker. Both byte
values must be positive and smaller than the actual heap. They are tuning
examples, not measured optimal values. Actual heap validation runs after storage
initialization and before shared-cache startup success is broadcast; invalid
budgets reach passive TP ranks through the existing failure envelope.
Update both LMCache Python repositories
and restart the participating decoder group; preserve its matching existing
native checkpoint extension and vLLM/Ascend checkpoint support.

## Behavior

1. Keep the existing capture and D2H fence before HBM reuse.
2. Seal only scheduler-confirmed accepted/computed tokens. Before local readiness,
   retain the actual paired CPU source owners if their unique physical bytes fit
   the budget. Preserve older nonresident latent sources across preemption.
3. Resume locally without requiring a Mooncake PUT to finish.
4. A lazy background worker samples allocator capacity while protected records
   exist. Retention pressure starts at 80% of the budget and targets 60%; low
   allocator free space triggers spilling toward the reserve. Checks occur about
   every 250 ms. Idle workers sleep until notified.
5. Spill canonical pages through the existing external-buffer API, one page per
   native call. Reuse complete pages; assemble cropped/boundary key and scale
   planes only when necessary. No cache alias mutations occur during spilling.
6. After both groups' selected coverage has been persisted successfully, release
   waiting protection. Ordinary cache eviction can now reclaim the pages when
   other owners permit it. The release is not an assertion of actual free bytes.
7. On resume after local eviction, probe exact remote keys and fetch only missing
   ranges through the background checkpoint restore path. Preserve existing TP
   agreement, paired completion, and coordinated shorter-prefix/recompute fallback.

No network wait or pressure scan is added to ordinary warm decode steps. Spill
work uses a separate thread from the existing local publication executor. The
connector's existing external GET/PUT lock is preserved; bounded writes can still
delay a GET by the duration of an in-flight native transfer.

The retention limit bounds waiting and spill-source ownership. Existing capture
staging, active request/graph references and bounded boundary workspace are
additional allocations. Reaching the limit does not block capture on the network:
new protection may be refused, retaining the previous evictable local checkpoint
behavior. Such refusals are logged. This is not a promise to retain every
checkpoint under arbitrary pressure.

## Lifetime and failures

- Local readiness and successful remote persistence remain separate states.
- Cancellation removes an offer promptly; pending native work keeps independent
  references until completion. Cancellation during boundary preparation is checked
  again before submitting a PUT.
- Successful resumed dispatch and the matching all-worker release acknowledgement
  jointly retire waiting ownership. Failed attempts preserve usable waiting data.
- Known PUT failures keep local protection and retry with a one-second backoff.
  Source references remain counted during an in-flight spill, including after
  request cancellation.
- Unknown native completion quarantines source/workspace owners and refuses safe
  shutdown. Follow the existing coordinated restart policy; do not force-free the
  registered heap.
- Strict remote presence checks distinguish explicit absence from error statuses.
  Positive short reads, malformed metadata and transport failures are not silently
  accepted as cache misses.
- A lookup overlapping spill confirmation rechecks the source transition before
  declaring a local miss. Confirmed remote coverage remains usable after eviction.
- Cache freeze pauses new spill writes and keeps checkpoint reads local-only.
  Already submitted transfers retain their owners until terminal completion.
- Finishing a request does not delete canonical remote keys, which can be shared.
- Normalization updates a manifest only if its original snapshot is still current;
  an old restore cannot replace a newer record under a reused identity.

Original prompt availability is still checked independently. Mooncake can also
evict data. Remote copies do not provide request restart recovery: manifests,
accepted history and request state remain process-local, and disk durability is
determined by the external storage deployment.

## Validation

Focused CPU tests cover retention budgeting, shared-owner accounting, partial
accepted tails, mixed-C8 planes, boundary lengths, alternating sources, remote
eviction, allocator refusal, cancellation, stale notifications, GC-off cleanup,
and unknown completion. They exercise real Python control/assembly methods with
mocked NPU and network boundaries; some integration checks use AST extraction.

Before production enablement, force preemption and LocalCPU pressure on NPU,
verify restored KV/next-token outputs and MTP acceptance, and measure checkpoint
retention, remote transfer latency, recomputed tokens, TTFT/TPOT and EP-wide
throughput. Additional Python work, CPU memory retention and network traffic are
real costs; CPU test success is not a hardware speedup claim.
