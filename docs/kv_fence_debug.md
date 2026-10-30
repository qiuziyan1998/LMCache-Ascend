# Isolating layerwise-prefill timing faults

This diagnostic separates the host waits previously combined with byte
sampling by `LMCACHE_PREFILL_STORE_DEBUG`. It does not read or save KV payloads,
create a queue, modify tensor layouts, or change graph/scheduler settings.
Update **LMCache and LMCache-Ascend** together. It is disabled by default.

Disable the older probes in the worker environment:

```bash
export LMCACHE_PREFILL_STORE_DEBUG=0
export LMCACHE_PREFILL_AUDIT=0
```

Set `LMCACHE_PREFILL_FENCE_DEBUG` before starting workers. The same value may
be used on P and D; each worker applies only the stages belonging to its role.

| Value | Added behavior |
| --- | --- |
| `check` | Host-only P handoff checks; no event queries or device waits |
| `p_load` | Wait for each P H2D completion event immediately after submission |
| `p_use` | Wait for the P layer's existing final-bank-access event immediately after SFA, before output projection |
| `p_reuse` | Wait for that same event immediately before the bank's next save/load submission |
| `p_save` | Wait for each P D2H completion event immediately after submission |
| `d_load` | Wait for each D dense/sparse NPU load before returning from its submission |
| `all` | The original four synchronization stages; uses `p_use`, excludes `p_reuse` |
| `0` | Disabled, including the host checks |

Comma-separated stages, e.g. `p_load,p_use`, are supported. Host-only P checks
are separate: enable `check`, or `check,p_use` to combine them with a fence.
`p_use,p_reuse` is rejected: these are alternative positions for the same wait.
The fence-only modes do not build expected-load metadata or assert on cache
misses, so the first isolation run adds just the selected waits.
Enabling a fence together with the older
STORE_DEBUG or AUDIT switch is rejected at startup to keep the experiment clear.

`p_use` waits for the already-recorded last access, not a later snapshot of the
compute stream. It does not wait for the next layer's FFN. `d_load` covers the
connector's NPU H2D operators; direct Mooncake-to-NPU imports already use the
blocking external-get contract and are not turned into another payload copy.
None of these modes is a performance benchmark.

## Comparing immediate use and deferred reuse fences

For the latent-bank corruption experiment, set these on **all P workers**,
leave D fences disabled, and restart the P processes between modes:

```bash
export LMCACHE_PREFILL_STORE_DEBUG=0
export LMCACHE_PREFILL_AUDIT=0
export LMCACHE_PREFILL_FENCE_GROUPS=0
unset LMCACHE_PREFILL_FENCE_LAYERS

# Mode A: the existing fence immediately after SFA.
export LMCACHE_PREFILL_FENCE_DEBUG=p_use

# Mode B: use this instead of Mode A, before launching the next run.
export LMCACHE_PREFILL_FENCE_DEBUG=p_reuse
```

Both modes use the original final-access `event_N`, recorded on the compute
stream. Mode B keeps the existing device-side stream waits, but lets the host
submit projection/FFN before synchronizing that event at the actual DMA
boundary. It does not record a new event or synchronize the whole stream.
Normal intermediate layers defer their save to the next layer's entry; final
target layers and one-layer MTP forwards keep their immediate save path.

The reuse fence covers both directions: before save N, and before load N+B
for ranks that skip saving. It also covers the next chunk's initial loads when
a previous reader event is present. Multiple requests sharing a bank retain
their separate load-completion events; the shared final-reader event is host
synchronized only once, even if followed by several saves/loads. The layer
filter and activation log identify the **source layer N**, not load N+B.

`[KVFENCE]` emits the selected mode at startup and one activation per group.
There are no payload copies, hashes, files or background queues. With the
switch unset or `0`, no diagnostic object, event query or host wait is added.

Keep input, concurrency, graphs, scheduler and bank count unchanged. If Mode B
also removes corruption, investigate the compute-to-DMA handoff next. If only
Mode A works, waiting before bank reuse alone is insufficient: inspect the
lifetimes of SFA inputs/output/workspace and subsequent host submissions. These
experiments narrow the suspect window; neither result alone proves a root cause.

## First run

Use the same concurrent client input, normal graph settings, bank count,
batch limit and scheduling configuration as the failing run. Initially keep
the warmup and first followup round: do not add another five rounds after the
first corrupt response.

```bash
export LMCACHE_PREFILL_FENCE_DEBUG=p_use
```

This tests the compute-to-bank-reuse boundary while D remains unsynchronized.
The output `[KVFENCE] ... "event":"active","stage":"p_use" ...` confirms
that the hook actually ran. If a `KVFENCE` error is emitted, inspect that error
before interpreting the final generated text.

## Evidence and narrowing

When `check` is enabled, the checks derive expected loads from each step's request metadata, before
submitting transfers. They require a load for every declared hit, distinguish
real no-load requests, and compare each request's consumer event to its last
submitted load event. Layer ordinals select fixed banks, independent of prefix
length; latent and indexer ordinals start at zero independently. A completed event is not queried as a substitute
for this ownership check.

- `request_load_mismatch`: a declared request's load is missing, or a load was
  submitted for a request not declared for this bank.
- `wrong_load_event`: the consumer lost or replaced the actual latest event;
  it must not silently fall back to an older save event.
- `load_after_consume`: another load was submitted after this layer consumed
  that bank.
- `device_fence_failed`: the device reported failure at the named boundary.

Each error contains worker, step, group, layer, bank and relevant request IDs.
Successful checks produce no per-layer log. Each selected stage/group emits
only its first activation line per worker. Extract the small diagnostic log:

```bash
grep -h '\[KVFENCE\]' tp4_dp4_logs/*.log
```

If `p_use` removes corruption, keep that stage and bisect its scope:

```bash
export LMCACHE_PREFILL_FENCE_GROUPS=0
export LMCACHE_PREFILL_FENCE_LAYERS=0:40
```

Group 0 is latent KV; group 1 is the indexer. Layers are **group-local physical
owner ordinals**, not necessarily model layer numbers. `0:40` includes 0–39.
Repeat with the other half, then split the half whose waits are needed. Keep
both halves when neither alone suffices; there may be multiple failing
boundaries. Defaults are both groups and all layers.

If `p_use` does not remove corruption, test `p_load`, then `p_save`, then
`d_load`, changing only this value. `all` is a positive control for the
combined waits when no individual stage suffices. Do not interpret `all`
failing while the file-dumping probe passes as proof that device waits fixed
the bug: file I/O, references, and host delays were removed by design.

A host-protocol error is direct evidence at the named boundary. A successful
synchronization experiment narrows the suspect lifetime/dependency window;
confirm the fix by repairing that dependency and reproducing normal output
with **all diagnostics disabled**. A good output under a fence alone is not
an end-to-end correctness certification.

`request_load_mismatch` means that a declared hit was not loaded, not necessarily
that a stream dependency was lost. Inspect preceding retrieval errors as well;
the checker intentionally stops before the model can consume a missing prefix,
including cases where normal execution would ask the scheduler to recompute it.
