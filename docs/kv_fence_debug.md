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
| `p_use` | Wait for the P layer's existing final-bank-access event, before its bank can be saved/reloaded |
| `p_save` | Wait for each P D2H completion event immediately after submission |
| `d_load` | Wait for each D dense/sparse NPU load before returning from its submission |
| `all` | All four synchronization stages |
| `0` | Disabled, including the host checks |

Comma-separated stages, e.g. `p_load,p_use`, are supported. Host-only P checks
are separate: enable `check`, or `check,p_use` to combine them with a fence.
The fence-only modes do not build expected-load metadata or assert on cache
misses, so the first isolation run adds just the selected waits.
Enabling a fence together with the older
STORE_DEBUG or AUDIT switch is rejected at startup to keep the experiment clear.

`p_use` waits for the already-recorded last access, not a later snapshot of the
compute stream. It does not wait for the next layer's FFN. `d_load` covers the
connector's NPU H2D operators; direct Mooncake-to-NPU imports already use the
blocking external-get contract and are not turned into another payload copy.
None of these modes is a performance benchmark.

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
real no-load requests, and compare the consumer event to the last submitted
event on that physical bank. A completed event is not queried as a substitute
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
