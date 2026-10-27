# CPU validation of the vLLM / LMCache P-D control path

`vllm_cpu_e2e.py` runs two **real `EngineCore` instances** with
`AsyncRecomputeScheduler`, the original `NPUModelRunner`, the Ascend connector,
`AscendLMCacheEngine`,
`StorageManager`, `LocalCPUBackend`, and `MooncakestoreConnector`. These are
normal imports of the complete repository modules, not extracted method bodies.
Keep the four repositories in sibling directories: `vllm`, `vllm-ascend`,
`LMCache`, and `LMCache-Ascend`.

The CPU backend replaces the executor RPC boundary, native device functions,
streams/events, shared-memory mapping, and native Mooncake store SDK. The model
is an identity model with deterministic KV values. The real NPU runner executes
input preparation, request/block-table updates, attention metadata construction,
model dispatch, sampling, and asynchronous output materialization. The real SFA
transfer-window callbacks surround the identity model's CPU arithmetic.
The real scheduler supplies
request admission, chunk boundaries, block allocations, bank phase metadata,
and completion state. P and D have distinct HBM buffers and distinct CPU slabs.
Only the simulated remote store connects their payloads. Every TP worker has
its own connector and compute stream; nonzero ranks consume the real shared
cache descriptor protocol.

## Run

Use a CPU PyTorch environment with vLLM/LMCache's Python dependencies. No model
weights, NPU extension, Mooncake service, or server configuration file is needed.

```bash
python tools/vllm_cpu_e2e.py --output cpu-e2e-report.json
python tools/vllm_cpu_e2e.py --layers 79 --tp-size 2 --max-num-seqs 1 --rounds 1 --output cpu-e2e-79-report.json
python tools/vllm_cpu_e2e.py --requests 2 --max-num-seqs 2 --output cpu-e2e-mixed-report.json
python -m pytest --confcutdir tests/standalone tests/standalone/test_vllm_cpu_e2e.py -q
```

Defaults: 512,000 context tokens, 16,384 scheduled tokens per batch, TP4,
four banks, one request, two conversation rounds, nine model layers, and 16 D
output tokens per request per round. The first round therefore has **32 actual
chunked-prefill steps**, including a final 4,096-token chunk. The second round appends
the simulated response and more input to the previous prefix. P uses async
store; D uses synchronous store configuration and the P feature is disabled.
With more than one request, lengths differ by 37 tokens and the scheduler
creates mixed batches naturally. The LMCache object size is separately set to
1,024 tokens; it does not change the 16k prefill scheduling budget.
`--banks`, `--seed`, `--layers`, `--requests`, `--rounds`, `--context-tokens`, `--batch-tokens`,
`--cache-chunk-tokens`,
`--max-num-seqs`, and `--decode-tokens` can change the case. `--connector-only`
retains the earlier lightweight executor for isolating connector behavior.
The completion
messages are merged by vLLM's real `KVOutputAggregator` before being delivered
to the scheduler.

## What is checked

- P compares every historical KV element against an independent expected
  value before writing newly computed KV. D checks its actual selected latent
  rows and full indexer history; deterministic top-k spans the complete prefix.
- P continuation chunks and subsequent conversation rounds must retrieve the
  earlier prefix correctly. Logical layer names share physical P storage.
- D lookup must hit remotely stored data. Its initial load and later explicit
  sparse loads run through the actual connector and allocator code.
- Queue effects are deferred. Recording an event captures a dependency; it
  does not execute the transfer. Compute and transfer queues may finish in a
  different order from their host submission order.
- Each modeled KV write has a separate start and completion. Start fills the
  destination with `0xff` (invalid for the finite dummy values); completion
  writes the correct payload. Events point to completion, including simulated
  Mooncake receive and publish operations. Pending writer checks reject an
  early read even when the previous contents happen to equal the next payload.
  Poisoning follows stream dependencies so it cannot damage a preceding
  layer that still legitimately owns the bank.
- Native copies check source and destination bounds. Missing operations fail
  explicitly instead of silently becoming no-ops.
- All requests must finish and return the expected token count and token IDs.
  A run with no scheduled work cannot pass. Reports include remote reads/writes
  and checked row counts, exact chunk ranges and completed/poisoned operation
  counts. Unused large CPU allocation capacity uses zero-filled anonymous
  mappings to avoid touching gigabytes that the tested request never uses.
- Negative tests remove load waits, remove save waits, incorrectly mark data
  ready at transfer start, or corrupt the remote payload. All four must be
  detected. A deliberate device fault terminates its isolated
  test process without waiting for failed asynchronous stores to shut down.

## Limits

This tests the Python execution path with a CPU identity model. It does not
execute GLM arithmetic, the numerical SFA attention implementation,
MTP acceptance, native kernels, HCCL, HTTP/proxy routing, or ACL graph replay.
The identity model supplies deterministic top-k indices. The transport has
the native store's API but not its network implementation or failure modes.
The CPU results therefore do **not** establish that the production NPU race
has been fixed. Use this harness to reproduce and reject ownership/order
violations in the covered path; retain real-device tests for the boundaries
listed above.
