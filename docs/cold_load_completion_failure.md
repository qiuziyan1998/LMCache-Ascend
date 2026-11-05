# Cold-load failure followed by an unknown send completion

## Observed failure

The 2026-09-11 12:32 DP3 log reports:

1. LocalCPU allocation failures for 1,179,648-byte objects.
2. Unsuccessful Group-0 Mooncake legacy reads (`code=-704`); the sampled batch
   reports 5,451 buffers, 6,386,486,400 bytes requested and zero bytes read.
3. A known-terminal cold-load error for request
   `cmpl-3420ce07-4f2a-4cf4-92f5-c8474d4cb4d5-0-94038a41`, propagated across TP
   workers before shared-handle publication. The connector reports invalid
   indexer blocks and receive completion.
4. Compact KV blocks are freed, then EngineCore fails at
   `_update_from_kv_xfer_finished`: `assert req_id in self.requests`, specifically
   in the `finished_sending` loop. The API later reports EngineDeadError.

The immediate server failure is a request-lifetime assertion, not an NPU timeout.
The log establishes the failed reads and memory pressure but does not identify
why every persistent read failed. No placement, capacity or read-error policy
change is justified by this excerpt alone.

## Reproduced cause

Native scheduler `finish_requests` retains an unfinished receive's blocks. The
Ascend connector can separately return true from `request_finished`, promising
an asynchronous send/store-cleanup acknowledgement. The scheduler previously
combined those obligations into one Boolean.

For a failed or aborted receiver, `finished_recving` deleted the request even
when connector cleanup still owed `finished_sending`. The later send completion
then hit the exact assertion from the incident. If the send arrived first, it
could instead free blocks while receive ownership remained live.

The reproducer executes the actual scheduler finish/completion methods and the
actual Ascend `get_finished_stores` method. It failed at the same sending-loop
assertion before the fix. This lifecycle code also existed on the throughput
baseline; it does not depend on `decode_preemption_checkpoint`.

## Correction and ownership contract

- Record independent pending-receive and pending-send obligations when a request
  finishes. These are internal Request defaults, not new request API fields.
- Each terminal notification clears only its own obligation. Free blocks and
  delete the request exactly once, when both obligations are complete.
- Preserve strict checks for unknown request IDs and send completions for active
  requests. Do not mask arbitrary stale replies with an unconditional skip.
- When a client aborts during a cold load, emit the connector cleanup
  acknowledgement after both load workers and readiness fences retire. Ascend
  uses its existing store finalizer, so pending store ownership is also respected.
- Drain acknowledgements created by cold-load retirement in the same worker
  `get_finished` call. The existing late-completion set is reused.

The aborted-load acknowledgement is necessary: merely making the scheduler wait
for both signals could otherwise strand requests whose finish notification
arrived while the load was still in flight. Four worker regressions reproduced
that missing acknowledgement before the companion correction.

No per-layer or per-token model path changes, new collective, synchronization,
cache retry loop, configuration knob or native-code change is introduced.
Bookkeeping runs on request completion/abort/error. Normal successful receives,
send-only finishes and receive-only cancellations retain their existing behavior.

## Validation and deployment

The focused CPU suite passes 101 tests: vLLM 16, vLLM-Ascend 26,
LMCache-NPU 18, LMCache-Ascend 41. New cases cover failure/abort, receive-first,
send-first, same-batch signals, unrelated active requests, strict protocol errors,
already-completed receives, pending sibling/readiness fences, delayed stores,
and modes that do not promise a send acknowledgement (pure consumer or synchronous store).
They execute production control methods with device/storage boundaries mocked;
they do not qualify NPU execution or the underlying Mooncake read failure.

Deploy the coordinated Python changes in vLLM, LMCache-NPU and LMCache-Ascend on
`fix/decoder-resume-dp-sync`, then restart the services. vLLM-Ascend needs no new
change for this incident. No additional native rebuild is required by this
follow-up. The previously documented native binding remains required if enabling
generated-KV checkpoint capture.

The failed request still follows the configured KV-load failure policy. This
fix prevents its completion bookkeeping from killing the engine; it does not
turn unavailable KV data into a successful cache hit.
