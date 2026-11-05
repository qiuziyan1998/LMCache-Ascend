# Direct Remote LMCache qualification completion matrix

**Audit date:** 2026-08-25
**Wire protocol:** unchanged (`NEGOTIATE`, `OPEN`, `RESERVE_WINDOW`,
`ARM_WINDOW`, `REPORT_TRANSFER_COMPLETE`, `FINISH`, `ABORT`, `STATUS`)
**Production default:** disabled

This file records implementation evidence separately from hardware evidence.
Only a report produced on the exact manifest and production topology may turn
V1, V2, or V6 into `FIXED_AND_PASS`.

## Phase V

| Item | Current evidence | Local status | Required closure |
|---|---|---|---|
| V1 connector forwarding | `tests/v1/test_remote_fill_decoder.py::test_standalone_connector_forwards_remote_fill_contract` covers the standalone handoff, placement, metrics, and fatal forwarding; DP identity injection has focused coverage | `PENDING_HARDWARE` | Construct the actual standalone `LMCacheAscendConnectorV1Dynamic` on P and D |
| V2 complete producer fences | `LMCache-Ascend/tests/v1/test_remote_fill_producer.py::test_direct_page_batch_retains_same_owner_and_all_producer_events`; incomplete coverage fails before ARM in `lmcache_ascend/v1/cache_engine.py` | `PENDING_HARDWARE` | NPU proof that every producer stream is represented or causally joined |
| V3 physical persistent completion | Single-key, merged-batch, and partial-page completion tests in `LMCache-NPU/tests/v1/storage_backend/test_mooncake_store_completion.py`; batched writers now publish the real shared physical future | `PASS_INSPECTION`; Torch test execution pending | Run the focused Torch suite in CI/NPU image |
| V4 operation deadlines and replay | Every operation timeout and exact lost-reply replay are covered by `LMCache-NPU/tests/v1/remote_fill/test_transport.py` and `test_state.py` | `PASS_UNIT` | None before production-path qualification; real armed-DMA loss remains prohibited |
| V5 producer wait placement | Source event and registration precede semaphore admission; slow-event/ready-request regression is in `test_mooncake_remote_fill_direct_push.py` | `PASS_INSPECTION`; Torch test execution pending | Run focused connector test in CI/NPU image |
| V6 all-rank agreement | Core consensus success/fail-closed unit coverage is in `LMCache-NPU/tests/v1/test_remote_fill_actual_load.py` | `PENDING_HARDWARE` | Real TP8 Group 0 and Group 1 passive-failure campaigns with equal computed-token counts |
| V7 existing LocalCPU validation | Object kind, state, static metadata, valid-token, atomic visibility, and existing-winner tests are in `test_local_cpu_layer_page_contract.py` | `PASS_INSPECTION`; Torch test execution pending | Run focused LocalCPU tests in CI/NPU image |
| V8 no-benefit requests | Addressable-source skip is in `LMCache-Ascend/tests/v1/test_remote_fill_producer.py`; first-hole publication ineligibility is covered by pure state tests | `PASS_UNIT` plus inspection | Confirm zero RPC/native/source retention on deployed full-hit traffic |

## Qualification deliverables

| Deliverable | Infrastructure | Evidence state |
|---|---|---|
| Exact manifest | `remote_fill_qualification.py` plus strict inventory schema | Awaiting production inventory and clean deployed SHAs |
| V1--V8 record | strict evidence validator and example | Hardware closure pending V1/V2/V6 |
| Optional H0 A--G | `remote_fill_h0.py` production-adapter runner and strict validator | Nonblocking transport diagnostic; hardware report not run |
| C1 nine-scenario matrix | `remote_fill_c1.py` fixed production-adapter runner | TP8/DP2/MTP report not run |
| P1 A/B/C workload | client runner, comparator, trace builder, fixed campaign runner, strict aggregate evaluator | Performance campaign not run; asynchronous D-mode tool support is pending Phase 8 |
| Paired restart | executable `PairedRestartAdapter` state machine with stale-identity and incomplete-stop rejection | Production supervisor adapter not yet qualified; fault injection remains prohibited |
| O1/O2 | no code added | Correctly deferred until measured thresholds pass |

## Locally executable evidence

```bash
# LMCache-NPU: pure bounded protocol/state suite
PYTHONPATH="$TMP/remote-fill-test-deps:$PWD" \
  python -m pytest --confcutdir=tests/v1/remote_fill tests/v1/remote_fill -q

# LMCache-Ascend: manifest/H0/C1/P1 orchestration contracts
PYTHONPATH="$TMP/remote-fill-test-deps:$PWD" \
  python -m pytest --confcutdir=tests/v1 \
  tests/v1/test_remote_fill_qualification.py \
  tests/v1/test_remote_fill_h0.py \
  tests/v1/test_remote_fill_campaign.py -q

# vLLM-Ascend: pure paired-restart state machine
PYTHONPATH="$TMP/remote-fill-test-deps:$PWD" \
  python -m pytest --confcutdir=tests/ut/kv_connector \
  tests/ut/kv_connector/test_remote_fill_restart.py -q
```

Torch/NPU-dependent tests must run in the serving image. H0, C1, TP8/DP2, and
performance results cannot be replaced by these local tests.
