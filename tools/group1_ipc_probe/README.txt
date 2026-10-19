Mooncake / Ascend imported-HBM feasibility probe
===============================================

This standalone probe tests whether a separate process can import a worker's
NPU allocation, register it with the installed Mooncake/Ascend transfer engine,
and use Store GET to write known bytes directly into that allocation.

No LMCache/vLLM imports, source changes or Mooncake rebuild are needed.
Run it in the D container's Python environment containing torch, torch_npu and
the same mooncake.store .so used by the decoder. Use idle NPU devices: the probe
starts extra NPU contexts, whose memory exceeds the ~520 KiB test buffer per W.
It does not modify the running server or use its KV allocations.

Files
-----
mooncake_hbm_ipc_probe.py   Standalone NPU test; this is the only required file.
test_probe_cpu.py          Local CPU helper and real subprocess protocol tests.

Repository location: LMCache-Ascend/tools/group1_ipc_probe. After pulling
feat/lmcache-separate-process, enter this directory to run the commands below.

Setup
-----
Copy the script into the container. Keep the deployment's CANN/library/network
environment. Supply the existing Mooncake master address and this D host's
routable IP. These are placeholders, not inferred production addresses:

  export PROBE_HOST='YOUR_D_HOST_IP'
  export PROBE_MASTER='YOUR_MOONCAKE_MASTER_IP:PORT'

--metadata-server defaults to P2PHANDSHAKE for Store.setup; override it if your
Mooncake store configuration uses a different metadata server. TransferEngine
initialization uses P2PHANDSHAKE, matching the current vLLM-Ascend wrapper.
--device-name defaults to the empty transport device name used by that wrapper.
Both W and M inherit the same device visibility environment; --devices lists
logical NPU IDs in that environment. The script does not remap device visibility.

Run these stages in order
-------------------------
1. One device: validate IPC import, registration, direct read and retirement.

  python3 -u mooncake_hbm_ipc_probe.py \
    --host "$PROBE_HOST" --master "$PROBE_MASTER" --devices 0 \
    --output ipc-one.json 2>&1 | tee ipc-one.log

2. Four devices in ONE M process, with a separate engine/store per device:

  python3 -u mooncake_hbm_ipc_probe.py \
    --host "$PROBE_HOST" --master "$PROBE_MASTER" --devices 0,1,2,3 \
    --engine-mode per-device --output ipc-four-separate.json \
    2>&1 | tee ipc-four-separate.log

3. Four devices in ONE M process, sharing one engine/store initialized on the
   first device. This tests whether that alternative is supported at all; it is
   not assumed to work merely because GlobalTE is currently a singleton:

  python3 -u mooncake_hbm_ipc_probe.py \
    --host "$PROBE_HOST" --master "$PROBE_MASTER" --devices 0,1,2,3 \
    --engine-mode shared --output ipc-four-shared.json \
    2>&1 | tee ipc-four-shared.log

Use `set -o pipefail` when relying on the exit status of a command piped to tee.
Start with stage 1. Send its JSON report and log if it fails; do not continue
through all stages after a native error. Default control timeout is 180 seconds.

What it does
------------
- Uses Python spawn for all subprocesses; the supervisor does not initialize
  an NPU runtime. There is one exporter W per device and one importing M.
- Uses the same _share_npu_ / _new_shared_npu primitive sequence as the existing
  AscendIPCWrapper, without importing the larger LMCache stack.
- Exports a nonzero-offset uint8 view of a worker-owned allocation. IPC messages
  contain primitive handle values and geometry, not pickled live tensors.
- M computes addresses from its imported view, never W's numerical pointer.
- Creates unique `lmcache-ipc-probe-<uuid>/...` objects through the installed
  batch_put_from_multi_buffers API, using registered CPU fixtures. Reader stores
  have zero global segment and local staging-pool size, matching the existing
  external-page reader. The existing cluster must have writable storage space.
- Calls batch_get_into_multi_buffers into contiguous, fragmented and short
  partial destinations, including multiple destination buffers per key.
- Launches the per-device GET calls behind a thread barrier for the four-device
  cases. The test attempts concurrent calls; timings do not prove transport
  overlap or that the binding releases the GIL.
- Repeats all three cases three times, changing the source patterns and keys
  between iterations. Each W checks the ENTIRE allocation against expected
  bytes, including the leading/trailing guard and untouched internal gaps.
- Keeps GC disabled in W/M, explicitly retains exporters and importers, and
  delays each M completion reply by 100 ms to exercise retention across replies.
- Unregisters M's buffers and closes its stores, waits for M to exit, and only
  then releases W's allocations. Removes only the unique keys it created when
  Store.remove is available. Deletion failures appear in cleanup.warnings; the
  JSON retains the namespace for identifying any small leftover probe objects.

Result interpretation
---------------------
PASS means all tested bytes/guards matched and normal process retirement
completed. native_call_ms covers the synchronous GET call, not just network
transfer, and is not a throughput benchmark.

owner_error: device initialization or IPC export failed.
imported: M reconstructed the exported view; registration may still fail later.
reader_ready: engine and Store setup succeeded; no HBM data transfer proven yet.
storage_ready: all imported HBM registrations succeeded.
phase_result with ok=true: byte counts, actual data and guards passed that case.
Fixture PUT failed: inspect source allocation/master/storage setup first; this
does not establish that imported-HBM reads are unsupported.

A native call exception or timeout may leave completion unknown. The probe
will NOT automatically unregister/free/reuse those buffers, kill the importer,
or restart a service. It writes UNKNOWN_COMPLETION_OR_RETIREMENT (or a retirement
failure), prints the probe PIDs and retains them for inspection. Such a run
requires operator diagnosis and coordinated cleanup of these probe processes;
process death alone is not treated as proof of DMA completion. Do not use a
shell timeout that blindly kills the whole group during this test. Ctrl-C is not
a native-transfer cancellation API.

Scope limits
------------
This checks the installed transport and IPC mechanism with deterministic raw
bytes. It does not validate GLM's mixed BF16/C8 planner, production key/layout
identity, checkpoint composition, service restart, cancellation during real DMA,
graph replay, inference accuracy, cross-host source placement or throughput.
The short/fragmented byte layouts exercise scatter destinations; they are not
claimed to be complete GLM key/scale layout tests. Successful completion is a
feasibility gate before integration, not a production qualification result.

The probe does not use replica descriptors to bypass Store GET or assume their
lifetime is pinned. It does not change the installed Mooncake shared library.

Optional CPU tests (no Torch/NPU/Mooncake required)
------------------------------------------------
  python3 -m unittest -v test_probe_cpu.py

These test byte plans, guard expectations, failure latching and supervisor
handling with real spawned CPU-only fake children. They are separate from NPU
qualification and do not prove that the installed IPC or transfer APIs work.
One optional test verifies actual CPU Torch fixture alignment and mmap retirement;
it is skipped when Torch is unavailable.

Return the selected .json report and matching .log. The report/log identify the
actual imported Mooncake module, device IDs, IPC stage and each transfer result.
