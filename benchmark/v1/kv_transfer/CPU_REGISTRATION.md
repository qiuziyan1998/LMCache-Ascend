# CPU-cache registration probe

Run from the LMCache-Ascend checkout with the same installed native extension,
driver, container limits, device visibility and NUMA settings as the deployment.
This benchmark introduces no native or serving-path changes, so it needs no
rebuild if the extension already matches `feat/indexer-c8-shared-lcm`.

## Actual code path

- `lmcache_ascend/__init__.py::_patch_ops` redirects LMCache's native and
  non-CUDA allocator imports to `lmcache_ascend.c_ops`.
- LMCache-NPU `lmcache/v1/memory_management.py::_resolve_pinned_alloc_free`
  selects shared, NUMA-bound, or ordinary pinned allocation.
- `csrc/common/mem_alloc.cpp`: shared owner uses `shm_open(O_EXCL)`,
  `ftruncate`, `posix_fallocate`, `mmap`, first-touch, then `register_ptr`.
  First-touch uses `MADV_POPULATE_WRITE` or a page-write loop; optional NUMA
  interleaving applies during population. Other TP processes map the SAME slab
  and independently register it for their current NPU. The engine normally
  tries read-only attachment before writable fallback; the probe does too.
- Ordinary allocation uses `aclrtMallocHost`, then `register_ptr` except on
  310 devices. NUMA allocation uses anonymous mmap, mbind and memset before
  registration.
- `csrc/common/managed_mem.cpp::register_ptr` chooses mapped
  `aclrtHostRegister` on drivers accepted by `is_version_at_least_25`, otherwise
  `halHostRegister` followed by explicit `mlock`. Device pointers are retained
  in a per-process registry. Free/detach uses the corresponding native API.

No explicit 200 GB cap appears in these allocation/registration functions.
Allocation capacity, tmpfs quota, cgroup limits, NUMA placement, pinning limits,
and driver/device mapping resources are distinct possible failure stages.
This does NOT prove which caused the previous failure.

This is the LMCache LocalCPU slab, not Mooncake's `global_segment_size` or
Mooncake transport registration. No Mooncake store, KV tensors, or model is
created. A passing result does not establish Mooncake capacity or KV-transfer
correctness. `get_device_ptr` checks registry coverage, not an actual DMA read.

## Run

Use an idle test host with sufficient RAM and `/dev/shm`. The requested memory
is really populated/pinned; a container OOM kill is possible. Do not run the
large sweep alongside production. GiB means 2^30 bytes: 200 GiB is 214.75 GB;
200 decimal GB is approximately 186.265 GiB. Keep visibility and device IDs
consistent with serving. For example, logical IDs 0..3 must all be visible.

Start small:

```bash
python benchmark/v1/kv_transfer/benchmark_cpu_registration.py --sizes-gib 1 --devices 0 1 2 3 2>&1 | tee registration-smoke.log
```

One slab, one NPU; each size runs in a fresh process:

```bash
python benchmark/v1/kv_transfer/benchmark_cpu_registration.py --sizes-gib 160 180 200 220 240 --devices 0 2>&1 | tee registration-single.log
```

The same slab registered in four TP workers (240 GiB physical total, not 960):

```bash
python benchmark/v1/kv_transfer/benchmark_cpu_registration.py --sizes-gib 240 --devices 0 1 2 3 2>&1 | tee registration-tp4.log
```

Compare six simultaneously live 40 GiB slabs with one 240 GiB slab:

```bash
python benchmark/v1/kv_transfer/benchmark_cpu_registration.py --sizes-gib 240 --chunk-gib 40 --devices 0 1 2 3 2>&1 | tee registration-chunks.log
```

Use `--interleave-nodes 0 1` only to match the actual shared-cache NUMA policy.
Use `--attach writable` if serving explicitly sets passive writable mapping.
For non-shared allocation use `--mode private`, or `--mode numa --numa-node 0`.
**Those modes allocate a separate full-sized pool per worker.** Multiple DP
groups likewise need separate shared-slab runs; their physical allocations add.
`--hold-seconds 60` keeps registrations live for external inspection.

## Interpret results

Capture stdout AND stderr. JSON records report each region's duration, memory
snapshot, process/device, native extension path, Torch versions and exit status.
Native shared-slab logs expose reserve/populate/owner_register/attach_register
start and completion. The last started phase without completion localizes a
hang. ACL/HAL error codes remain in stderr. Private/NUMA timings are combined
allocation+registration; the benchmark does not falsely separate them.

Before spawning workers, the probe reports `/dev/shm` total and available bytes.
If the page-rounded total exceeds available space, it exits with
`failure_stage=shm_capacity_preflight` and `registration_attempted=false`.
Chunking does not bypass this check: all chunks remain allocated together.
This is a snapshot, not a reservation; concurrent allocations can still cause
a later native reserve failure. A 200 GiB tmpfs cannot test a 240 GiB shared slab;
increase the container's shm capacity first. Passing this check does not prove
that the container has enough RAM.

- `reserve` failure: examine `/dev/shm` free space/quota.
- `populate` failure/kill: inspect available RAM, NUMA policy and cgroup OOM
  counters (also ancestor cgroups if the current cgroup has no explicit limit).
- `owner_register` or `attach_register` failure: preserve the exact driver code;
  compare one vs many devices and one vs many regions at the same total size.
- Single region fails but chunked succeeds: evidence of registration shape
  sensitivity, not yet proof that splitting production slabs is safe.
- First device passes but additional devices fail: investigate aggregate
  registration resources and passive mapping permissions.

Record `npu-smi info`, `ulimit -l`, `df -h /dev/shm`, and kernel OOM/driver logs
around failures. `/proc` snapshots include memlock limits, RSS/locked memory,
host memory and best-effort cgroup v1/v2 counters. Controller paths are resolved
against `/proc/self/mountinfo`, including container bind-mount roots. The
`memory_cgroups` field records each readable group and visible ancestor, since
an ancestor can impose a lower limit. An empty field means the limits could not
be read, not that memory is unlimited; ancestors outside the container's visible
mount remain unknown. VmLck alone does not measure all driver-pinned pages.
No system limits are changed by the benchmark.

Workers are started in order to isolate the first failing device, not to
measure parallel startup throughput. All earlier registrations remain live.
Cleanup detaches passive workers before freeing the owner's slabs. A timeout
terminates the worker; unique trial shm names are unlinked by the parent.
SIGKILL of the parent itself cannot run cleanup; remove only that run's
`/dev/shm/lmcache_regbench_<uuid>_*` after ensuring its workers have stopped.
Default per-worker startup/cleanup timeout is 600 seconds; increase it if a
large healthy population exceeds that. The sweep stops on its first failure.

Host-only checks:

```bash
python tests/standalone/test_cpu_registration_benchmark.py
```
