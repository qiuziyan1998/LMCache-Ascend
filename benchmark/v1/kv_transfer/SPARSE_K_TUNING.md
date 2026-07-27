# Sparse K transfer tuning benchmark

`benchmark_sparse_k_transfer_tuning.py` compares four runtime-selectable paths
from one LMCache-Ascend build:

- `production`: the unchanged prepared one-plane sparse transfer.
- `original_tuned`: the same generic kernel with an explicit AIV count.
- `k_serial`: the dedicated K-only kernel with tiled, non-overlapped copy.
- `k_pipeline`: the dedicated K-only kernel with a two-buffer
  host-to-UB / UB-to-NPU pipeline.

The source uses the production-style `MLA_LATENT` chunk allocation: a
configurable K plane (512 BF16 elements by default) followed by the normal
64-element V/rope plane. The latter remains in memory so chunk spacing,
registration, and page behavior are realistic, but neither kernel reads it.
The destination is prepared through the existing one-plane `DSA_INDEX`
specialization over the K tensor; this invokes the same generic MLA transfer
code with `V=0`. Bandwidth accounting includes K only.
The harness suppresses its optional per-rank CPU reference clones, so a shared
slab remains the only source allocation and is not pre-scanned by every
worker.

All tuning parameters are kernel arguments. Rebuilding between configurations
is not required.

Each experimental configuration performs one fully validated launch before
warmup. Timed launches disable repeated tensor/device validation so their host
path is comparable to the production prepared kernel. `--verify` additionally
checks exact K values, untouched V, and untouched padded row tails.

## Build

Build and install LMCache-Ascend normally after checking out both the parent
repository and its updated `third_party/kvcache-ops` revision. Confirm that the
Python process imports the rebuilt extension rather than an older installed
copy:

```bash
git submodule update --init --recursive
pip install -v --no-build-isolation -e .
python3 -c 'import lmcache_ascend.c_ops as c; print(c.__file__)'
```

## Smoke test

Run the correctness check and a small representative matrix first:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
numactl --interleave=all \
python3 benchmark/v1/kv_transfer/benchmark_sparse_k_transfer_tuning.py \
  --devices 0,1,2,3,4,5,6,7 \
  --preset smoke \
  --num-layers 8 \
  --warmup 3 \
  --iters 20 \
  --repeats 5 \
  --shared-cpu-slab \
  --verify \
  --output-json /workspace/qzy/sparse-k-smoke.json \
  --output-csv /workspace/qzy/sparse-k-smoke.csv
```

The smoke preset covers 256 and 2048 selected tokens, both no-count and
one-row count-aware metadata, and contiguous plus fully scattered addresses.

## Focused tuning runs

Do not begin with the full Cartesian product. First prune AIV/tile/pipeline
candidates on the scattered no-count path:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
numactl --interleave=all \
python3 benchmark/v1/kv_transfer/benchmark_sparse_k_transfer_tuning.py \
  --devices 0,1,2,3,4,5,6,7 \
  --num-tokens 20000 \
  --selected-counts 256,2048 \
  --chunk-sizes 256 \
  --request-counts 0 \
  --aiv-counts 4,8,16,0 \
  --tile-tokens 1,8,16 \
  --addressing-modes auto \
  --work-assignments striped \
  --variants production,k_serial,k_pipeline \
  --localities src_scattered__dst_scattered \
  --num-layers 8 \
  --warmup 3 \
  --iters 20 \
  --repeats 5 \
  --shared-cpu-slab \
  --output-json /workspace/qzy/sparse-k-prune.json \
  --output-csv /workspace/qzy/sparse-k-prune.csv
```

Then verify a short list around the winners across request counts and
localities. For example, the following checks the most useful serial
shift/striped candidates:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
numactl --interleave=all \
python3 benchmark/v1/kv_transfer/benchmark_sparse_k_transfer_tuning.py \
  --devices 0,1,2,3,4,5,6,7 \
  --num-tokens 20000 \
  --selected-counts 256,2048 \
  --chunk-sizes 256 \
  --request-counts 0,1,4 \
  --valid-fractions 1.0,0.75 \
  --aiv-counts 4,16,0 \
  --tile-tokens 8,16 \
  --addressing-modes auto \
  --work-assignments striped \
  --variants production,k_serial \
  --localities all \
  --num-layers 8 \
  --warmup 3 \
  --iters 20 \
  --repeats 5 \
  --shared-cpu-slab \
  --verify \
  --output-json /workspace/qzy/sparse-k-validate.json \
  --output-csv /workspace/qzy/sparse-k-validate.csv
```

`request-counts=0` omits `selected_token_counts`. Positive values create
fixed-width rows, and `valid-fractions` leaves padded row tails that must
remain untouched. Shapes whose row capacity exceeds `num-tokens` are skipped.
The capacity must also remain strictly smaller than `num-tokens`, because this
benchmark intentionally exercises the sparse prepared path.

For a faster first performance run, use only
`--localities src_scattered__dst_scattered`, then validate the winner with
`--localities all`.

The benchmark prints the number of configuration cases and estimated layer
launches before starting. Add `--dry-run` to inspect that work without
starting NPU workers. Rank 0 prints flushed setup and periodic ETA messages;
`--progress-every 10` is the default, and `--progress-every 0` disables only
the periodic messages.

## Broader presets

`--preset standard` covers token counts from 64 through 4096 and both
power-of-two and non-power-of-two chunk addressing. `--preset full` also
sweeps chunk sizes, multiple request-row counts, and padded variable-count
rows. The script prints the estimated number of timed repeats before starting;
the full Cartesian matrix can take a long time.

Any preset list can be replaced with its corresponding comma-separated flag.
For example, this isolates AIV tuning of the original kernel:

```bash
python3 benchmark/v1/kv_transfer/benchmark_sparse_k_transfer_tuning.py \
  --devices 0 \
  --selected-counts 64,128,256,512,1024,2048,4096 \
  --chunk-sizes 256 \
  --request-counts 0 \
  --variants production,original_tuned \
  --aiv-counts 2,4,8,12,16,24,32,0 \
  --localities src_scattered__dst_scattered \
  --verify \
  --output-json /workspace/qzy/sparse-k-original-aiv.json \
  --output-csv /workspace/qzy/sparse-k-original-aiv.csv
```

Requested AIV values larger than the device AIV count are filtered at runtime.
`aiv=0` means the production automatic count.

## Interpreting output

For every shape and locality, results are sorted by NPU event time and include:

- microseconds per layer;
- per-rank effective K payload GB/s;
- cross-rank critical wall GB/s;
- speedup over the unchanged production kernel.

The final “stable settings” table uses geometric-mean speedup and reports the
worst case. Prefer a setting that:

1. passes `--verify`, including padded-tail checks;
2. improves both 256- and 2048-token cases;
3. has no meaningful worst-case regression across all four locality patterns;
4. uses fewer AIVs when its latency is within roughly 3% of the fastest result.

JSON contains every sample-derived summary and the per-shape winner. CSV is a
flat table suitable for plotting or pivoting.
