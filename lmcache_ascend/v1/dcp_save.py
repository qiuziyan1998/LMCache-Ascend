# SPDX-License-Identifier: Apache-2.0
"""Foreground DCP save boundary; persistence and promotion keep existing APIs."""

from bisect import bisect_right
from typing import Any

import torch

from lmcache.v1.memory_management import LayerPageMemoryObj
from lmcache_ascend.v1.dcp_store import store_group


def save_batch(adapter: Any, context: dict) -> None:
    """Capture all owners after the forward, before vLLM can recycle its blocks."""
    metadata = adapter._parent._get_connector_metadata()
    engine = adapter.lmcache_engine
    caches = None
    for request in metadata.requests:
        spec = request.save_spec
        if adapter.kv_role != "kv_consumer" and spec is not None and spec.can_save:
            input_error = None
            start, end, mask = 0, len(request.token_ids), None
            try:
                if request.dcp_write_layout is None:
                    raise ValueError("DCP save has no scheduler-owned block snapshot")
                if caches is None:
                    caches = adapter._direct_group_caches()
                adapter._note_decode_window_save_seen(request)
                start, end = adapter._layerwise_save_range(request)
                mask = torch.ones(end, dtype=torch.bool, device="cpu")
                mask[:start] = False
            except Exception as exc:
                input_error = str(exc)
            for group in (0, 1):
                if not (spec.can_save_latent if group == 0 else spec.can_save_indexer):
                    continue
                # Preserve the existing save/publication dependency gate. The
                # coarse write agreement has its own Gloo group, while source
                # promotion still uses the ordered shared-handle stream.
                options = {}
                error = input_error
                try:
                    if error is not None:
                        raise ValueError(error)
                    options = adapter._layerwise_store_kwargs(request, group)
                    prior = options.get("process_collective_predecessor")
                    if prior is not None:
                        prior.result()
                    old = adapter._worker_retrieve_state.get(request.req_id)
                    source = None if old is None else old.prepared_sparse_sources.get(0)
                    if group == 0 and source is not None:
                        if not engine.compact_source_importer.is_certified_source(
                            source,
                            request.req_id,
                            old.process_load_generation,
                        ):
                            raise ValueError(
                                "DCP save prefix is not the current request source"
                            )
                        options["dcp_prefix_source"] = source
                    if group == 1:
                        mapping = request.save_indexer_slot_mapping
                        base = request.save_slot_mapping_base
                        if not mapping:
                            mapping, base = request.indexer_slot_mapping, 0
                        if (
                            not mapping
                            or base is None
                            or base > start
                            or len(mapping[0]) < end - base
                        ):
                            raise ValueError(
                                "DCP save has incomplete replicated indexer slots"
                            )
                        options["slot_mapping"] = mapping[0][
                            start - base : end - base
                        ].to(
                            device=adapter.device,
                            dtype=torch.long,
                        )
                        options["slot_mapping_base"] = start
                    if group not in caches:
                        raise ValueError("DCP save has no registered group caches")
                except Exception as exc:
                    error = str(exc)
                result = store_group(
                    engine,
                    request,
                    group,
                    (caches or {}).get(group),
                    mask,
                    options,
                    input_error=error,
                )
                adapter._consume_completed_layerwise_store(request, group, True, result)
            if adapter._is_decode_window_save_request(request):
                context.setdefault("decode_window_saves", []).append(request)
            adapter._mark_prefill_committed(request)
        adapter._mark_initial_sparse_release_ready(request)
        adapter._maybe_lookup_unpin_for_request(request)


def seed_prefix(
    engine: Any, source: Any, objects: list, starts: list, ends: list
) -> int:
    """Reuse immutable canonical CPU planes; never read absent compact HBM.

    The importer validates source identity, mixed representations, offsets and
    allocation capacities. CPU copies are synchronous under an explicit child
    lease. Only rank zero calls this, before any rank submits destination DMA.
    """
    if source is None:
        return 0
    manifest = engine.compact_source_importer.source_manifest(source)
    if manifest.group_id != 0:
        raise ValueError("DCP save prefix belongs to the wrong group")
    stop = min(source.total_tokens, ends[-1])
    if stop <= starts[0]:
        return stop
    rows, hold = engine.compact_source_importer.borrow_source_tensors(source)
    try:
        if len(rows) != len(objects):
            raise ValueError("DCP prefix layer count differs from destination")
        # Build the interval intersections once, without rescanning prefix
        # history for every destination page and every layer.
        intersections = []
        index = bisect_right(manifest.chunk_ends, starts[0])
        for start, end in zip(starts, ends, strict=True):
            end, cursor, pieces = min(end, stop), start, []
            while cursor < end and index < len(manifest.chunk_starts):
                left, right = manifest.chunk_starts[index], manifest.chunk_ends[index]
                if not left <= cursor < right:
                    raise ValueError("DCP prefix has a hole before the resident tail")
                b = min(right, end)
                pieces.append((index, cursor, b))
                cursor = b
                if cursor == right:
                    index += 1
            if cursor < end:
                raise ValueError("DCP prefix has a hole before the resident tail")
            intersections.append(pieces)
        for layer, (srcs, dsts) in enumerate(zip(rows, objects, strict=True)):
            for page, start, end, pieces in zip(
                dsts, starts, ends, intersections, strict=True
            ):
                end = min(end, stop)
                if start >= end:
                    continue
                if isinstance(page, LayerPageMemoryObj):
                    raw = page.raw_data.narrow(
                        0, page.group_prefix_sum[layer], page.layer_size_bytes(layer)
                    )
                    dtype = page.get_dtypes()[layer]
                    dst = raw.view(dtype).view(-1)
                else:
                    dst = page.tensor.view(-1)
                capacity, rem = divmod(dst.numel(), 576)
                if rem or capacity < end - start:
                    raise ValueError("DCP prefix destination has invalid plane extent")
                for index, a, b in pieces:
                    src = srcs[index]
                    left, right = (
                        manifest.chunk_starts[index],
                        manifest.chunk_ends[index],
                    )
                    src_capacity, rem = divmod(src.numel(), 576)
                    if rem or src_capacity < right - left or src.dtype != dst.dtype:
                        raise ValueError("DCP prefix coverage or precision differs")
                    src = src.view(-1)
                    for width, prefix in ((512, 0), (64, 512)):
                        dst.narrow(
                            0, capacity * prefix + (a - start) * width, (b - a) * width
                        ).copy_(
                            src.narrow(
                                0,
                                src_capacity * prefix + (a - left) * width,
                                (b - a) * width,
                            ),
                        )
        return stop
    finally:
        hold.release("never_imported")
