# SPDX-License-Identifier: Apache-2.0
"""Native shared-slab startup contract; run with the Ascend extension built."""

# Standard
import ctypes
import uuid

# Third Party
import pytest

# First Party
import lmcache_ascend.c_ops as lmc_ops


@pytest.mark.parametrize("size", [2 << 20, 4 << 20])
def test_shared_slab_is_zeroed_registered_and_visible_to_attacher(size):
    name = f"/lmcache-startup-test-{uuid.uuid4().hex}"
    owner = lmc_ops.alloc_shm_pinned_ptr(size, name, [])
    passive = 0
    try:
        assert lmc_ops.get_device_ptr(owner, size) != 0
        assert ctypes.string_at(owner, size) == bytes(size)
        owner_view = (ctypes.c_ubyte * size).from_address(owner)
        owner_view[0] = 37
        owner_view[-1] = 99

        passive = lmc_ops.attach_shm_pinned_ptr(size, name, True)
        assert lmc_ops.get_device_ptr(passive, size) != 0
        passive_view = (ctypes.c_ubyte * size).from_address(passive)
        assert (passive_view[0], passive_view[-1]) == (37, 99)
        passive_view[1] = 42
        assert owner_view[1] == 42
    finally:
        try:
            if passive:
                lmc_ops.detach_shm_pinned_ptr(passive, size)
        finally:
            lmc_ops.free_shm_pinned_ptr(owner, size, name)

    with pytest.raises(RuntimeError, match="shm_open attach failed"):
        lmc_ops.attach_shm_pinned_ptr(size, name, True)
