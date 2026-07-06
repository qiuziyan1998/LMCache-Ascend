#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

/*
 * These following APIs are called directly in LMCache,
 * therefore we assume the ptrs management are done by the python program for
 * now.
 */

uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, std::size_t size);

uintptr_t alloc_shm_pinned_ptr(std::size_t size, const std::string &shm_name);

uintptr_t attach_shm_pinned_ptr(
    std::size_t size, const std::string &shm_name, bool writable);

void free_shm_pinned_ptr(
    uintptr_t ptr, std::size_t size, const std::string &shm_name);

void detach_shm_pinned_ptr(uintptr_t ptr, std::size_t size);

void unlink_shm(const std::string &shm_name);
