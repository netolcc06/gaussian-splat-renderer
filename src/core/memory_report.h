// GPU memory accounting (Phase 5): which buffer/image holds how much device memory, owned
// by whom, created for what.
//
// Every createBuffer / createImage2D (core/resources.h) registers its allocation here and
// destroyBuffer / destroyImage removes it, so a report can be produced at any time:
//
//   owner      the subsystem the allocation belongs to: the innermost active MemoryScope
//              (e.g. "scene", "splat-tiles", "app") or, without a scope, the prefix of the
//              debug name before ':' ("radix sort: keys B" -> "radix sort").
//   name       the debug name (also visible in validation messages / RenderDoc / Nsight).
//   requested  the size the code asked for; allocated = VMA's allocation size (alignment
//              and padding included).
//   usage      buffer/image usage flags, memory kind (device-local, upload, readback).
//
// The report also carries VMA's per-heap budget (VK_EXT_memory_budget when available):
// what this process uses in each heap and what the OS lets it use. The difference between
// "registered" and the VMA heap usage is memory allocated outside our helpers (ImGui's
// font atlas and buffers, VMA block slack, driver-internal allocations).
#pragma once

#include "core/vk_common.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gsr {

class VulkanContext;

struct MemoryRecord {
    std::string owner;
    std::string name;
    VkDeviceSize requested = 0;
    VkDeviceSize allocated = 0;
    bool image = false;
    uint32_t usage = 0;     // VkBufferUsageFlags or VkImageUsageFlags
    uint32_t memoryKind = 0;  // 0 device-local, 1 upload (host-visible), 2 readback (host-visible)
    uint64_t serial = 0;    // creation order
};

class MemoryRegistry {
public:
    MemoryRegistry() = default;
    // Copyable (VulkanContext::destroy() resets the context by assignment); the mutex is not copied.
    MemoryRegistry(const MemoryRegistry& other) { *this = other; }
    MemoryRegistry& operator=(const MemoryRegistry& other) {
        if (this == &other) return *this;
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_records = other.m_records;
        m_owners = other.m_owners;
        m_serial = other.m_serial;
        return *this;
    }

    void add(uint64_t handle, MemoryRecord record);
    void remove(uint64_t handle);
    std::vector<MemoryRecord> snapshot() const;  // in creation order

    // Owner stack (see MemoryScope).
    void pushOwner(std::string owner);
    void popOwner();
    std::string currentOwner(const char* debugName) const;

private:
    mutable std::mutex m_mutex;
    std::unordered_map<uint64_t, MemoryRecord> m_records;
    std::vector<std::string> m_owners;
    uint64_t m_serial = 0;
};

// RAII: allocations made while it is alive belong to `owner`.
class MemoryScope {
public:
    MemoryScope(const VulkanContext& ctx, std::string owner);
    ~MemoryScope();
    MemoryScope(const MemoryScope&) = delete;
    MemoryScope& operator=(const MemoryScope&) = delete;

private:
    MemoryRegistry* m_registry;
};

struct MemoryHeapInfo {
    uint32_t index = 0;
    bool deviceLocal = false;
    VkDeviceSize size = 0;         // heap size
    VkDeviceSize budget = 0;       // what the process may use (VK_EXT_memory_budget, else an estimate)
    VkDeviceSize usage = 0;        // what the process uses (all allocations, incl. outside VMA)
    VkDeviceSize vmaBlocks = 0;    // VMA device memory blocks
    VkDeviceSize vmaAllocations = 0;  // VMA allocations inside them
};

struct MemoryOwnerTotal {
    std::string owner;
    VkDeviceSize requested = 0, allocated = 0;
    uint32_t count = 0;
};

struct MemoryReport {
    std::vector<MemoryRecord> records;   // creation order
    std::vector<MemoryOwnerTotal> owners;  // sorted by allocated size, descending
    VkDeviceSize totalRequested = 0, totalAllocated = 0;      // all registered allocations
    VkDeviceSize deviceLocalAllocated = 0;                    // registered, device-local only
    std::vector<MemoryHeapInfo> heaps;
    bool budgetExtension = false;
};

MemoryReport buildMemoryReport(const VulkanContext& ctx);
std::string bufferUsageString(uint32_t usage);  // "storage|transfer_dst|..."
std::string imageUsageString(uint32_t usage);
const char* memoryKindName(uint32_t kind);
inline double toMiB(VkDeviceSize bytes) { return double(bytes) / (1024.0 * 1024.0); }

}  // namespace gsr
