#include "core/memory_report.h"

#include "core/context.h"

#include <algorithm>
#include <map>

namespace gsr {

void MemoryRegistry::add(uint64_t handle, MemoryRecord record) {
    std::lock_guard<std::mutex> lock(m_mutex);
    record.serial = m_serial++;
    m_records[handle] = std::move(record);
}

void MemoryRegistry::remove(uint64_t handle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_records.erase(handle);
}

std::vector<MemoryRecord> MemoryRegistry::snapshot() const {
    std::vector<MemoryRecord> out;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        out.reserve(m_records.size());
        for (const auto& [handle, r] : m_records) out.push_back(r);
    }
    std::sort(out.begin(), out.end(), [](const MemoryRecord& a, const MemoryRecord& b) { return a.serial < b.serial; });
    return out;
}

void MemoryRegistry::pushOwner(std::string owner) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_owners.push_back(std::move(owner));
}

void MemoryRegistry::popOwner() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_owners.empty()) m_owners.pop_back();
}

std::string MemoryRegistry::currentOwner(const char* debugName) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_owners.empty()) return m_owners.back();
    if (debugName) {
        const std::string name(debugName);
        const size_t colon = name.find(':');
        if (colon != std::string::npos && colon > 0) return name.substr(0, colon);
        return name;
    }
    return "other";
}

MemoryScope::MemoryScope(const VulkanContext& ctx, std::string owner) : m_registry(&ctx.memory) {
    m_registry->pushOwner(std::move(owner));
}

MemoryScope::~MemoryScope() { m_registry->popOwner(); }

MemoryReport buildMemoryReport(const VulkanContext& ctx) {
    MemoryReport report;
    report.records = ctx.memory.snapshot();
    std::map<std::string, MemoryOwnerTotal> owners;
    for (const MemoryRecord& r : report.records) {
        MemoryOwnerTotal& o = owners[r.owner];
        o.owner = r.owner;
        o.requested += r.requested;
        o.allocated += r.allocated;
        ++o.count;
        report.totalRequested += r.requested;
        report.totalAllocated += r.allocated;
        if (r.memoryKind == 0) report.deviceLocalAllocated += r.allocated;
    }
    for (auto& [name, o] : owners) report.owners.push_back(o);
    std::sort(report.owners.begin(), report.owners.end(),
              [](const MemoryOwnerTotal& a, const MemoryOwnerTotal& b) { return a.allocated > b.allocated; });

    if (ctx.allocator) {
        const VkPhysicalDeviceMemoryProperties* props = nullptr;
        vmaGetMemoryProperties(ctx.allocator, &props);
        std::vector<VmaBudget> budgets(props->memoryHeapCount);
        vmaGetHeapBudgets(ctx.allocator, budgets.data());
        for (uint32_t h = 0; h < props->memoryHeapCount; ++h) {
            MemoryHeapInfo info;
            info.index = h;
            info.deviceLocal = (props->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            info.size = props->memoryHeaps[h].size;
            info.budget = budgets[h].budget;
            info.usage = budgets[h].usage;
            info.vmaBlocks = budgets[h].statistics.blockBytes;
            info.vmaAllocations = budgets[h].statistics.allocationBytes;
            report.heaps.push_back(info);
        }
    }
    report.budgetExtension = ctx.memoryBudgetEnabled;
    return report;
}

namespace {

std::string flagsString(uint32_t flags, const std::pair<uint32_t, const char*>* names, size_t count) {
    std::string out;
    for (size_t i = 0; i < count; ++i) {
        if (!(flags & names[i].first)) continue;
        if (!out.empty()) out += '|';
        out += names[i].second;
    }
    return out.empty() ? "-" : out;
}

}  // namespace

std::string bufferUsageString(uint32_t usage) {
    static const std::pair<uint32_t, const char*> kNames[] = {
        {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "storage"},
        {VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "uniform"},
        {VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "vertex"},
        {VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "index"},
        {VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "indirect"},
        {VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "transfer_src"},
        {VK_BUFFER_USAGE_TRANSFER_DST_BIT, "transfer_dst"},
        {VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, "device_address"},
    };
    return flagsString(usage, kNames, std::size(kNames));
}

std::string imageUsageString(uint32_t usage) {
    static const std::pair<uint32_t, const char*> kNames[] = {
        {VK_IMAGE_USAGE_STORAGE_BIT, "storage"},
        {VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "color_attachment"},
        {VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth_attachment"},
        {VK_IMAGE_USAGE_SAMPLED_BIT, "sampled"},
        {VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "transfer_src"},
        {VK_IMAGE_USAGE_TRANSFER_DST_BIT, "transfer_dst"},
    };
    return "image:" + flagsString(usage, kNames, std::size(kNames));
}

const char* memoryKindName(uint32_t kind) {
    switch (kind) {
        case 0: return "device-local";
        case 1: return "upload";
        case 2: return "readback";
        default: return "?";
    }
}

}  // namespace gsr
