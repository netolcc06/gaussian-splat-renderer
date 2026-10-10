#include "core/profiler.h"

#include "core/context.h"

#include <algorithm>
#include <cstring>

namespace gsr {

void GpuProfiler::init(const VulkanContext& ctx, uint32_t framesInFlight) {
    m_device = ctx.device;
    // timestampPeriod = nanoseconds per timestamp tick on this GPU.
    m_nsPerTick = ctx.properties.limits.timestampPeriod;

    // Queues may implement fewer than 64 valid timestamp bits; mask the rest off.
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &familyCount, families.data());
    const uint32_t validBits = families[ctx.queueFamily].timestampValidBits;
    m_validMask = validBits >= 64 ? ~0ull : ((1ull << validBits) - 1);

    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = framesInFlight * kMaxScopesPerFrame * 2;
    VK_CHECK(vkCreateQueryPool(m_device, &info, nullptr, &m_pool));
    // Queries must be reset before their first use. hostQueryReset (Vulkan 1.2) lets
    // us do that from the CPU instead of recording vkCmdResetQueryPool.
    vkResetQueryPool(m_device, m_pool, 0, info.queryCount);

    m_slots.assign(framesInFlight, FrameSlot{});
}

void GpuProfiler::destroy() {
    if (m_pool) vkDestroyQueryPool(m_device, m_pool, nullptr);
    m_pool = VK_NULL_HANDLE;
    m_slots.clear();
}

uint32_t GpuProfiler::statIndexFor(const char* name) {
    for (uint32_t i = 0; i < m_stats.size(); ++i)
        if (m_stats[i].name == name) return i;
    Stats s;
    s.name = name;
    s.history.reserve(TimingStats::kHistory);
    m_stats.push_back(std::move(s));
    return static_cast<uint32_t>(m_stats.size() - 1);
}

void GpuProfiler::beginFrame(uint32_t frameSlot, uint64_t frameNumber) {
    collect(frameSlot);
    m_currentSlot = frameSlot;
    m_slots[frameSlot].frameNumber = frameNumber;
}

uint32_t GpuProfiler::beginScope(VkCommandBuffer cmd, const char* name) {
    FrameSlot& slot = m_slots[m_currentSlot];
    if (slot.queriesUsed + 2 > kMaxScopesPerFrame * 2) return UINT32_MAX;  // out of queries: ignore

    const uint32_t first = m_currentSlot * kMaxScopesPerFrame * 2 + slot.queriesUsed;
    slot.queriesUsed += 2;
    slot.scopes.push_back({statIndexFor(name), first});

    // ALL_COMMANDS: the timestamp is written once all previously submitted work has
    // finished every stage. That brackets the scope's work as tightly as Vulkan allows.
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_pool, first);
    return static_cast<uint32_t>(slot.scopes.size() - 1);
}

void GpuProfiler::endScope(VkCommandBuffer cmd, uint32_t scopeId) {
    if (scopeId == UINT32_MAX) return;
    const PendingScope& scope = m_slots[m_currentSlot].scopes[scopeId];
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_pool, scope.firstQuery + 1);
}

void GpuProfiler::collect(uint32_t frameSlot) {
    FrameSlot& slot = m_slots[frameSlot];
    if (slot.queriesUsed == 0) return;

    const uint32_t base = frameSlot * kMaxScopesPerFrame * 2;
    std::vector<uint64_t> ticks(slot.queriesUsed);
    // The frame that wrote these queries has completed (the caller waited on its
    // timeline value), so WAIT returns immediately; it is there as a safety net.
    VK_CHECK(vkGetQueryPoolResults(m_device, m_pool, base, slot.queriesUsed,
                                   ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    // Scopes with the same name in one frame (e.g. one per radix sort pass) are summed
    // into a single per-frame sample.
    std::vector<double> frameMs(m_stats.size(), -1.0);
    for (const PendingScope& scope : slot.scopes) {
        const uint64_t begin = ticks[scope.firstQuery - base] & m_validMask;
        const uint64_t end = ticks[scope.firstQuery - base + 1] & m_validMask;
        const double ms = static_cast<double>((end - begin) & m_validMask) * m_nsPerTick * 1e-6;
        double& acc = frameMs[scope.statIndex];
        acc = acc < 0.0 ? ms : acc + ms;
    }
    for (size_t i = 0; i < frameMs.size(); ++i)
        if (frameMs[i] >= 0.0) m_stats[i].addSample(frameMs[i], slot.frameNumber >= m_accumulateFrom);

    vkResetQueryPool(m_device, m_pool, base, slot.queriesUsed);
    slot.scopes.clear();
    slot.queriesUsed = 0;
}

void GpuProfiler::collectAll() {
    for (uint32_t i = 0; i < m_slots.size(); ++i) collect(i);
}

void GpuProfiler::resetAccumulators(uint64_t firstFrame) {
    m_accumulateFrom = firstFrame;
    for (Stats& s : m_stats) s.resetAccumulators();
}

}  // namespace gsr
