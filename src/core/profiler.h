// GpuProfiler: named GPU timing scopes using timestamp queries.
//
// Each frame in flight owns a slice of one VkQueryPool. A scope writes two timestamps
// (begin/end) into that slice. Results are read back when the frame slot comes
// around again (its timeline value has been reached, so the queries are complete),
// which means GPU timings lag the CPU by `framesInFlight` frames but never stall it.
//
// Usage per frame:
//   profiler.beginFrame(slot, frameNumber);    // after waiting for this slot's previous frame
//   { auto s = profiler.scope(cmd, "cube"); ...record commands... }
#pragma once

#include "core/timing_stats.h"
#include "core/vk_common.h"

#include <string>
#include <vector>

namespace gsr {

class VulkanContext;

class GpuProfiler {
public:
    using Stats = TimingStats;

    static constexpr uint32_t kMaxScopesPerFrame = 96;  // Phase 4: per-pass sort scopes

    void init(const VulkanContext& ctx, uint32_t framesInFlight);
    void destroy();

    // Reads back the results recorded the last time `frameSlot` was used and resets
    // its queries. Call once per frame, after the slot's previous submission completed.
    // `frameNumber` is a monotonically increasing frame counter (see resetAccumulators).
    void beginFrame(uint32_t frameSlot, uint64_t frameNumber);

    // Begin/end a named scope. Scopes may nest. Prefer the RAII `scope()` helper.
    // Several scopes with the same name in one frame add up to one sample per frame.
    uint32_t beginScope(VkCommandBuffer cmd, const char* name);
    void endScope(VkCommandBuffer cmd, uint32_t scopeId);

    class Scope {
    public:
        Scope(GpuProfiler& p, VkCommandBuffer cmd, const char* name)
            : m_profiler(p), m_cmd(cmd), m_id(p.beginScope(cmd, name)) {}
        ~Scope() { m_profiler.endScope(m_cmd, m_id); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        GpuProfiler& m_profiler;
        VkCommandBuffer m_cmd;
        uint32_t m_id;
    };
    [[nodiscard]] Scope scope(VkCommandBuffer cmd, const char* name) { return Scope(*this, cmd, name); }

    // Reads every pending slot. Call after vkDeviceWaitIdle (e.g. at the end of a benchmark).
    void collectAll();
    // Clears min/max/sum/count and only accumulates frames with number >= firstFrame
    // from now on (frames still in flight from before are ignored).
    void resetAccumulators(uint64_t firstFrame);

    const std::vector<Stats>& stats() const { return m_stats; }

private:
    struct PendingScope {
        uint32_t statIndex;
        uint32_t firstQuery;  // begin = firstQuery, end = firstQuery + 1
    };
    struct FrameSlot {
        std::vector<PendingScope> scopes;
        uint32_t queriesUsed = 0;
        uint64_t frameNumber = 0;
    };

    uint32_t statIndexFor(const char* name);
    void collect(uint32_t frameSlot);

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueryPool m_pool = VK_NULL_HANDLE;
    double m_nsPerTick = 1.0;
    uint64_t m_validMask = ~0ull;
    std::vector<FrameSlot> m_slots;
    uint32_t m_currentSlot = 0;
    uint64_t m_accumulateFrom = 0;
    std::vector<Stats> m_stats;
};

}  // namespace gsr
