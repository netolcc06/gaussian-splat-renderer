// Timing statistics shared by the GPU profiler (timestamp queries) and the CPU profiler
// (std::chrono): a rolling average for the UI plus min/max/sum accumulators for benchmarks.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace gsr {

struct TimingStats {
    static constexpr size_t kHistory = 120;

    std::string name;
    double lastMs = 0.0;
    double avgMs = 0.0;  // rolling average over the last kHistory samples (for the UI)
    // Accumulators since the last reset (for benchmarks).
    double sumMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    uint64_t count = 0;
    // Rolling-window storage.
    std::vector<double> history;
    size_t historyNext = 0;

    void addSample(double ms, bool accumulate) {
        lastMs = ms;
        if (history.size() < kHistory) {
            history.push_back(ms);
        } else {
            history[historyNext] = ms;
            historyNext = (historyNext + 1) % kHistory;
        }
        double sum = 0.0;
        for (double v : history) sum += v;
        avgMs = sum / static_cast<double>(history.size());
        if (!accumulate) return;
        minMs = count == 0 ? ms : std::min(minMs, ms);
        maxMs = count == 0 ? ms : std::max(maxMs, ms);
        sumMs += ms;
        count++;
    }
    void resetAccumulators() {
        sumMs = minMs = maxMs = 0.0;
        count = 0;
    }
};

// Named CPU timings, e.g. the per-frame CPU depth sort of the splat renderer. Samples
// are added by name; the app resets the accumulators when a benchmark's measured
// frames start (together with the GPU profiler).
class CpuProfiler {
public:
    void add(const char* name, double ms) {
        for (TimingStats& s : m_stats)
            if (s.name == name) return s.addSample(ms, true);
        TimingStats s;
        s.name = name;
        s.addSample(ms, true);
        m_stats.push_back(std::move(s));
    }
    void resetAccumulators() {
        for (TimingStats& s : m_stats) s.resetAccumulators();
    }
    const std::vector<TimingStats>& stats() const { return m_stats; }

    // RAII: measures the scope's wall time and adds it under `name`.
    class Scope {
    public:
        Scope(CpuProfiler& p, const char* name) : m_profiler(p), m_name(name), m_start(Clock::now()) {}
        ~Scope() {
            m_profiler.add(m_name, std::chrono::duration<double, std::milli>(Clock::now() - m_start).count());
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        using Clock = std::chrono::steady_clock;
        CpuProfiler& m_profiler;
        const char* m_name;
        Clock::time_point m_start;
    };
    [[nodiscard]] Scope scope(const char* name) { return Scope(*this, name); }

private:
    std::vector<TimingStats> m_stats;
};

}  // namespace gsr
