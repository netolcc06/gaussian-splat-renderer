// parallelChunks: a minimal fork-join loop for the load-time work of the scene library
// (encoding, k-means). Splits [0, count) into fixed chunks handed out to `threads` threads
// (0 = all hardware threads); fn(begin, end) must only write data owned by its range, so the
// result never depends on the thread count.
//
// The web build (Emscripten) is single-threaded (no pthreads, docs/web/DESIGN.md section 2.4):
// there hardwareThreads() is 1, so every loop here and in the PLY loader runs inline.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace gsr {

// Threads a load-time loop may use: all hardware threads natively, 1 on the web.
inline uint32_t hardwareThreads() {
#ifdef __EMSCRIPTEN__
    return 1;
#else
    return std::max(1u, std::thread::hardware_concurrency());
#endif
}

template <typename Fn>
void parallelChunks(uint32_t count, uint32_t threads, Fn&& fn, uint32_t chunk = 4096) {
    if (count == 0) return;
    const uint32_t chunks = (count + chunk - 1) / chunk;
    if (threads == 0) threads = hardwareThreads();
#ifdef __EMSCRIPTEN__
    threads = 1;  // even when asked for more: the web build has no threads
#endif
    threads = std::max(1u, std::min(threads, chunks));
    if (threads == 1) {
        fn(0u, count);
        return;
    }
    std::atomic<uint32_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const uint32_t c = next.fetch_add(1);
            if (c >= chunks) return;
            fn(c * chunk, std::min(count, (c + 1) * chunk));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (uint32_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

}  // namespace gsr
