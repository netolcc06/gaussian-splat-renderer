// GpuRadixSort (Phase 3): a stable LSD radix sort of 32-bit (key, value) pairs in GLSL
// compute, written for teaching (no library). Shaders: shaders/radix_*.comp; the
// algorithm is documented in shaders/radix_sort_common.glsl and docs/notes/phase3.md.
//
//   record():  setup (1 thread: count -> indirect dispatch args)
//              4 x [ histogram -> scan -> scatter ]   (8-bit digits, shift 0, 8, 16, 24)
//
// The number of keys lives in a GPU buffer (e.g. the atomic counter of a stream
// compaction), so the CPU never needs to know it: the per-block dispatches are
// vkCmdDispatchIndirect with arguments written by the setup pass.
//
// The caller owns the key/value buffers ("A"); the sort owns a second pair ("B") and
// ping-pongs A -> B -> A -> B -> A, so after the 4 passes the sorted data is back in
// the caller's buffers. Ties keep their input order (stable), hence the result equals
// DepthSorter::radixSortPairs on the same input, bit for bit.
//
// Phase 4 extensions (the defaults keep the Phase 3 behaviour exactly):
//   - keyBits: only the passes over the low `keyBits` bits are recorded (e.g. 13 bits of
//     tile id = 2 passes). The pass count is rounded UP TO EVEN so that the ping-pong
//     ends in the caller's buffers; an extra pass sorts on an all-zero digit, which is
//     the identity permutation (stable), so the result is unchanged.
//   - 64-bit keys: init() with a `keysHi` buffer sorts (hi << 32 | lo) keys stored as two
//     words (shaders built with -DGSR_KEY64); passes with shift >= 32 use the high word.
//   - optional per-pass profiler scopes and a scope label, so that two sorts in one frame
//     are timed separately.
#pragma once

#include "core/descriptors.h"
#include "core/resources.h"

#include <array>

namespace gsr {

class VulkanContext;
class GpuProfiler;

class GpuRadixSort {
public:
    // Scatter step implementation (the result is identical; see shaders/radix_scatter*.comp):
    //   Classic  (default, reference) workgroup-wide rounds of 256 keys, 4 workgroup
    //            barriers per round, every key written straight to its global slot
    //            (radix_scatter.comp)
    //   Reorder  each subgroup ranks its own contiguous 256 keys with subgroup-only sync,
    //            the block is sorted locally in shared memory, then written out in
    //            contiguous per-digit runs (coalesced) (radix_scatter_reorder.comp)
    enum class Variant : uint32_t { Classic = 0, Reorder = 1 };

    static constexpr uint32_t kRadixBits = 8;
    static constexpr uint32_t kRadixSize = 1u << kRadixBits;
    static constexpr uint32_t kPasses = 32 / kRadixBits;  // passes for full 32-bit keys
    static constexpr uint32_t kBlockSize = 2048;  // keys per workgroup (SORT_BLOCK_SIZE)

    // Passes recorded for keys with `keyBits` significant low bits: ceil(keyBits / 8)
    // rounded up to an even number (ping-pong parity), at least 2.
    static uint32_t passCount(uint32_t keyBits);

    // keys/values: >= maxCount uints each, STORAGE usage, sorted in place.
    // countBuffer: holds the number of keys (uint) at byte offset countOffset (multiple of
    // 4); it is clamped to maxCount on the GPU. All of them stay owned by the caller.
    // keysHi (optional): the high words of 64-bit keys (>= maxCount uints); enables the
    // 64-bit mode, in which record() accepts keyBits up to 64.
    void init(VulkanContext& ctx, uint32_t maxCount, VkBuffer keys, VkBuffer values, VkBuffer countBuffer,
              uint32_t countOffset, VkBuffer keysHi = VK_NULL_HANDLE);
    void destroy(VulkanContext& ctx);

    struct RecordOptions {
        uint32_t keyBits = 32;       // significant low bits of the keys (<= 32, or <= 64 in 64-bit mode)
        const char* label = "sort";  // profiler scopes "<label>: setup", "<label>: histogram", ...
        bool perPassScopes = false;  // also one scope per pass: "<label>: pass <i> (bits a-b)"
    };

    // Records the whole sort. Synchronization contract:
    //   before: keys, values and the count must have been written by COMPUTE_SHADER (or
    //           made available to it by a caller barrier, e.g. after a transfer). record()
    //           starts with a COMPUTE -> COMPUTE barrier, which also orders this sort after
    //           the previous one's reads of its scratch buffers (write-after-read).
    //   after:  the sorted keys/values were last written by COMPUTE_SHADER (SHADER_WRITE);
    //           the caller adds a barrier before reading them anywhere.
    // `profiler` (optional) gets one scope per step: "sort: setup", "sort: histogram",
    // "sort: scan", "sort: scatter" (the last three summed over the passes).
    void record(VkCommandBuffer cmd, GpuProfiler* profiler) const { record(cmd, profiler, RecordOptions{}); }
    void record(VkCommandBuffer cmd, GpuProfiler* profiler, const RecordOptions& options) const;

    void setVariant(Variant v) { m_variant = v; }
    Variant variant() const { return m_variant; }

    uint32_t maxCount() const { return m_maxCount; }
    bool keys64() const { return m_keys64; }
    VkDeviceSize scratchBytes() const;  // device memory owned by the sort

private:
    uint32_t m_maxCount = 0;
    uint32_t m_countOffset = 0;
    uint32_t m_maxBlocks = 0;
    bool m_keys64 = false;

    Buffer m_keysB, m_valuesB;  // ping-pong partners of the caller's buffers
    Buffer m_keysHiB;           // 64-bit mode: partner of the caller's high-word buffer
    Buffer m_blockHist;         // kRadixSize * maxBlocks uints, digit-major
    Buffer m_digitCounts;       // kRadixSize uints
    Buffer m_args;              // SortArgs (indirect dispatch arguments + count)

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    DescriptorPool m_pool;
    std::array<VkDescriptorSet, 2> m_sets{};  // [0]: A -> B, [1]: B -> A
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_setup = VK_NULL_HANDLE;
    VkPipeline m_histogram = VK_NULL_HANDLE;
    VkPipeline m_scan = VK_NULL_HANDLE;
    VkPipeline m_scatter = VK_NULL_HANDLE;
    VkPipeline m_scatterReorder = VK_NULL_HANDLE;
    Variant m_variant = Variant::Classic;

public:
    // Phase 6 `--pipeline-stats`.
    std::vector<std::pair<std::string, VkPipeline>> pipelines(const std::string& label) const {
        return {{label + ": radix setup", m_setup}, {label + ": radix histogram", m_histogram},
                {label + ": radix scan", m_scan}, {label + ": radix scatter (classic)", m_scatter},
                {label + ": radix scatter (reorder)", m_scatterReorder}};
    }
};

}  // namespace gsr
