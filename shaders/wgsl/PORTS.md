# GLSL -> WGSL port manifest

The web build's shaders (`shaders/wgsl/`) are hand-written ports of the native GLSL
(`docs/web/DESIGN.md` section 4). This table records, per GLSL source, the WGSL file that ports it
and the SHA-256 (first 16 hex digits, LF line endings) of the GLSL file when its port was last
reviewed. `tools/web/check_shader_sync.py` (run by `run_web_checks.mjs --test w2`) recomputes the
hashes and fails when a GLSL file changed after its port was reviewed: update the WGSL (or, if the
change does not matter on the web, only the hash: `python tools/web/check_shader_sync.py --update`).
Behavioural equality is checked by the in-page GPU tests and the reference images, not by text.

| GLSL source | WGSL port | GLSL SHA-256/16 | deliberate differences |
|---|---|---|---|
| `scene_decode.glsl` | `scene_decode.wgsl` | `a06c1caff3f1b5e8` | W1. Specialization constants -> `override`; no out parameters; the `cb` codebook path is not ported. |
| `points.vert` | `points.wgsl` | `f5fb8d5b823439a6` | W1. Clip y negated (Vulkan projection kept); flat colour. |
| `points.frag` | `points.wgsl` | `a2b5ff22758e1ec1` | W1. |
| `splat_common.glsl` | `splat_common.wgsl` | `6563e480b05f6f42` | No `precise` in WGSL (see `coverage_spans.wgsl`); `conicAndRadius` returns a struct. |
| `splat_preprocess.comp` | `splat_preprocess.wgsl` | `2a0c9a93786f93a7` | GSR_TILES build only. Stores the coverage spans (new); cull counters portable (shared atomics) or subgroup ballots; full 48-B Splat only (no p32/p24, no StopThePop parameters); depth may differ from the CPU by an ulp (FMA). |
| `tile_common.glsl` | `tile_common.wgsl` | `0139fdcce9612c0a` | The control block is indexed as u32 words (WGSL cannot mix atomic and plain reads of a struct); list dispatch arguments also in their own buffer. |
| `tile_coverage.glsl` | `tile_coverage.wgsl` | `b4318d394d3e8fd5` | rect + tight only (no cutoff-only); evaluated once per splat, in the preprocess. |
| `workgroup_scan.glsl` | `workgroup_scan.wgsl` | `d256ced6a2f6d8bf` | Portable Hillis-Steele scan (default) + the subgroup scan (GSR_SUBGROUPS); an 8-counter variant for the radix rank; 2D workgroup linearization helper. |
| `scan_common.glsl` | `scan.wgsl` | `580ccbae80bae56d` | One module, three entry points; count source and total in one read_write binding; uniform count via `workgroupUniformLoad`. |
| `scan_reduce.comp` | `scan.wgsl` | `9b6fadd444712cce` | Entry point `scan_reduce`. |
| `scan_blocks.comp` | `scan.wgsl` | `a7b0a6762d900dcc` | Entry point `scan_blocks`. |
| `scan_downsweep.comp` | `scan.wgsl` | `8ca0d9aec6c75007` | Entry point `scan_downsweep`. |
| `radix_sort_common.glsl` | `radix_sort.wgsl` | `9ca4adcd85157243` | 4-bit digits, 1024-key blocks (16 KB workgroup memory, no subgroups needed); `shift` from a dynamic-offset uniform; no KEY64 variant. |
| `radix_setup.comp` | `radix_sort.wgsl` | `cd5b686037a0c541` | Entry point `radix_setup`; workgroup count split into a 2D grid above the per-dimension limit. |
| `radix_histogram.comp` | `radix_sort.wgsl` | `ce216d662a9ce3f7` | Entry point `radix_histogram`. |
| `radix_scan.comp` | `radix_sort.wgsl` | `6ab9996f31810ea8` | Entry point `radix_scan` (16 workgroups). |
| `radix_scatter.comp` | `radix_sort.wgsl` | `0eb95260b49c107f` | Entry point `radix_scatter`: stable local rank from packed per-thread digit counters and one workgroup scan (instead of subgroup match ballots). `radix_scatter_reorder.comp` is not ported. |
| `tile_compact.comp` | `tile_compact.wgsl` | `ef1dd1fe4c85aeca` | Full Splat only. |
| `tile_setup.comp` | `tile_setup.wgsl` | `c6ce3ef82fece535` | 2D-clamped indirect grid; arguments also written to a separate buffer (WebGPU usage rules). |
| `tile_duplicate.comp` | `tile_duplicate.wgsl` | `0af5200e6eea08e9` | Binning + per-entry build only; enumerates the STORED spans (integers), no float coverage math. |
| `tile_ranges.comp` | `tile_ranges.wgsl` | `2e4b9e3346a221e2` | 2D workgroup linearization. |
| `tile_render.comp` | `tile_render.wgsl` | `a9ce695c3a9b32cf` | Vote at the end of each batch, read with `workgroupUniformLoad`; portable vote (shared atomics) or subgroup ballot; rgba16float (+ rgba32float test build) only; no stats / StopThePop / packed / radius-bound / fp16 / AoS variants. |
| `(new)` | `coverage_spans.wgsl` | `-` | Web only: the integer coverage record (DESIGN.md section 3, option d); C++ twin `src/webgpu/coverage_spans.*`. |
| `(new)` | `gradient.wgsl` | `-` | W0 test pattern (counterpart of `gradient.comp`, not a port). |
| `(new)` | `resolve.wgsl` | `-` | Replaces `vkCmdBlitImage` (rgba16float -> canvas / rgba8 capture). |
| `(new)` | `fullscreen.wgsl` | `-` | Fullscreen triangle for the resolve. |
| `(new)` | `decode_test.wgsl` | `-` | W1 self-test of the scene decoder. |
