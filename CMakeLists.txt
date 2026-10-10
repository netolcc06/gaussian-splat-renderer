cmake_minimum_required(VERSION 3.24)
project(GaussianSplatRenderer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put every executable/DLL of every configuration under build/bin/<Config>/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The web build (Emscripten toolchain, `cmake --preset web`) shares the Vulkan-free libraries
# below and then continues in cmake/web.cmake (WebGPU backend + page). Everything Vulkan, GLFW
# and ImGui is native only. See docs/web/DESIGN.md.
if(EMSCRIPTEN)
  # The scene library throws (PlyError, runtime_error): native wasm exception handling.
  add_compile_options(-fwasm-exceptions)
  add_link_options(-fwasm-exceptions)
endif()

# ---------------------------------------------------------------------------
# Vulkan SDK
# ---------------------------------------------------------------------------
# VULKAN_SDK is set machine-wide but is not always visible in every shell.
# Fall back to the known install location so configuration works anyway.
if(NOT EMSCRIPTEN)
if(NOT DEFINED ENV{VULKAN_SDK} AND EXISTS "C:/VulkanSDK/1.4.363.0")
  set(ENV{VULKAN_SDK} "C:/VulkanSDK/1.4.363.0")
endif()
find_package(Vulkan REQUIRED COMPONENTS glslc)
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies (FetchContent, pinned versions)
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
  GIT_SHALLOW    TRUE)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.9b
  GIT_SHALLOW    TRUE)

# The SDK does not ship vk_mem_alloc.h in this installation, so fetch VMA.
FetchContent_Declare(vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG        v3.3.0
  GIT_SHALLOW    TRUE)

FetchContent_Declare(json
  URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)

# stb has no release tags; pin an exact commit.
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20)

if(EMSCRIPTEN)
  FetchContent_MakeAvailable(glm json stb)
else()
  FetchContent_MakeAvailable(glfw glm json imgui vma stb)
endif()

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

if(NOT EMSCRIPTEN)  # ImGui, VMA and the GLSL shaders are native only (up to the web branch below).
# Dear ImGui has no CMake build of its own: compile the core + GLFW/Vulkan backends.
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

add_library(vma INTERFACE)
target_include_directories(vma SYSTEM INTERFACE ${vma_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Shaders: GLSL -> SPIR-V with glslc, as part of the build
# ---------------------------------------------------------------------------
set(SHADER_SOURCES
  shaders/gradient.comp
  shaders/cube.vert
  shaders/cube.frag
  shaders/points.vert
  shaders/points.frag
  shaders/splat_preprocess.comp
  shaders/splat.vert
  shaders/splat.frag
  shaders/radix_setup.comp
  shaders/radix_histogram.comp
  shaders/radix_scan.comp
  shaders/radix_scatter.comp
  shaders/radix_scatter_reorder.comp
  shaders/scan_reduce.comp
  shaders/scan_blocks.comp
  shaders/scan_downsweep.comp
  shaders/tile_compact.comp
  shaders/tile_setup.comp
  shaders/tile_duplicate.comp
  shaders/tile_ranges.comp
  shaders/tile_render.comp)
# Included by the shaders above (tracked through glslc -MD depfiles).
set(SHADER_HEADERS shaders/splat_common.glsl shaders/radix_sort_common.glsl shaders/workgroup_scan.glsl
  shaders/scan_common.glsl shaders/tile_common.glsl shaders/scene_decode.glsl shaders/render_splat.glsl
  shaders/tile_coverage.glsl)
# Variants: one source compiled again with preprocessor defines ("source|output|DEFINES").
set(SHADER_VARIANTS
  "shaders/splat_preprocess.comp|splat_preprocess_compact.comp.spv|GSR_COMPACT"
  "shaders/splat.vert|splat_indirect.vert.spv|GSR_INDIRECT_COUNT"
  "shaders/splat_preprocess.comp|splat_preprocess_tiles.comp.spv|GSR_TILES"
  "shaders/radix_histogram.comp|radix_histogram_key64.comp.spv|GSR_KEY64"
  "shaders/radix_scatter.comp|radix_scatter_key64.comp.spv|GSR_KEY64"
  "shaders/radix_scatter_reorder.comp|radix_scatter_reorder_key64.comp.spv|GSR_KEY64"
  "shaders/tile_duplicate.comp|tile_duplicate_global.comp.spv|GSR_TILE_GLOBAL64"
  "shaders/tile_duplicate.comp|tile_duplicate_per_entry.comp.spv|GSR_DUPLICATE_PER_ENTRY"
  "shaders/tile_duplicate.comp|tile_duplicate_global_per_entry.comp.spv|GSR_TILE_GLOBAL64|GSR_DUPLICATE_PER_ENTRY"
  "shaders/tile_render.comp|tile_render_rgba32f.comp.spv|GSR_OUT_RGBA32F"
  "shaders/tile_render.comp|tile_render_rgba8.comp.spv|GSR_OUT_RGBA8"
  "shaders/tile_render.comp|tile_render_stats.comp.spv|GSR_TILE_STATS"
  "shaders/tile_render.comp|tile_render_packed.comp.spv|GSR_PACKED_SPLAT"
  "shaders/tile_render.comp|tile_render_packed_rgba32f.comp.spv|GSR_PACKED_SPLAT|GSR_OUT_RGBA32F"
  "shaders/tile_render.comp|tile_render_packed_rgba8.comp.spv|GSR_PACKED_SPLAT|GSR_OUT_RGBA8"
  "shaders/tile_render.comp|tile_render_packed_stats.comp.spv|GSR_PACKED_SPLAT|GSR_TILE_STATS"
  # Phase 6: StopThePop window, render-loop experiments (shared AoS, fp16 math).
  "shaders/tile_render.comp|tile_render_stp.comp.spv|GSR_STOPTHEPOP"
  "shaders/tile_render.comp|tile_render_stp_rgba32f.comp.spv|GSR_STOPTHEPOP|GSR_OUT_RGBA32F"
  "shaders/tile_render.comp|tile_render_packed_stp.comp.spv|GSR_PACKED_SPLAT|GSR_STOPTHEPOP"
  "shaders/tile_render.comp|tile_render_packed_stp_rgba32f.comp.spv|GSR_PACKED_SPLAT|GSR_STOPTHEPOP|GSR_OUT_RGBA32F"
  "shaders/tile_render.comp|tile_render_aos.comp.spv|GSR_SHARED_AOS"
  "shaders/tile_render.comp|tile_render_packed_aos.comp.spv|GSR_PACKED_SPLAT|GSR_SHARED_AOS"
  "shaders/tile_render.comp|tile_render_fp16.comp.spv|GSR_FP16_MATH"
  "shaders/tile_render.comp|tile_render_packed_fp16.comp.spv|GSR_PACKED_SPLAT|GSR_FP16_MATH")
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SPIRV_OUTPUTS)
function(gsr_add_shader src spvName)  # remaining args: preprocessor defines
  set(spv ${SHADER_OUTPUT_DIR}/${spvName})
  set(defs)
  foreach(d ${ARGN})
    list(APPEND defs -D${d})
  endforeach()
  add_custom_command(
    OUTPUT  ${spv}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND Vulkan::glslc --target-env=vulkan1.3 -O -g ${defs} -MD -MF ${spv}.d
            -o ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    DEPFILE ${spv}.d
    COMMENT "glslc ${spvName}"
    VERBATIM)
  set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} ${spv} PARENT_SCOPE)
endfunction()
foreach(src ${SHADER_SOURCES})
  get_filename_component(name ${src} NAME)
  gsr_add_shader(${src} ${name}.spv)
endforeach()
foreach(variant ${SHADER_VARIANTS})
  string(REPLACE "|" ";" parts "${variant}")
  list(GET parts 0 src)
  list(GET parts 1 out)
  list(SUBLIST parts 2 -1 defines)
  gsr_add_shader(${src} ${out} ${defines})
endforeach()
add_custom_target(shaders DEPENDS ${SPIRV_OUTPUTS} SOURCES ${SHADER_SOURCES} ${SHADER_HEADERS})
endif()  # NOT EMSCRIPTEN

# Compile definitions every target that includes GLM / Windows headers must agree on.
set(GSR_COMMON_DEFINITIONS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLM_FORCE_RADIANS
  NOMINMAX
  WIN32_LEAN_AND_MEAN)
set(GSR_WARNING_FLAGS /W4 /permissive- /utf-8)

# ---------------------------------------------------------------------------
# Scene library (no Vulkan): PLY loading, Gaussian data, scene transform, cameras.
# Shared by the app and the tests (and later by the CPU reference renderer).
# ---------------------------------------------------------------------------
add_library(gsr_scene STATIC
  src/scene/mapped_file.h       src/scene/mapped_file.cpp
  src/scene/gaussian_scene.h    src/scene/gaussian_scene.cpp
  src/scene/ply_loader.h        src/scene/ply_loader.cpp
  src/scene/scene_transform.h   src/scene/scene_transform.cpp
  src/scene/training_cameras.h  src/scene/training_cameras.cpp
  src/scene/half.h              src/scene/parallel.h
  src/scene/scene_compression.h src/scene/scene_compression.cpp
  src/scene/kmeans.cpp)
target_include_directories(gsr_scene PUBLIC src)
target_link_libraries(gsr_scene PUBLIC glm::glm PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(gsr_scene PUBLIC ${GSR_COMMON_DEFINITIONS})
if(MSVC)
  target_compile_options(gsr_scene PRIVATE ${GSR_WARNING_FLAGS})
endif()

# ---------------------------------------------------------------------------
# Splatting math, CPU depth sort and the CPU reference renderer (no Vulkan).
# Shared by the app (sort), tools/gsplat_ref and the tests.
# ---------------------------------------------------------------------------
add_library(gsr_splat STATIC
  src/splat/splat_math.h          src/splat/splat_math.cpp
  src/splat/depth_sort.h          src/splat/depth_sort.cpp
  src/splat/reference_renderer.h  src/splat/reference_renderer.cpp
  src/splat/tile_coverage.h       src/splat/tile_coverage.cpp)
target_link_libraries(gsr_splat PUBLIC gsr_scene)
if(MSVC)
  target_compile_options(gsr_splat PRIVATE ${GSR_WARNING_FLAGS})
endif()

# ---------------------------------------------------------------------------
# Viewer-side helpers without Vulkan: camera, scene camera/setup, image I/O.
# Shared by the app and the offline tools so both resolve cameras identically.
# ---------------------------------------------------------------------------
add_library(gsr_common STATIC
  src/app/camera.h          src/app/camera.cpp
  src/app/scene_camera.h    src/app/scene_camera.cpp
  src/app/image_io.h        src/app/image_io.cpp)
target_include_directories(gsr_common PUBLIC src)
target_link_libraries(gsr_common PUBLIC gsr_scene glm::glm nlohmann_json::nlohmann_json stb)
if(MSVC)
  target_compile_options(gsr_common PRIVATE ${GSR_WARNING_FLAGS})
endif()

# ---------------------------------------------------------------------------
# Web build: WebGPU backend, wasm module and static page (build-web/dist/). Stops here.
# ---------------------------------------------------------------------------
if(EMSCRIPTEN)
  include(cmake/web.cmake)
  return()
endif()

# ---------------------------------------------------------------------------
# Vulkan core + renderers (library): shared by the app and the GPU tests.
# ---------------------------------------------------------------------------
add_library(gsr_render STATIC
  src/core/vk_common.h
  src/core/context.h        src/core/context.cpp
  src/core/swapchain.h      src/core/swapchain.cpp
  src/core/resources.h      src/core/resources.cpp
  src/core/descriptors.h    src/core/descriptors.cpp
  src/core/pipelines.h      src/core/pipelines.cpp
  src/core/profiler.h       src/core/profiler.cpp
  src/core/memory_report.h  src/core/memory_report.cpp
  src/core/timing_stats.h
  src/core/vma_impl.cpp
  src/render/renderer.h
  src/render/gpu_scene.h    src/render/gpu_scene.cpp
  src/render/test_renderer.h src/render/test_renderer.cpp
  src/render/point_cloud_renderer.h src/render/point_cloud_renderer.cpp
  src/render/splat_draw.h   src/render/splat_draw.cpp
  src/render/splat_renderer.h src/render/splat_renderer.cpp
  src/render/gpu_radix_sort.h src/render/gpu_radix_sort.cpp
  src/render/splat_gpusort_renderer.h src/render/splat_gpusort_renderer.cpp
  src/render/gpu_scan.h     src/render/gpu_scan.cpp
  src/render/splat_tile_renderer.h src/render/splat_tile_renderer.cpp
  src/render/render_splat.h)
target_include_directories(gsr_render PUBLIC src)
target_link_libraries(gsr_render PUBLIC gsr_scene gsr_splat Vulkan::Vulkan glfw glm::glm imgui vma)
target_compile_definitions(gsr_render PUBLIC
  GSR_SHADER_DIR="${SHADER_OUTPUT_DIR}"
  GLFW_INCLUDE_NONE
  ${GSR_COMMON_DEFINITIONS})
if(MSVC)
  target_compile_options(gsr_render PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_dependencies(gsr_render shaders)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(gsplat
  src/app/main.cpp
  src/app/cli.h             src/app/cli.cpp
  src/app/application.h     src/app/application.cpp
  src/app/scene_loading.h   src/app/scene_loading.cpp
  src/app/file_dialog.h     src/app/file_dialog.cpp)
target_link_libraries(gsplat PRIVATE gsr_render gsr_scene gsr_splat gsr_common Vulkan::Vulkan glfw glm::glm imgui vma
                                     stb nlohmann_json::nlohmann_json)
if(WIN32)
  # Native "Open scene..." dialog (src/app/file_dialog.cpp): COM + the shell item API.
  target_link_libraries(gsplat PRIVATE ole32 shell32 uuid)
endif()
if(MSVC)
  target_compile_options(gsplat PRIVATE ${GSR_WARNING_FLAGS})
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT gsplat)
endif()
add_dependencies(gsplat shaders)

# ---------------------------------------------------------------------------
# Offline tools: CPU reference renderer and image diff (no Vulkan)
# ---------------------------------------------------------------------------
add_executable(gsplat_ref tools/gsplat_ref.cpp)
target_link_libraries(gsplat_ref PRIVATE gsr_splat gsr_common)
add_executable(imgdiff tools/imgdiff.cpp)
target_link_libraries(imgdiff PRIVATE gsr_common)
# Phase 6: temporal popping metric of a frame sequence vs a ground-truth sequence.
add_executable(flicker tools/flicker.cpp)
target_link_libraries(flicker PRIVATE gsr_common nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(gsplat_ref PRIVATE ${GSR_WARNING_FLAGS})
  target_compile_options(imgdiff PRIVATE ${GSR_WARNING_FLAGS})
  target_compile_options(flicker PRIVATE ${GSR_WARNING_FLAGS})
endif()

# ---------------------------------------------------------------------------
# Tests (CTest): ctest --test-dir build -C <Config> --output-on-failure
# ---------------------------------------------------------------------------
enable_testing()
find_package(Python3 COMPONENTS Interpreter)

add_executable(test_ply_loader tests/test_ply_loader.cpp)
target_link_libraries(test_ply_loader PRIVATE gsr_scene nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(test_ply_loader PRIVATE ${GSR_WARNING_FLAGS})
endif()

set(GSR_TEST_DIR ${CMAKE_BINARY_DIR}/tests)
if(Python3_Interpreter_FOUND)
  # Fixture: the values gen_synthetic_ply.py writes into data/synthetic/ (no .ply written).
  add_test(NAME synthetic_expected_values
    COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/gen_synthetic_ply.py
            --expected-json ${GSR_TEST_DIR}/synthetic_expected.json)
  set_tests_properties(synthetic_expected_values PROPERTIES FIXTURES_SETUP synthetic_expected)
endif()
add_test(NAME ply_loader
  COMMAND test_ply_loader
          --synthetic-dir ${CMAKE_SOURCE_DIR}/data/synthetic
          --expected ${GSR_TEST_DIR}/synthetic_expected.json
          --scratch-dir ${GSR_TEST_DIR}/scratch
          --scenes-dir ${CMAKE_SOURCE_DIR}/data/scenes)
set_tests_properties(ply_loader PROPERTIES FIXTURES_REQUIRED synthetic_expected)

# Web port (W1): the in-memory and streaming loader entry points vs loadGaussianPly, bit for bit,
# on every synthetic file and utsuho_plush (if present), plus malformed files (same errors).
add_executable(test_ply_stream tests/test_ply_stream.cpp)
target_link_libraries(test_ply_stream PRIVATE gsr_scene)
if(MSVC)
  target_compile_options(test_ply_stream PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME ply_stream
  COMMAND test_ply_stream
          --synthetic-dir ${CMAKE_SOURCE_DIR}/data/synthetic
          --scratch-dir ${GSR_TEST_DIR}/scratch_stream
          --scenes-dir ${CMAKE_SOURCE_DIR}/data/scenes)

# Splatting math (cov3D/cov2D/conic/radius/SH/tile rect), depth sort, reference renderer.
add_executable(test_splat_math tests/test_splat_math.cpp)
target_link_libraries(test_splat_math PRIVATE gsr_splat)
if(MSVC)
  target_compile_options(test_splat_math PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME splat_math COMMAND test_splat_math)

# GPU radix sort + stream compaction vs the CPU sorter (headless Vulkan compute; needs a
# Vulkan 1.3 GPU). `test_gpu_sort --benchmark` prints sort throughput instead.
add_executable(test_gpu_sort tests/test_gpu_sort.cpp)
target_link_libraries(test_gpu_sort PRIVATE gsr_render)
if(MSVC)
  target_compile_options(test_gpu_sort PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME gpu_sort COMMAND test_gpu_sort)

# Phase 4: GPU scan, 64-bit/short-key radix sort, and the splat-tiles pipeline (duplication,
# tile ranges, per-tile lists, rendered image) vs CPU implementations (headless Vulkan).
add_executable(test_tile_raster tests/test_tile_raster.cpp)
target_link_libraries(test_tile_raster PRIVATE gsr_render)
if(MSVC)
  target_compile_options(test_tile_raster PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME tile_raster COMMAND test_tile_raster)

# Phase 5: scene compression on the CPU (half floats, Q8, encode/decode bounds, k-means
# determinism) and on the GPU (scene decoder vs the CPU decoder, packed render splats vs the
# 48-byte Splat, adaptive list capacity).
add_executable(test_compression tests/test_compression.cpp)
target_link_libraries(test_compression PRIVATE gsr_scene gsr_splat)
add_executable(test_scene_formats tests/test_scene_formats.cpp)
target_link_libraries(test_scene_formats PRIVATE gsr_render)
if(MSVC)
  target_compile_options(test_compression PRIVATE ${GSR_WARNING_FLAGS})
  target_compile_options(test_scene_formats PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME compression COMMAND test_compression)
add_test(NAME scene_formats COMMAND test_scene_formats)

# Phase 5: precision analysis of a scene (dynamic ranges, per-format errors), no GPU.
add_executable(scene_precision tools/scene_precision.cpp)
target_link_libraries(scene_precision PRIVATE gsr_splat nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(scene_precision PRIVATE ${GSR_WARNING_FLAGS})
endif()

# Phase 6: tight (opacity-aware) tile bounds, 8-px tiles, count/write agreement, exactness,
# StopThePop window (CPU + headless Vulkan).
add_executable(test_tight_bounds tests/test_tight_bounds.cpp)
target_link_libraries(test_tight_bounds PRIVATE gsr_render)
if(MSVC)
  target_compile_options(test_tight_bounds PRIVATE ${GSR_WARNING_FLAGS})
endif()
add_test(NAME tight_bounds COMMAND test_tight_bounds)
