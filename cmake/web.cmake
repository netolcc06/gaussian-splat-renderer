# Web build (Emscripten + WebGPU), included from CMakeLists.txt when EMSCRIPTEN is set.
# Configure and build with the `web` preset (see CMakePresets.json, tools/web/build_web.ps1):
#   cmake --preset web && cmake --build --preset web
# Output: a static site in ${CMAKE_BINARY_DIR}/dist (index.html, app.js, style.css,
# gsplat.mjs, gsplat.wasm). Design and phases: docs/web/DESIGN.md.

# WebGPU binding: Dawn's emdawnwebgpu port (the standard C webgpu.h). The Dawn-style C++ wrapper
# (webgpu_cpp.h) is not stable across releases, so it is kept off the include path.
set(GSR_WEBGPU_PORT --use-port=emdawnwebgpu:cpp_bindings=false)
set(GSR_WEB_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow)

# ---------------------------------------------------------------------------
# WGSL shaders: shaders/wgsl/*.wgsl embedded into the wasm as byte arrays (no .data file,
# no virtual file system). The WGSL preprocessor resolves #include against this table.
# ---------------------------------------------------------------------------
file(GLOB GSR_WGSL_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/shaders/wgsl/*.wgsl)
set(GSR_WGSL_EMBEDDED ${CMAKE_BINARY_DIR}/generated/wgsl_embedded.cpp)
add_custom_command(
  OUTPUT  ${GSR_WGSL_EMBEDDED}
  COMMAND ${CMAKE_COMMAND} -DWGSL_DIR=${CMAKE_SOURCE_DIR}/shaders/wgsl -DOUTPUT=${GSR_WGSL_EMBEDDED}
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_wgsl.cmake
  DEPENDS ${GSR_WGSL_SOURCES} ${CMAKE_SOURCE_DIR}/cmake/embed_wgsl.cmake
  COMMENT "Embedding WGSL shaders"
  VERBATIM)

# ---------------------------------------------------------------------------
# WebGPU backend (webgpu.h only; no Emscripten APIs here so it could also build against
# native Dawn later, DESIGN.md section 9.2).
# ---------------------------------------------------------------------------
add_library(gsr_webgpu STATIC
  src/webgpu/wgpu_context.h       src/webgpu/wgpu_context.cpp
  src/webgpu/wgpu_resources.h     src/webgpu/wgpu_resources.cpp
  src/webgpu/wgsl_preprocess.h    src/webgpu/wgsl_preprocess.cpp
  src/webgpu/wgsl_sources.h       ${GSR_WGSL_EMBEDDED}
  src/webgpu/wgpu_shaders.h       src/webgpu/wgpu_shaders.cpp
  src/webgpu/wgpu_gradient.h      src/webgpu/wgpu_gradient.cpp
  src/webgpu/wgpu_resolve.h       src/webgpu/wgpu_resolve.cpp
  src/webgpu/wgpu_scene.h         src/webgpu/wgpu_scene.cpp
  src/webgpu/wgpu_points.h        src/webgpu/wgpu_points.cpp
  src/webgpu/wgpu_decode_selftest.h src/webgpu/wgpu_decode_selftest.cpp
  src/webgpu/shader_constants.h
  src/webgpu/coverage_spans.h     src/webgpu/coverage_spans.cpp
  src/webgpu/wgpu_compute.h       src/webgpu/wgpu_compute.cpp
  src/webgpu/wgpu_scan.h          src/webgpu/wgpu_scan.cpp
  src/webgpu/wgpu_radix_sort.h    src/webgpu/wgpu_radix_sort.cpp
  src/webgpu/wgpu_tile_renderer.h src/webgpu/wgpu_tile_renderer.cpp
  src/webgpu/wgpu_tile_selftest.h src/webgpu/wgpu_tile_selftest.cpp)
target_include_directories(gsr_webgpu PUBLIC src)
# gsr_splat: the splat math, tile coverage, CPU depth sort and reference renderer (W2 tests).
target_link_libraries(gsr_webgpu PUBLIC gsr_scene gsr_splat glm::glm PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(gsr_webgpu PUBLIC ${GSR_WEBGPU_PORT} PRIVATE ${GSR_WEB_WARNING_FLAGS})
target_link_options(gsr_webgpu PUBLIC ${GSR_WEBGPU_PORT})

# ---------------------------------------------------------------------------
# The wasm module (ES6 module factory, no main(): app.js calls gsr_init()).
# ---------------------------------------------------------------------------
add_executable(gsplat_web src/web/main_web.cpp)
# Shared CPU code: the streaming PLY loader and scene encoders (gsr_scene), camera and scene
# framing (gsr_common).
target_link_libraries(gsplat_web PRIVATE gsr_webgpu gsr_scene gsr_common nlohmann_json::nlohmann_json)
target_compile_options(gsplat_web PRIVATE ${GSR_WEB_WARNING_FLAGS})
set(GSR_WEB_DIST ${CMAKE_BINARY_DIR}/dist)
set_target_properties(gsplat_web PROPERTIES
  OUTPUT_NAME gsplat
  SUFFIX ".mjs"
  RUNTIME_OUTPUT_DIRECTORY ${GSR_WEB_DIST})
target_link_options(gsplat_web PRIVATE
  --no-entry
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sEXPORT_NAME=createGsplatModule
  -sENVIRONMENT=web
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=4GB
  -sEXPORTED_FUNCTIONS=_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString,stringToNewUTF8)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  # Closure minifies the JS glue (emdawnwebgpu README: always enable it in release builds).
  target_link_options(gsplat_web PRIVATE --closure=1)
else()
  target_link_options(gsplat_web PRIVATE -sASSERTIONS=2)
endif()

# ---------------------------------------------------------------------------
# Page sources (web/) copied next to the module.
# ---------------------------------------------------------------------------
set(GSR_WEB_PAGE_FILES index.html app.js style.css)
set(GSR_WEB_PAGE_OUTPUTS)
foreach(f ${GSR_WEB_PAGE_FILES})
  add_custom_command(
    OUTPUT  ${GSR_WEB_DIST}/${f}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/web/${f} ${GSR_WEB_DIST}/${f}
    DEPENDS ${CMAKE_SOURCE_DIR}/web/${f}
    COMMENT "Copying web/${f}"
    VERBATIM)
  list(APPEND GSR_WEB_PAGE_OUTPUTS ${GSR_WEB_DIST}/${f})
endforeach()
# Demo assets shipped with the page (published; only permissively licensed / our own scenes).
set(GSR_WEB_DEMO_FILES
  "data/demo/knock-community-hall/knock-community-hall_300k.ply.gz|demo/knock-community-hall_300k.ply.gz"
  "data/demo/knock-community-hall/camera.json|demo/knock-community-hall.camera.json"
  "data/demo/knock-community-hall/ATTRIBUTION.txt|demo/knock-community-hall.ATTRIBUTION.txt"
  "data/synthetic/grid_512_sh3.ply|demo/grid_512_sh3.ply")
foreach(entry ${GSR_WEB_DEMO_FILES})
  string(REPLACE "|" ";" pair "${entry}")
  list(GET pair 0 src)
  list(GET pair 1 dst)
  # Optional: the demo scene is not in the source repo (tools/demo/make_demo_scene.ps1 builds it)
  # and the synthetic grid comes from tools/gen_synthetic_ply.py. The page still works without them.
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${src})
    message(STATUS "Web demo asset not found, skipped: ${src}")
    continue()
  endif()
  add_custom_command(
    OUTPUT  ${GSR_WEB_DIST}/${dst}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/${src} ${GSR_WEB_DIST}/${dst}
    DEPENDS ${CMAKE_SOURCE_DIR}/${src}
    COMMENT "Copying demo asset ${dst}"
    VERBATIM)
  list(APPEND GSR_WEB_PAGE_OUTPUTS ${GSR_WEB_DIST}/${dst})
endforeach()
add_custom_target(web_page ALL DEPENDS ${GSR_WEB_PAGE_OUTPUTS})
add_dependencies(gsplat_web web_page)
