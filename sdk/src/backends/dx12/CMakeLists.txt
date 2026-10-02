# This file is part of the FidelityFX SDK.
#
# SPDX-License-Identifier: MIT
# =============================================================================
# D3D12 backend for the Neural Graphics SDK.
# =============================================================================
# This backend was DELETED by the Arm fork (which went Vulkan-only and converted every
# shader to GLSL), and is restored here for one purpose: to host the portable dp4a
# inference backend on D3D12, so that the NFRU data graph runs on both graphics APIs from
# one implementation.
#
# It is deliberately NOT a full D3D12 backend, and the omissions are choices with
# reasons, not gaps:
#
#   * Effect macros (FFX_FI / FFX_NSS / FFX_OF) are NOT defined. Those gate
#     `shared/ffx_shader_blobs.cpp`'s dispatch into `shared/blob_accessors/`, whose
#     sources include generated `*_permutations.h` headers full of SHADER BLOBS. Shader
#     blobs are produced by `compile_shaders_with_depfile`, which chooses SPIR-V or DXIL
#     from the file extension; this tree has zero `.hlsl` files, so D3D12 could only ever
#     get SPIR-V, which it cannot consume. Leaving the macros undefined drops the whole
#     shader-permutation step -- and with it the FidelityFX_SC dependency for this target.
#
#   * `ffx_dx12_datagraph_blobs.cpp` replaces `shared/ffx_shader_blobs.cpp` here. It
#     serves only the data-graph model descriptors, which are the same hand-written
#     headers the Vulkan backend uses, so the effect code's name-based tensor joining
#     behaves identically on both APIs.
#
#   * `FrameInterpolationSwapchain/` IS built. Dropping it looked attractive -- it is the
#     DXGI swapchain replacement, a presentation feature, and it is the only thing here
#     needing `antilag2/ffx_antilag2_dx12.h` and a frame-generation enum that 1.1.3 took
#     from ffx_fsr3.h. But the public DX12 frame-generation provider
#     (ffx-api/src/dx12/ffx_provider_framegenerationswapchain_dx12.cpp) calls into it, and
#     that provider is registered unconditionally by ffx_provider.cpp whenever
#     FFX_BACKEND_DX12 is defined. Excluding the implementation produced six unresolved
#     externals instead of a smaller build.
#
# The dp4a module carries its own premade SPIR-V and DXIL, so nothing here depends on the
# SDK's shader compilation at all.
#
# NOTE ON OUTPUT NAMES: SDK_LIB_NAME is `ngsdk_${FFX_PLATFORM_NAME}`, and both `vk` and
# `dx12` have FFX_PLATFORM_NAME=windows_x64. The two backends therefore produce the SAME
# dll name and must be configured into separate build directories.
# -----------------------------------------------------------------------------

# No FFX_API_<API> guard here, deliberately.
#
# The Vulkan backend's guard is `if(NOT ${FFX_API_VK}) return()`, which looks like a gate
# but is not: FFX_API_VK is never defined anywhere in this tree, so CMake expands it to
# `if(NOT )`, which is FALSE, and the body always runs. Writing the same intent as
# `if(NOT FFX_API_DX12)` -- without the `${}` -- is the opposite: an undefined variable
# expands to a false value, `NOT` makes it true, and the backend silently never gets
# defined. That produced a link error ("cannot open input file
# ffx_backend_dx12_windows_x64.lib") with no hint as to why.
#
# The dispatch is already handled where it belongs: sdk/CMakeLists.txt only adds this
# subdirectory when FFX_GRAPHICS_API is dx12.

message(STATUS "Configure sdk/src/backends/dx12")

include("${FFX_GPU_PATH}/CMakeCompileShaders.txt")

set(FFX_SRC_BACKENDS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")

file(GLOB PRIVATE_SOURCE
    "${FFX_SHARED_PATH}/ffx_assert.cpp"
    "${FFX_SHARED_PATH}/ffx_util.cpp"
    "${FFX_SRC_BACKENDS_PATH}/shared/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# shared/*.cpp is globbed above for the genuinely shared pieces, but
# ffx_shader_blobs.cpp is the Vulkan-side blob dispatcher and must be excluded: it
# references the per-effect accessors under shared/blob_accessors/, which are not built
# here (see the header comment). ffx_dx12_datagraph_blobs.cpp supplies the one symbol the
# backend actually needs from it.
list(REMOVE_ITEM PRIVATE_SOURCE "${FFX_SRC_BACKENDS_PATH}/shared/ffx_shader_blobs.cpp")

# The DXGI swapchain replacement. The public DX12 frame-generation provider calls into
# it, so it has to be part of this library.
file(GLOB FRAME_INTERPOLATION_SWAPCHAIN_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/FrameInterpolationSwapchain/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/FrameInterpolationSwapchain/*.cpp"
)
list(APPEND PRIVATE_SOURCE ${FRAME_INTERPOLATION_SWAPCHAIN_SRC})

# -----------------------------------------------------------------------------
# Portable dp4a inference backend (shared with the Vulkan backend).
# -----------------------------------------------------------------------------
file(GLOB NFRU_DP4A_SOURCE
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a/*.h"
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a/*.cpp"
)
list(APPEND PRIVATE_SOURCE ${NFRU_DP4A_SOURCE})

# NSS's dp4a backend, D3D12 half.
#
# Listed explicitly rather than globbed, and that is the point: src/ also contains
# nss_dp4a.cpp and nss_dp4a_vk.cpp, which are the Vulkan host. A `src/*.cpp` glob would
# compile the entire Vulkan context into the D3D12 library -- it would build, because the
# Vulkan headers are on the include path, and then sit there as dead weight with a
# duplicate of the model tables' consumer.
#
# nss_dp4a_model.cpp is API-agnostic (shape derivation plus the baked layer tables) and is
# shared; nss_dp4a_dx12.cpp is the emitter over nss::Model::dispatches().
set(NSS_DP4A_DX12_SOURCE
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src/nss_dp4a_model.cpp"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src/nss_dp4a_dx12.cpp"
)
list(APPEND PRIVATE_SOURCE ${NSS_DP4A_DX12_SOURCE})
# Optical flow's blob accessor, and SPD's (the OF luminance pyramid pass includes
# spd/ffx_spd.h). These are AMD 1.1.3 files restored from the reference; unlike the dp4a
# module above they hold no data of their own, only links to the generated permutation
# headers, so they are not a licensing question.
set(NSS_OF_ACCESSOR_SOURCE
    "${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_opticalflow_shaderblobs.cpp"
    "${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_spd_shaderblobs.cpp")
list(APPEND PRIVATE_SOURCE ${NSS_OF_ACCESSOR_SOURCE})

file(GLOB_RECURSE PUBLIC_SOURCE "${FFX_HOST_BACKENDS_PATH}/dx12/*.h")

if (FFX_BUILD_BACKEND_AS_DLL)
    add_library(ffx_backend_dx12_${FFX_PLATFORM_NAME} SHARED ${PRIVATE_SOURCE} ${PUBLIC_SOURCE})
else()
    add_library(ffx_backend_dx12_${FFX_PLATFORM_NAME} STATIC ${PRIVATE_SOURCE} ${PUBLIC_SOURCE})
endif()

target_include_directories(ffx_backend_dx12_${FFX_PLATFORM_NAME} PRIVATE
    ${FFX_INCLUDE_PATH}
    ${FFX_SRC_BACKENDS_PATH}/shared
    "${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors"
    # The blob accessors reach into the effect's own private header by the path it is
    # included with elsewhere, "opticalflow/ffx_opticalflow_private.h", which resolves
    # against the components directory.
    "${FFX_COMPONENTS_PATH}"
    # The GPU-side headers the restored optical flow passes are built from, and SPD's,
    # which the luminance pyramid pass includes.
    "${FFX_INCLUDE_PATH}/FidelityFX/gpu"
    "${FFX_INCLUDE_PATH}/FidelityFX/gpu/opticalflow"
    "${FFX_INCLUDE_PATH}/FidelityFX/gpu/spd"
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a"
    # The NSS module: include/ holds nss_dp4a.h and nss_dp4a_dx12.h, generated/ its baked
    # model and DXIL headers, src/ its own headers.
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/include"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/generated"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src"
    # The hand-written data-graph model descriptors, shared with the Vulkan backend.
    "${FFX_SRC_BACKENDS_PATH}/vk/data_graphs"
    ${FFX_SHARED_PATH}
    ${FFX_LIB_PATH}
    # Headers only, for the dp4a module's Vulkan device translation unit. Nothing links
    # vulkan-1: the module resolves every Vulkan entry point at runtime, so this only
    # needs the vendored headers.
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vulkan-headers"
)

# d3d12/dxgi for the backend itself, plus for the dp4a module's DX12 device.
# vulkan-1 is NOT linked: the dp4a module loads it at runtime through
# vkGetInstanceProcAddr, so carrying the Vulkan device translation unit costs nothing at
# load time.
target_link_libraries(ffx_backend_dx12_${FFX_PLATFORM_NAME} PRIVATE d3d12 dxgi)

target_compile_definitions(ffx_backend_dx12_${FFX_PLATFORM_NAME} PRIVATE
    # See the header comment in nfru_dp4a.h: keeps the module's API internal so the
    # host's export table is unchanged by linking it in.
    NFRU_DP4A_INTERNAL
    # Same arrangement for NSS: its C API must not reach the host's export table either.
    NSS_DP4A_INTERNAL
    # The optical flow effect is built into this backend; its blob accessor is guarded on
    # this, and so is the dispatcher it will eventually share with the Vulkan backend.
    FFX_OF
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

# The NSS module's translation units need two things this target does not set:
#
#   /utf-8      Its comments and log strings are UTF-8 Chinese. Decoded as the local ANSI
#               code page, a multi-byte sequence ending in 0x5C ("\") swallows the closing
#               quote of a string literal, which surfaces as C2001 "newline in constant"
#               far from the cause.
#
#   /std:c++20  It builds its layer tables with designated initialisers (`.field = value`),
#               which the SDK's C++17 default rejects.
#
# Scoped to the module's own sources so the backend's existing code is unaffected.
set_source_files_properties(${NSS_DP4A_DX12_SOURCE} PROPERTIES COMPILE_OPTIONS "/utf-8;/std:c++20")

if(MSVC)
    target_compile_options(ffx_backend_dx12_${FFX_PLATFORM_NAME} PRIVATE /W3 /permissive- /EHsc /utf-8 /MP)
endif()

set_target_properties(ffx_backend_dx12_${FFX_PLATFORM_NAME} PROPERTIES FOLDER Backends)

# -----------------------------------------------------------------------------
# Shader compilation.
#
# This target had none: it was built data-graph-only, so the fork's deletion of every
# HLSL simply meant there was nothing to compile. Optical flow changes that -- its seven
# compute passes are real HLSL shaders -- so the permutation step is switched on here.
#
# Only optical flow and SPD are listed. Every other effect's HLSL is still absent from
# this tree; their blob accessors are not compiled and their pipeline creation requests
# are refused explicitly by ffx_dx12_datagraph_blobs.cpp. That is deliberate: this is the
# first compute-shader capability in this backend, and widening it is a separate job.
#
# The D3D12 permutation driver treats .hlsl as HLSL and sets SUPPORT_WAVE64 ON, so all
# four variants per pass are generated here -- which is why the reference's optical flow
# accessor works unmodified, and why the Vulkan side (GLSL, two variants) needs a
# different fix for the same component.
# -----------------------------------------------------------------------------
get_filename_component(FFX_PASS_SHADER_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../shaders/dx12 ABSOLUTE)
target_include_directories(ffx_backend_dx12_${FFX_PLATFORM_NAME} PRIVATE ${FFX_PASS_SHADER_OUTPUT_PATH})
file(MAKE_DIRECTORY ${FFX_PASS_SHADER_OUTPUT_PATH})
set(FFX_SC_DEPENDENT_TARGET ffx_backend_dx12_${FFX_PLATFORM_NAME})

if (FFX_OF OR FFX_ALL)
    # SPD first: the luminance pyramid pass includes spd/ffx_spd.h.
    include (CMakeShadersSPD.txt)
    include (CMakeShadersOpticalflow.txt)
endif()

if (FFX_SC_PERMUTATION_OUTPUTS)
    add_custom_target(ffx_shader_permutations_dx12 DEPENDS ${FFX_SC_PERMUTATION_OUTPUTS})
    add_dependencies(${FFX_SC_DEPENDENT_TARGET} ffx_shader_permutations_dx12)
    set_target_properties(ffx_shader_permutations_dx12 PROPERTIES FOLDER Backends)
endif()