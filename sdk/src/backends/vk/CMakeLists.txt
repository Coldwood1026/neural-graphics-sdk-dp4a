# This file is part of the FidelityFX SDK.
# 
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

if(NOT ${FFX_API_VK})
    return()
endif()

message(STATUS "Configure sdk/src/backends/vk")
message(STATUS "FFX_BUILD_BACKEND_AS_DLL =${FFX_BUILD_BACKEND_AS_DLL}")

#include CMake custom command generator functions for compiling shaders.
include("${FFX_GPU_PATH}/CMakeCompileShaders.txt")

function (CheckVulkanSDKVersion)
    # set the desired version
    set(vulkan_min_version "1.3.250")
    message(STATUS "Vulkan_VERSION=${Vulkan_VERSION}")

    if (Vulkan_VERSION VERSION_LESS ${vulkan_min_version})
        message(WARNING "Vulkan SDK ${vulkan_min_version} or above is required. Found Vulkan SDK ${Vulkan_VERSION}")
    endif()
endfunction()

# Add your local Vulkan headers directory to the search path
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vulkan-headers")
set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR}")

message(STATUS "Vulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan_INCLUDE_DIRS=${Vulkan_INCLUDE_DIRS}")

# Headers only: this backend is compiled with VK_NO_PROTOTYPES and resolves every entry
# point at runtime through the host's vkGetInstanceProcAddr, so it links no Vulkan import
# library. find_package(Vulkan REQUIRED) treated Vulkan_LIBRARY as required and made the
# SDK unconfigurable without a LunarG SDK installation, while using nothing from it.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/ffx_vulkan_headers.cmake")
ffx_require_vulkan_headers()
set(Vulkan_VERSION "${VULKAN_HEADERS_VERSION}")
message(STATUS "Vulkan headers: ${VULKAN_HEADERS_VERSION} at ${Vulkan_INCLUDE_DIR}")
CheckVulkanSDKVersion()

set(PUBLIC_SHADERS "")
set(PRIVATE_SHADERS "")

file(GLOB PRIVATE_SOURCE
    "${FFX_SHARED_PATH}/ffx_assert.cpp"
    "${FFX_SHARED_PATH}/ffx_util.cpp"
	"${FFX_SRC_BACKENDS_PATH}/shared/*.h"
	"${FFX_SRC_BACKENDS_PATH}/shared/*.cpp"
	"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/"
	"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# -----------------------------------------------------------------------------
# Portable dp4a inference backend.
#
# This is the whole replacement for the Arm data-graph execution path: an int8
# dp4a compute graph that runs on ordinary Vulkan and D3D12. It lives under
# shared/ because ONE implementation serves both APIs -- nfru_plan.cpp decides what
# to record, and nfru_vk.cpp / nfru_dx12.cpp only translate that into API calls. That
# split is what makes "Vulkan and DX12 produce bit-identical output" a property of the
# code rather than a promise about two files kept in step by hand.
#
# Both device translation units are compiled into both backends. That is deliberate:
# it keeps this module byte-identical to the standalone nfru_dp4a.dll that was
# validated bit-exact against the reference on real hardware, and the cost is only a
# link-time dependency on d3d12.lib. Vulkan is loaded at RUNTIME through
# vkGetInstanceProcAddr, so the Vulkan backend still has no import-library dependency
# on vulkan-1.lib, and the DX12 backend still has none on it either.
# -----------------------------------------------------------------------------
file(GLOB NFRU_DP4A_SOURCE
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a/*.h"
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a/*.cpp"
)
list(APPEND PRIVATE_SOURCE ${NFRU_DP4A_SOURCE})

# NSS's dp4a backend. Independent of the NFRU module above: different network, different
# kernels, its own baked weights and its own toolchain.
file(GLOB NSS_DP4A_SOURCE
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src/*.h"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src/*.cpp"
)
list(APPEND PRIVATE_SOURCE ${NSS_DP4A_SOURCE})

if (FFX_FI OR FFX_ALL)
	#if(WIN32)
		file(GLOB FRAME_INTERPOLATION_SWAPCHAIN_SRC
			"${CMAKE_CURRENT_SOURCE_DIR}/FrameInterpolationSwapchain/*.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/FrameInterpolationSwapchain/*.cpp"
		)
		list(APPEND PRIVATE_SOURCE ${FRAME_INTERPOLATION_SWAPCHAIN_SRC})
	#endif()
endif()

if (FFX_FI OR FFX_ALL)
	set(FFX_FI_PRIVATE_SOURCE
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_frameinterpolation_shaderblobs.h"
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_frameinterpolation_shaderblobs.cpp")
	list(APPEND PRIVATE_SOURCE ${FFX_FI_PRIVATE_SOURCE})

	# Add shaders for Frame Interpolation

	file(GLOB FFX_FI_PRIVATE_SHADERS
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/frameinterpolation/*.glsl")
	list(APPEND PRIVATE_SHADERS ${FFX_FI_PRIVATE_SHADERS})

	file(GLOB FFX_FI_PUBLIC_SHADERS
		"${FFX_GPU_PATH}/frameinterpolation/*.h"
		"${FFX_GPU_PATH}/frameinterpolation/*.glsl")
	list(APPEND PUBLIC_SHADERS ${FFX_FI_PUBLIC_SHADERS})
endif()

if (FFX_NSS OR FFX_ALL)
	set(FFX_NSS_PRIVATE_SOURCE
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_nss_shaderblobs.h"
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_nss_shaderblobs.cpp")
	list(APPEND PRIVATE_SOURCE ${FFX_NSS_PRIVATE_SOURCE})
	
	# Add shaders for nss
	
	file(GLOB FFX_NSS_PRIVATE_SHADERS
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/nss/*.glsl")
	list(APPEND PRIVATE_SHADERS ${FFX_NSS_PRIVATE_SHADERS})
	
	file(GLOB FFX_NSS_PUBLIC_SHADERS
		"${FFX_GPU_PATH}/nss/*.h"
		"${FFX_GPU_PATH}/nss/*.glsl")
	list(APPEND PUBLIC_SHADERS ${FFX_NSS_PUBLIC_SHADERS})
endif()

if (FFX_OF OR FFX_ALL)
	set(ARM_OF_PRIVATE_SOURCE
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_opticalflow_shaderblobs.h"
		"${FFX_SRC_BACKENDS_PATH}/shared/blob_accessors/ffx_opticalflow_shaderblobs.cpp")
	list(APPEND PRIVATE_SOURCE ${ARM_OF_PRIVATE_SOURCE})

	# Add shaders for Arm Optical Flow

	file(GLOB ARM_OF_PRIVATE_SHADERS
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/opticalflow/*.glsl"
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/opticalflow/*.comp"
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/opticalflow/*.frag"
		"${CMAKE_CURRENT_SOURCE_DIR}/shaders/opticalflow/*.vert")
	list(APPEND PRIVATE_SHADERS ${ARM_OF_PRIVATE_SHADERS})

	file(GLOB ARM_OF_PUBLIC_SHADERS
		"${FFX_GPU_PATH}/opticalflow/*.h"
		"${FFX_GPU_PATH}/opticalflow/*.glsl")
	list(APPEND PUBLIC_SHADERS ${ARM_OF_PUBLIC_SHADERS})
endif()

file(GLOB_RECURSE PUBLIC_SOURCE
    "${FFX_HOST_BACKENDS_PATH}/vk/*.h")

if (FFX_BUILD_BACKEND_AS_DLL)
    add_library(ffx_backend_vk_${FFX_PLATFORM_NAME} SHARED ${PRIVATE_SOURCE} ${PUBLIC_SOURCE} ${PRIVATE_SHADERS} ${PUBLIC_SHADERS})
else()
    add_library(ffx_backend_vk_${FFX_PLATFORM_NAME} STATIC ${PRIVATE_SOURCE} ${PUBLIC_SOURCE} ${PRIVATE_SHADERS} ${PUBLIC_SHADERS})
endif()

# Link against SPIRV-Tools built from source
target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE SPIRV-Tools-opt)
target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE SPIRV-Tools)

# Include SPIRV-Tools headers
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE
    "${FFX_LIB_PATH}/SPIRV-Tools/include"
    "${FFX_LIB_PATH}/SPIRV-Headers/include"
)

# Depend on vulkan-headers in the sdk/include folder to ensure that the Vulkan headers are available.
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC "${Vulkan_INCLUDE_DIR}")

# vk backend source
source_group("private_source"  FILES ${PRIVATE_SOURCE})
source_group("public_source"   FILES ${PUBLIC_SOURCE})
source_group("private_shaders" FILES ${PRIVATE_SHADERS})
source_group("public_shaders"  FILES ${PUBLIC_SHADERS})

# --- Portable, short per-config output dir for generated FidelityFX headers ---
if(CMAKE_CONFIGURATION_TYPES)
  # Multi-config generators (Visual Studio, Xcode)
  # Use CMAKE_CFG_INTDIR so the path is per-config *at build time* (e.g. $(Configuration)).
  set(FFX_PASS_SHADER_OUTPUT_PATH
      "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/ffx_shaders"
      CACHE PATH "" FORCE)
  set(FFX_PASS_DATA_GRAPH_OUTPUT_PATH
      "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/ffx_graphs"
      CACHE PATH "" FORCE)

  # Pre-create real folders for all known configs at *configure* time (no genex here)
  foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${cfg}/ffx_shaders")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${cfg}/ffx_graphs")
  endforeach()
else()
  # Single-config generators (Ninja/Make — Android uses Ninja)
  set(FFX_PASS_SHADER_OUTPUT_PATH
      "${CMAKE_BINARY_DIR}/ffx_shaders"
      CACHE PATH "" FORCE)
  set(FFX_PASS_DATA_GRAPH_OUTPUT_PATH
      "${CMAKE_BINARY_DIR}/ffx_graphs"
      CACHE PATH "" FORCE)

  file(MAKE_DIRECTORY "${FFX_PASS_SHADER_OUTPUT_PATH}")
  file(MAKE_DIRECTORY "${FFX_PASS_DATA_GRAPH_OUTPUT_PATH}")
endif()

target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_INCLUDE_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_LIB_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_PASS_SHADER_OUTPUT_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})
# Checked-in data-graph model descriptors. These used to be generated into
# FFX_PASS_DATA_GRAPH_OUTPUT_PATH by Model_Parser; they are hand-written now (see
# the header comments) because the only thing that consumed them was the ARM
# data-graph pipeline creation, and Model_Parser crashes on the repo's own .vgf.
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data_graphs")

# The dp4a module: headers, and NFRU_DP4A_INTERNAL so its API is neither imported nor
# exported (see the linkage block in nfru_dp4a.h -- without it the module's own
# definitions would be declared dllimport and the link would fail).
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE
    "${FFX_SRC_BACKENDS_PATH}/shared/nfru_dp4a")
target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE NFRU_DP4A_INTERNAL)

# The NSS module: include/ holds nss_dp4a.h, generated/ its baked model headers, src/ its
# own headers. NSS_DP4A_INTERNAL keeps its C API out of the host's export table, exactly
# as NFRU_DP4A_INTERNAL does for NFRU.
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/include"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/generated"
    "${FFX_SRC_BACKENDS_PATH}/shared/nss_dp4a/src"
)
target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE NSS_DP4A_INTERNAL)

# The NSS module needs two things the SDK target does not set, and both are hard errors
# without them:
#
#   /utf-8      Its comments and log strings are UTF-8 Chinese. Without this MSVC decodes
#               them as the local ANSI code page, and a multi-byte sequence whose trailing
#               byte is 0x5C ("\") swallows the closing quote of the string literal. That
#               surfaces as C2001 "newline in constant" at eleven sites, none of them
#               anywhere near the actual problem.
#
#   /std:c++20  It builds its layer tables with designated initialisers (`.field = value`).
#               The SDK defaults to C++17, where those are rejected under this target's
#               settings (C7555), which then cascades into C2027 "undefined type
#               nss_baked::LayerDesc" at forty more sites.
#
# Scoped to the module's own translation units so the SDK's existing sources keep being
# compiled exactly as before.
set_source_files_properties(${NSS_DP4A_SOURCE} PROPERTIES COMPILE_OPTIONS "/utf-8;/std:c++20")

# d3d12/dxgi: the dp4a module carries its DX12 device translation unit too (see above).
target_link_libraries(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE dxgi d3d12)
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_COMPONENTS_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_SHARED_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE "${FFX_SRC_BACKENDS_PATH}/shared")
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_PASS_SHADER_OUTPUT_PATH})
target_include_directories(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})

set_source_files_properties(${PRIVATE_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${PUBLIC_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

file(MAKE_DIRECTORY ${FFX_PASS_SHADER_OUTPUT_PATH})
file(MAKE_DIRECTORY ${FFX_PASS_DATA_GRAPH_OUTPUT_PATH})

if (FFX_AUTO_COMPILE_SHADERS)
    set(FFX_SC_DEPENDENT_TARGET ffx_backend_vk_${FFX_PLATFORM_NAME})
else()
    set(FFX_SC_DEPENDENT_TARGET ffx_backend_vk_shaders_${FFX_PLATFORM_NAME})
    add_custom_target(${FFX_SC_DEPENDENT_TARGET})
    #add_dependencies(ffx_backend_vk_${FFX_PLATFORM_NAME} ${FFX_SC_DEPENDENT_TARGET})
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
	cmake_policy(SET CMP0116 OLD)
endif()

if (FFX_FI OR FFX_ALL)
	target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE FFX_FI)
	include (CMakeShadersArmFrameinterpolation.txt)
endif()

if (FFX_OF OR FFX_ALL)
	target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE FFX_OF)
	include (CMakeShadersOpticalflow.txt)
endif()

if (FFX_NSS OR FFX_ALL)
	target_compile_definitions(ffx_backend_vk_${FFX_PLATFORM_NAME} PRIVATE FFX_NSS)
	include (CMakeShadersNSS.txt)
endif()

add_custom_target(ffx_shader_permutations_vk DEPENDS ${FFX_SC_PERMUTATION_OUTPUTS})
add_dependencies(${FFX_SC_DEPENDENT_TARGET} ffx_shader_permutations_vk)

# Make sure shader builds are a dependency of the backend
add_dependencies(ffx_backend_vk_${FFX_PLATFORM_NAME} ffx_shader_permutations_vk)

# Add to solution folder.
set_target_properties(ffx_backend_vk_${FFX_PLATFORM_NAME} PROPERTIES FOLDER Backends)
set_target_properties(ffx_shader_permutations_vk PROPERTIES FOLDER Backends)
