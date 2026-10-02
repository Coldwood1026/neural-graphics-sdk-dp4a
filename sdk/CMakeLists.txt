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
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.23)
message(STATUS "Start configure sdk-core")
message(STATUS "FFX_BUILD_AS_DLL = ${FFX_BUILD_AS_DLL}")

if(POLICY CMP0147)
	# parallel custom command builds (used for shader compilation)
	cmake_policy(SET CMP0147 NEW)
endif()

message(STATUS "FFX_API_BACKEND is ${FFX_API_BACKEND}")

if (FFX_GRAPHICS_API STREQUAL "dx12")
	# Restored. The Arm fork deleted the D3D12 backend outright; it is back for one
	# purpose -- hosting the portable dp4a inference backend on D3D12 -- and it is
	# deliberately compute-only and data-graph-only. See
	# sdk/src/backends/dx12/CMakeLists.txt for what is omitted and why.
	message(STATUS "Building FidelityFX SDK with DX12 backend (dp4a inference host)")
elseif(FFX_GRAPHICS_API STREQUAL "vk")
	message(STATUS "Building FidelityFX SDK with VK backend")
else()
	message(FATAL_ERROR "Unsupported graphics API!")
endif()

# Pre-compile shaders
set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile shaders automatically as a prebuild step.")

if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(USE_DEPFILE TRUE)
else()
    set(USE_DEPFILE FALSE)
endif()

# Set Visual Studio version
if(CMAKE_GENERATOR STREQUAL "Visual Studio 14 2015")
    set(FFX_VS_VERSION 2015)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 15 2017")
	set(FFX_VS_VERSION 2017)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 16 2019")
    set(FFX_VS_VERSION 2019)
elseif(CMAKE_GENERATOR STREQUAL "Visual Studio 17 2022")
	set(FFX_VS_VERSION 2022)
endif()

if (NOT FFX_FSR3_AS_LIBRARY)
	# Set configuration types based on what we are building for
	if(MSVC)
		# Embed PDBs in the debug versions of the libs
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7 /Od")
	endif()

	# Write both debug and release versions of the static libs to the /bin folder as they are uniquely named
	# Use platform-specific directory to avoid conflicts between different architectures
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)

	if (FFX_BUILD_AS_DLL)
		# Write both debug and release versions of the dlls to the /bin folder as they are uniquely named
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_HOME_DIRECTORY}/bin/ffx_sdk/${FFX_PLATFORM_NAME}/)
		set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS FFX_BUILD_AS_DLL)
	endif()
endif()

# Add requred compile definitions
add_compile_definitions(_UNICODE)
add_compile_definitions(UNICODE)

if(FFX_VS_VERSION STREQUAL 2015 OR FFX_VS_VERSION STREQUAL 2017)
    message(NOTICE "Forcing the SDK path for VS 2015 and VS 2017")
    set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0.18362.0")
endif()

# Setup common variables
# NOTE: VGF_PARSER_EXECUTABLE (the Arm Model Parser) is deliberately no longer set.
# It serialised the .vgf data-graph models into FfxDataGraphBlob headers for the
# ARM execution path, which the portable dp4a backend replaces; the descriptors are
# now checked in by hand under sdk/src/backends/vk/data_graphs/. It also crashed
# with 0xC0000409 on this repository's own .vgf files.
if(CMAKE_HOST_WIN32)
	# Two shader compilers live in tools/binary_store, and which one is needed depends on the
# backend:
#
#   FidelityFX_SC.exe        the fork's build. GLSL only -- it fails with
#                            "Only GLSLCompiler is currently supported." on any HLSL.
#                            This is what produced the currently-validated SPIR-V, so the
#                            Vulkan build keeps using it.
#   FidelityFX_SC_hlsl.exe   AMD FidelityFX SDK 1.1.3's build, which handles HLSL and
#                            needs dxcompiler.dll / dxil.dll alongside it. The D3D12
#                            build needs this one: the fork deleted every HLSL, but
#                            optical flow's seven compute passes are being restored.
if (FFX_GRAPHICS_API STREQUAL "dx12")
	set(FFX_SC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/FidelityFX_SC_hlsl.exe)
else()
	set(FFX_SC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/FidelityFX_SC.exe)
endif()
	# we need to use the glslangValidator that supports ML extensions for Vulkan
	set(FFX_GLSLANG_EXE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/glslangValidator.exe)
else()
	set(FFX_SC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/FidelityFX_SC)
	# we need to use the glslangValidator that supports ML extensions for Vulkan
	set(FFX_GLSLANG_EXE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tools/binary_store/glslangValidator)
endif()

set(FFX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FFX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set(FFX_BIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/ffx_sdk)
set(FFX_SHARED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/shared)
set(FFX_HOST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/FidelityFX/host)
set(FFX_GPU_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/FidelityFX/gpu)
set(FFX_COMPONENTS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/components)
set(FFX_HOST_BACKENDS_PATH ${FFX_HOST_PATH}/backends)
set(FFX_SRC_BACKENDS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/backends)

# ------------------------------------------------------
# Define common source groupings used by all effects

# Set shared and global public sources
file(GLOB FFX_SHARED_SOURCES
    "${FFX_SHARED_PATH}/*.cpp"
    "${FFX_SHARED_PATH}/*.h")

set(FFX_PUBLIC_SOURCES
    "${FFX_HOST_PATH}/ffx_assert.h"
    "${FFX_HOST_PATH}/ffx_error.h"
	"${FFX_HOST_PATH}/ffx_interface.h"
    "${FFX_HOST_PATH}/ffx_types.h"
    "${FFX_HOST_PATH}/ffx_util.h")

# Init shader sources (for easy viewing)
set(FFX_PUBLIC_SHADER_SOURCES
    "${FFX_GPU_PATH}/ffx_common_types.h"
    "${FFX_GPU_PATH}/ffx_core.h"
    "${FFX_GPU_PATH}/ffx_core_cpu.h"
    "${FFX_GPU_PATH}/ffx_core_glsl.h"
    "${FFX_GPU_PATH}/ffx_core_gpu_common.h"
    "${FFX_GPU_PATH}/ffx_core_gpu_common_half.h"
    "${FFX_GPU_PATH}/ffx_core_hlsl.h"
    "${FFX_GPU_PATH}/ffx_core_portability.h")

if(NOT FFX_FSR3_AS_LIBRARY)
	set(PROJ_NAME "FidelityFX-SDK_${FFX_API_BACKEND}")
	project(PROJ_NAME)
	message(STATUS "Creating project ${PROJ_NAME}")
endif()

# Components
#add_subdirectory(${FFX_COMPONENTS_PATH}/opticalflow)

if(FFX_NSS OR FFX_ALL)
	message(STATUS "Component FFX_NSS=ON")
	add_subdirectory(${FFX_COMPONENTS_PATH}/nss)
endif()

if(FFX_FI OR FFX_ALL)
	message(STATUS "Component FFX_FI=ON")
	add_subdirectory(${FFX_COMPONENTS_PATH}/frameinterpolation)
endif()

if (FFX_OF OR FFX_ALL)
	message(STATUS "Component FFX_OF=ON")
	add_subdirectory(${FFX_COMPONENTS_PATH}/opticalflow)
endif()

# Add appropriate graphics backend if requested
if (FFX_GRAPHICS_API STREQUAL "vk")
	# Configure SPIRV-Tools to build as static libraries
	# Neural Graphics SDK for Game Engines uses a custom fork of SPIRV-Tools with graph extensions
	set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
	set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
	set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
	set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
	# Set SPIRV-Headers location - must be non-cache variable so it propagates to subdirectory
	set(SPIRV-Headers_SOURCE_DIR "${FFX_LIB_PATH}/SPIRV-Headers")
	# Hide SPIRV-Tools symbols to prevent conflicts with applications that also use SPIRV-Tools
	# Save current visibility settings
	set(SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(SAVED_CXX_VISIBILITY_PRESET ${CMAKE_CXX_VISIBILITY_PRESET})
	set(SAVED_VISIBILITY_INLINES_HIDDEN ${CMAKE_VISIBILITY_INLINES_HIDDEN})
	# Set hidden visibility for SPIRV-Tools to avoid symbol conflicts
	set(BUILD_SHARED_LIBS OFF)
	set(CMAKE_CXX_VISIBILITY_PRESET hidden)
	set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
	# Add SPIRV-Tools subdirectory
	add_subdirectory(${FFX_LIB_PATH}/SPIRV-Tools)
	if(TARGET SPIRV-Tools-shared)
		set_target_properties(SPIRV-Tools-shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
	endif()
	# Restore previous visibility settings for SDK build
	set(BUILD_SHARED_LIBS ${SAVED_BUILD_SHARED_LIBS})
	set(CMAKE_CXX_VISIBILITY_PRESET ${SAVED_CXX_VISIBILITY_PRESET})
	set(CMAKE_VISIBILITY_INLINES_HIDDEN ${SAVED_VISIBILITY_INLINES_HIDDEN})
	add_subdirectory(${FFX_SRC_BACKENDS_PATH}/vk)
endif()

# The D3D12 backend is deliberately outside the block above: that block exists to
# configure and build SPIRV-Tools, which only the Vulkan backend needs (it post-processes
# SPIR-V for tensor/data-graph capability rewriting). The D3D12 backend compiles no
# SPIR-V and carries premade DXIL, so building SPIRV-Tools for it would be dead weight --
# 200 MB of intermediate libraries for nothing.
if (FFX_GRAPHICS_API STREQUAL "dx12")
	add_subdirectory(${FFX_SRC_BACKENDS_PATH}/dx12)
endif()

option(BUILD_TOOLS "Build the tools" OFF)
message(STATUS "Build tools: ${BUILD_TOOLS}")
if(BUILD_TOOLS)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/ffx_shader_compiler)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/ffx_model_parser)
endif()

# -----------------------------------------------------------------------------
# NFRU data-graph end-to-end verification.
#
# Off by default: it is a test harness, not part of the SDK. When on it links against the
# SDK dll's import library and drives the backend INTERFACE (fpCreateBackendContext,
# fpCreateResource with FFX_RESOURCE_TYPE_TENSOR, fpCreateDataGraphPipeline,
# fpScheduleGpuJob, fpExecuteGpuJobs) with a golden input whose expected output was
# established against a CPU reference on real hardware.
#
# Placed here, after the backend subdirectory, so ${SDK_LIB_NAME} is already a known
# target and ${FFX_INCLUDE_PATH} and friends are in scope.
# -----------------------------------------------------------------------------
option(FFX_BUILD_NFRU_TEST "Build the NFRU data-graph verification harness" OFF)
if (FFX_BUILD_NFRU_TEST)
	message(STATUS "Building the NFRU data-graph verification harness")
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/nfru_datagraph)
endif()
