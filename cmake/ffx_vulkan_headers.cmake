# This file is part of the FidelityFX SDK.
#
# SPDX-License-Identifier: MIT
# =============================================================================
# Vulkan headers, without requiring a Vulkan SDK installation.
# =============================================================================
# The Vulkan backend is built with VK_NO_PROTOTYPES and resolves every entry point at
# runtime through the host's vkGetInstanceProcAddr (see vk_wrapper.cpp). It therefore
# needs the Vulkan *headers* and nothing else -- in particular it does not link an import
# library.
#
# CMake's own FindVulkan, however, treats Vulkan_LIBRARY as required. Since the headers
# are vendored in this repository under sdk/include/vulkan-headers, the previous
# find_package(Vulkan REQUIRED) made the entire SDK unconfigurable unless the LunarG SDK
# happened to be installed, while simultaneously not using anything from it.
#
# This helper resolves the vendored headers, reports the version the headers declare, and
# fails with an actionable message if they are missing.

set(VULKAN_HEADERS_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../sdk/include/vulkan-headers"
    CACHE PATH "Directory containing the 'vulkan' subdirectory of Vulkan headers")

function(ffx_require_vulkan_headers)
    if(NOT EXISTS "${VULKAN_HEADERS_INCLUDE_DIR}/vulkan/vulkan.h")
        message(FATAL_ERROR
            "Vulkan headers not found at '${VULKAN_HEADERS_INCLUDE_DIR}/vulkan'.\n"
            "They are vendored under sdk/include/vulkan-headers; if that directory is "
            "missing, re-checkout the repository, or pass "
            "-DVULKAN_HEADERS_INCLUDE_DIR=<path containing the 'vulkan' directory>.")
    endif()

    # Report the header version rather than the SDK version: it is what the build
    # actually compiles against.
    set(_vk_core "${VULKAN_HEADERS_INCLUDE_DIR}/vulkan/vulkan_core.h")
    file(STRINGS "${_vk_core}" _vk_complete_line REGEX "^#define VK_HEADER_VERSION_COMPLETE ")
    string(REGEX MATCHALL "[0-9]+" _vk_nums "${_vk_complete_line}")
    if(_vk_nums)
        list(JOIN _vk_nums "." VULKAN_HEADERS_VERSION)
    else()
        file(STRINGS "${_vk_core}" _vk_hdr_line REGEX "^#define VK_HEADER_VERSION ")
        string(REGEX MATCHALL "[0-9]+" _vk_hdr "${_vk_hdr_line}")
        set(VULKAN_HEADERS_VERSION "1.3.${_vk_hdr}")
    endif()
    set(VULKAN_HEADERS_VERSION "${VULKAN_HEADERS_VERSION}" PARENT_SCOPE)

    # Keep the usual CMake variable names populated so existing references keep working.
    set(Vulkan_INCLUDE_DIR "${VULKAN_HEADERS_INCLUDE_DIR}" PARENT_SCOPE)
    set(Vulkan_INCLUDE_DIRS "${VULKAN_HEADERS_INCLUDE_DIR}" PARENT_SCOPE)
    set(Vulkan_VERSION "${VULKAN_HEADERS_VERSION}" PARENT_SCOPE)
    set(Vulkan_FOUND TRUE PARENT_SCOPE)
endfunction()
