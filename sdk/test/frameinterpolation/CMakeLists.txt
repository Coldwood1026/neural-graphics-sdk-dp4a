# This file is part of the FidelityFX SDK.
#
# SPDX-License-Identifier: MIT
# =============================================================================
# Execution check for Arm's neural frame interpolation.
# =============================================================================
# Builds one harness per graphics API. It links against the SDK dll's import library, so
# what it exercises is the SHIPPED ARTIFACT rather than a private copy of the backend -- the
# ffxFrameInterpolation* entry points and ffxGetInterfaceVK / ffxGetInterfaceDX12 are part of
# the SDK's export table.
#
# What it establishes: that the effect creates its ten pipelines, prepares, dispatches and
# destroys without error, which means the fragment passes really are drawn. Before this
# integration the D3D12 backend could not create a graphics pipeline at all, so the context
# could not even be created.
#
# What it does NOT establish: that the interpolated frame is correct. There is no reference
# image; see the note at the top of fi_sdk_test.cpp.
# =============================================================================

message(STATUS "Configure sdk/test/frameinterpolation")

set(FI_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fi_sdk_test.cpp")

# The Vulkan half initialises the backend's process-wide function table by going through
# ffxCreateContext with a VK backend desc -- InitVulkanWrapper() is what populates it and it
# is not exported. That needs the ffx-api headers, which live in their own tree.
get_filename_component(FFX_API_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../ffx-api/include" ABSOLUTE)

if (FFX_GRAPHICS_API STREQUAL "vk")
    add_executable(fi_sdk_test_vk ${FI_TEST_SOURCE})
    target_compile_definitions(fi_sdk_test_vk PRIVATE FI_TEST_VK)
    target_include_directories(fi_sdk_test_vk PRIVATE
        ${FFX_INCLUDE_PATH}
        # FFX_INCLUDE_PATH is <sdk>/include, so this is exact rather than a relative climb
        # that silently lands one level off.
        "${FFX_INCLUDE_PATH}/vulkan-headers"
        ${FFX_API_INCLUDE_PATH}
    )
    target_link_libraries(fi_sdk_test_vk PRIVATE ${SDK_LIB_NAME})
    set_target_properties(fi_sdk_test_vk PROPERTIES FOLDER Tests)
elseif (FFX_GRAPHICS_API STREQUAL "dx12")
    add_executable(fi_sdk_test_dx12 ${FI_TEST_SOURCE})
    target_compile_definitions(fi_sdk_test_dx12 PRIVATE FI_TEST_DX12 NOMINMAX WIN32_LEAN_AND_MEAN)
    target_include_directories(fi_sdk_test_dx12 PRIVATE ${FFX_INCLUDE_PATH})
    target_link_libraries(fi_sdk_test_dx12 PRIVATE ${SDK_LIB_NAME} d3d12 dxgi)
    set_target_properties(fi_sdk_test_dx12 PROPERTIES FOLDER Tests)
endif()
