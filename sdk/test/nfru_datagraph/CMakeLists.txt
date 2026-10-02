# This file is part of the FidelityFX SDK.
#
# SPDX-License-Identifier: MIT
# =============================================================================
# SDK-level end-to-end verification of the NFRU data graph.
# =============================================================================
# Builds one harness per graphics API. It links against the SDK dll's import library, so
# what it exercises is the SHIPPED ARTIFACT and not a private copy of the backend -- the
# backend interface entry points (ffxGetInterfaceVK / ffxGetInterfaceDX12) are part of the
# SDK's export table.
#
# If the SDK's own scratch-buffer sizing, tensor handling, data-graph pipeline creation or
# job dispatch were wrong, this is where it shows up.
# =============================================================================

message(STATUS "Configure sdk/test/nfru_datagraph")

set(NFRU_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/nfru_sdk_datagraph_test.cpp")

if (FFX_GRAPHICS_API STREQUAL "vk")
    add_executable(nfru_sdk_test_vk ${NFRU_TEST_SOURCE})
    target_compile_definitions(nfru_sdk_test_vk PRIVATE NFRU_TEST_VK)
    target_include_directories(nfru_sdk_test_vk PRIVATE
        ${FFX_INCLUDE_PATH}
        # FFX_INCLUDE_PATH is <sdk>/include, so this is exact rather than a relative
        # climb that silently lands one level off.
        "${FFX_INCLUDE_PATH}/vulkan-headers"
    )
    target_link_libraries(nfru_sdk_test_vk PRIVATE ${SDK_LIB_NAME})
    set_target_properties(nfru_sdk_test_vk PROPERTIES FOLDER Tests)
elseif (FFX_GRAPHICS_API STREQUAL "dx12")
    add_executable(nfru_sdk_test_dx12 ${NFRU_TEST_SOURCE})
    target_compile_definitions(nfru_sdk_test_dx12 PRIVATE NFRU_TEST_DX12 NOMINMAX WIN32_LEAN_AND_MEAN)
    target_include_directories(nfru_sdk_test_dx12 PRIVATE ${FFX_INCLUDE_PATH})
    target_link_libraries(nfru_sdk_test_dx12 PRIVATE ${SDK_LIB_NAME} d3d12 dxgi)
    set_target_properties(nfru_sdk_test_dx12 PROPERTIES FOLDER Tests)
endif()
