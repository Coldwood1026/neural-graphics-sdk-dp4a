/*
 * ffx_dx12_compat.h -- name compatibility with AMD FidelityFX SDK 1.1.3.
 *
 * SPDX-License-Identifier: MIT
 * The Arm fork renamed two things the imported D3D12 backend uses, without changing
 * either value:
 *
 *   FFX_RESOURCE_STATE_UNORDERED_ACCESS  ->  FFX_RESOURCE_STATE_COMPUTE_UAV
 *       Both are (1 << 1); the fork split the single UAV state into
 *       COMPUTE_UAV / PIXEL_UAV / GENERIC_UAV.
 *
 *   FFX_RING_BUFFER_DESCRIPTOR_COUNT     ->  FFX_RING_BUFFER_SIZE
 *       Both are (FFX_MAX_QUEUED_FRAMES * FFX_MAX_PASS_COUNT * FFX_MAX_RESOURCE_COUNT).
 *
 * Both are defined here rather than in the public headers so the divergence stays
 * visible in one place, and so a Vulkan build is unaffected.
 */
#ifndef FFX_DX12_COMPAT_H
#define FFX_DX12_COMPAT_H

#include <FidelityFX/host/ffx_types.h>

#ifndef FFX_RESOURCE_STATE_UNORDERED_ACCESS
#define FFX_RESOURCE_STATE_UNORDERED_ACCESS FFX_RESOURCE_STATE_COMPUTE_UAV
#endif
#ifndef FFX_RING_BUFFER_DESCRIPTOR_COUNT
#define FFX_RING_BUFFER_DESCRIPTOR_COUNT FFX_RING_BUFFER_SIZE
#endif

#endif /* FFX_DX12_COMPAT_H */
