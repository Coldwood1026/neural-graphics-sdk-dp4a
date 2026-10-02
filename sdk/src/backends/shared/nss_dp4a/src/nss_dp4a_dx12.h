/*
 * nss_dp4a_dx12.h -- D3D12 host for the NSS int8 dp4a graph.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Internal to the SDK. The public C API in nss_dp4a.h is the Vulkan-facing one; this is
 * the D3D12 equivalent, and like the rest of this module it is compiled with
 * NSS_DP4A_INTERNAL so none of it reaches the host's export table.
 *
 * The shape deliberately mirrors the Vulkan API: create a context for one (quality,
 * resolution) pair, then record dispatches into the host's command list. `Record` only
 * records -- it never submits and never waits, so the graph lands in the same submission
 * as whatever pre- and post-processing the host records around it.
 * =============================================================================
 */
#ifndef NSS_DP4A_DX12_H
#define NSS_DP4A_DX12_H

#include "nss_dp4a.h"   /* NssDp4aQuality, NssDp4aDispatchInfo */

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct NssDx12 NssDx12;

typedef struct NssDx12CreateInfo
{
    uint64_t       device;    /* ID3D12Device* */
    NssDp4aQuality quality;
    uint32_t       width;     /* input width, must be a multiple of 8 */
    uint32_t       height;
} NssDx12CreateInfo;

/*
 * Builds the pipelines, allocates the intermediate tensors, and uploads the baked
 * weights. Unlike the Vulkan context this one may create its own command queue, because
 * D3D12 places no constraint on an application making additional queues -- which is what
 * lets the weight upload complete here without the host's involvement.
 *
 * Returns null on failure; *outError (optional) receives a static string.
 */
NssDx12* NssDx12Create(const NssDx12CreateInfo* createInfo, const char** outError);
void     NssDx12Destroy(NssDx12* ctx);

/*
 * Records the whole graph into `commandList` (an ID3D12GraphicsCommandList*).
 *
 * The three host buffers are raw int8 buffers, bound directly -- no copies in or out:
 *   input           int8 NHWC [height][width][12]
 *   outputKpn       int8 [height/4][width/4][36 high | 16 mid_low]
 *   outputTemporal  int8 [height][width][4]
 *
 * They must already be in D3D12_RESOURCE_STATE_UNORDERED_ACCESS. The context's own
 * intermediate buffers are transitioned once and then only UAV-barriered, since each
 * layer reads what the previous one wrote.
 */
bool NssDx12Record(NssDx12* ctx, uint64_t commandList, const NssDp4aDispatchInfo* info);

/* Bytes of internal scratch the context allocated, for host-side budgeting. */
uint64_t NssDx12InternalBytes(NssDx12* ctx);

#if defined(__cplusplus)
}
#endif

#endif /* NSS_DP4A_DX12_H */
