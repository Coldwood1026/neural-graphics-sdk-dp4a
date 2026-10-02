/*
 * nfru_dp4a.cpp -- public C API for the NFRU v1 dp4a inference backend.
 *
 * Thin layer over the API-specific Device implementations. Its job is argument
 * validation, lifetime management and choosing a backend; everything numeric lives in
 * the shaders and the graph.
 *
 * Backend selection: NfruDp4aCreateInfo::backend decides, because only the caller knows
 * whether it handed over a VkInstance or an ID3D12Device. NFRU_DP4A_BACKEND_AUTO keeps the
 * historical behaviour -- the `NFRU_DP4A_BACKEND=dx12` / `=vulkan` environment variable,
 * and Vulkan when that is unset -- which is what the standalone regression harness uses
 * when it runs the same graph through both APIs and compares the bytes.
 *
 * Interface style follows the FidelityFX convention (FfxErrorCode-style returns plus
 * descriptor structs plus an opaque context) so that `ffx_vk.cpp` can call it from
 * `executeGpuJobDataGraph` with minimal glue.
 */
#include "nfru_dp4a.h"
#include "nfru_device.h"
#include "nfru_vk.h"
#include "nfru_dx12.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

using namespace nfru;

namespace {

enum BackendKind { kBackendVulkan = 0, kBackendDx12 = 1 };

BackendKind ChooseBackend(const NfruDp4aCreateInfo* createInfo)
{
    /* Explicit request wins: the caller knows which API it is holding. */
    if (createInfo != nullptr) {
        if (createInfo->backend == NFRU_DP4A_BACKEND_DX12)   return kBackendDx12;
        if (createInfo->backend == NFRU_DP4A_BACKEND_VULKAN) return kBackendVulkan;
    }

    /* AUTO: the standalone harness selects the API this way. */
    const char* env = getenv("NFRU_DP4A_BACKEND");
    if (env) {
        if (_stricmp(env, "dx12") == 0 || _stricmp(env, "d3d12") == 0) return kBackendDx12;
        if (_stricmp(env, "vulkan") == 0 || _stricmp(env, "vk") == 0) return kBackendVulkan;
    }
    return kBackendVulkan;
}

}  /* namespace */

struct NfruDp4aContext {
    Device* device;
    BackendKind backend;
    VkLoader loader;
    bool ownVkModule;
    uint32_t width;
    uint32_t height;
    char err[256];
};

extern "C" {

NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aQueryDeviceCaps(uint64_t instance, uint64_t physicalDevice, uint32_t apiVersion,
                        void* vkGetInstanceProcAddr, NfruDp4aDeviceCaps* outCaps)
{
    return VkQueryDeviceCaps(instance, physicalDevice, apiVersion, vkGetInstanceProcAddr,
                             outCaps);
}

NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aCreateContext(const NfruDp4aCreateInfo* createInfo, NfruDp4aContext** outContext)
{
    if (!createInfo || !outContext) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }
    *outContext = nullptr;
    if (createInfo->device == 0) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }
    if (createInfo->width == 0 || createInfo->height == 0) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }

    NfruDp4aContext* ctx = new NfruDp4aContext();
    memset(ctx->err, 0, sizeof(ctx->err));
    ctx->backend = ChooseBackend(createInfo);
    ctx->width = createInfo->width;
    ctx->height = createInfo->height;
    ctx->device = nullptr;
    ctx->loader.module = nullptr;
    ctx->loader.gipa = nullptr;
    ctx->ownVkModule = false;

    const char* err = nullptr;

    if (ctx->backend == kBackendVulkan) {
        if (createInfo->vkGetInstanceProcAddr) {
            ctx->loader.gipa = (PFN_vkGetInstanceProcAddr)createInfo->vkGetInstanceProcAddr;
        } else if (!VkLoadLibrary(ctx->loader, &ctx->ownVkModule)) {
            snprintf(ctx->err, sizeof(ctx->err),
                     "no vkGetInstanceProcAddr supplied and vulkan-1.dll could not be "
                     "loaded");
            delete ctx;
            return NFRU_DP4A_ERROR_UNSUPPORTED;
        }
        ctx->device = VkCreateDevice(*createInfo, ctx->ownVkModule, ctx->loader, &err);
    } else {
        ctx->device = Dx12CreateDevice(*createInfo, &err);
    }

    if (!ctx->device) {
        snprintf(ctx->err, sizeof(ctx->err), "%s",
                 err ? err : "backend initialisation failed");
        if (ctx->ownVkModule) {
            VkUnloadLibrary(ctx->loader, true);
        }
        delete ctx;
        return NFRU_DP4A_ERROR_UNSUPPORTED;
    }

    *outContext = ctx;
    return NFRU_DP4A_OK;
}

NFRU_DP4A_API void NFRU_DP4A_CALL nfruDp4aDestroyContext(NfruDp4aContext* context)
{
    if (!context) {
        return;
    }
    if (context->device) {
        delete context->device;
        context->device = nullptr;
    }
    if (context->ownVkModule) {
        VkUnloadLibrary(context->loader, true);
    }
    delete context;
}

NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aRecord(NfruDp4aContext* context, uint64_t commandBuffer,
               const NfruDp4aDispatchInfo* dispatchInfo)
{
    if (!context || !context->device || !dispatchInfo || !commandBuffer) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }
    if (dispatchInfo->input.buffer == 0 || dispatchInfo->output.buffer == 0) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }
    const NfruDp4aResult r = context->device->record(
        commandBuffer,
        dispatchInfo->input.buffer, dispatchInfo->input.offset, dispatchInfo->input.size,
        dispatchInfo->output.buffer, dispatchInfo->output.offset, dispatchInfo->output.size,
        dispatchInfo->scratch.buffer,
        dispatchInfo->width, dispatchInfo->height);
    if (r != NFRU_DP4A_OK) {
        snprintf(context->err, sizeof(context->err), "%s", context->device->lastError());
    }
    return r;
}

NFRU_DP4A_API uint64_t NFRU_DP4A_CALL nfruDp4aGetScratchSize(const NfruDp4aContext* context)
{
    return (context && context->device) ? context->device->scratchBytes() : 0;
}

NFRU_DP4A_API uint64_t NFRU_DP4A_CALL
nfruDp4aGetInternalMemoryUsage(const NfruDp4aContext* context)
{
    return (context && context->device) ? context->device->internalBytes() : 0;
}

NFRU_DP4A_API void NFRU_DP4A_CALL nfruDp4aFlushPendingUploads(NfruDp4aContext* context)
{
    /* Weight uploads go through host-visible memory that is unmapped immediately, so
     * there is no staging to free. Present for interface parity with the NSS backend,
     * whose external-upload mode does need it. */
    (void)context;
}

NFRU_DP4A_API uint32_t NFRU_DP4A_CALL nfruDp4aGetVersion(void)
{
    return NFRU_DP4A_VERSION;
}

NFRU_DP4A_API const char* NFRU_DP4A_CALL nfruDp4aGetResultString(NfruDp4aResult result)
{
    switch (result) {
    case NFRU_DP4A_OK: return "ok";
    case NFRU_DP4A_ERROR_GENERIC: return "generic error";
    case NFRU_DP4A_ERROR_UNSUPPORTED: return "device or backend unsupported";
    case NFRU_DP4A_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case NFRU_DP4A_ERROR_OUT_OF_MEMORY: return "out of memory";
    case NFRU_DP4A_ERROR_VULKAN_FAILED: return "graphics API call failed";
    case NFRU_DP4A_ERROR_NOT_READY: return "context not ready";
    case NFRU_DP4A_ERROR_MODEL_MISMATCH: return "model or resolution mismatch";
    case NFRU_DP4A_ERROR_IO_FAILED: return "io failed";
    default: return "unknown";
    }
}

NFRU_DP4A_API uint32_t NFRU_DP4A_CALL nfruDp4aGetBackend(NfruDp4aContext* context)
{
    if (!context || !context->device) {
        return 0xffffffffu;
    }
    return (context->backend == kBackendDx12) ? 1u : 0u;
}

NFRU_DP4A_API const char* NFRU_DP4A_CALL nfruDp4aGetBackendName(NfruDp4aContext* context)
{
    if (!context || !context->device) {
        return "none";
    }
    return context->device->backendName();
}

NFRU_DP4A_API const char* NFRU_DP4A_CALL nfruDp4aGetLastError(NfruDp4aContext* context)
{
    if (!context) {
        return "no context";
    }
    if (context->device && context->device->lastError()[0]) {
        return context->device->lastError();
    }
    return context->err;
}

}  /* extern "C" */
