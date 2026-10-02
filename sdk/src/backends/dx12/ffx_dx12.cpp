// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_util.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#include <FidelityFX/host/backends/dx12/d3dx12.h>
#include <ffx_shader_blobs.h>
#include <nfru_dp4a.h>
#include <nss_dp4a_dx12.h>
#include <ffx_breadcrumbs_list.h>
#include <codecvt>  // convert string to wstring
#include <memoryapi.h> // for VirtualAlloc
#include <mutex>
#include <tuple> // std::ignore
#include <limits> // std::numeric_limits

/*
 * =============================================================================
 * NAME COMPATIBILITY WITH AMD FIDELITYFX SDK 1.1.3
 * =============================================================================
 * This file IS the 1.1.3 D3D12 backend. The Arm fork renamed two things it uses,
 * without changing either value, so the mapping is exact and belongs in one place
 * rather than spread over hundreds of call sites:
 *
 *   FFX_RESOURCE_STATE_UNORDERED_ACCESS  ->  FFX_RESOURCE_STATE_COMPUTE_UAV
 *       Both are (1 << 1). The fork split the single 1.1.3 UAV state into
 *       COMPUTE_UAV / PIXEL_UAV / GENERIC_UAV; a D3D12 backend wants the compute one.
 *
 *   FFX_RING_BUFFER_DESCRIPTOR_COUNT    ->  FFX_RING_BUFFER_SIZE
 *       Both are (FFX_MAX_QUEUED_FRAMES * FFX_MAX_PASS_COUNT * FFX_MAX_RESOURCE_COUNT).
 *
 * Defining them locally keeps the public headers clean and makes the divergence
 * visible in one place instead of silently papered over.
 */
#ifndef FFX_RESOURCE_STATE_UNORDERED_ACCESS
#define FFX_RESOURCE_STATE_UNORDERED_ACCESS FFX_RESOURCE_STATE_COMPUTE_UAV
#endif
#ifndef FFX_RING_BUFFER_DESCRIPTOR_COUNT
#define FFX_RING_BUFFER_DESCRIPTOR_COUNT FFX_RING_BUFFER_SIZE
#endif

/*
 * Narrow (SDK) -> wide (Win32/D3D12).
 *
 * The fork changed every SDK name field from wchar_t to char -- FfxResource::name,
 * FfxResourceBinding::name, FfxPipelineState::name, FfxCreateResourceDescription::name,
 * FfxGpuJobDescription::jobLabel. This backend, written against 1.1.3, still keeps its
 * OWN wide Resource::resourceName[64], and D3D12's ID3D12Object::SetName takes LPCWSTR
 * regardless. So the conversion belongs at each boundary rather than on either side.
 */
static void ffxNarrowToWide(wchar_t* dst, size_t dstCount, const char* src)
{
    if (dst == nullptr || dstCount == 0)
    {
        return;
    }
    dst[0] = L'\0';
    if (src == nullptr || src[0] == '\0')
    {
        return;
    }
    MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, (int)dstCount);
    dst[dstCount - 1] = L'\0';
}

// Disable this to remove the dll load of PIX and PIX tracing
//
// Set to 0 for this tree: the WinPixEventRuntime is vendored at sdk/libs/pix in AMD's
// FidelityFX SDK, but the Arm fork dropped sdk/libs/pix along with the D3D12 backend, so
// the "pix/pix3.h" include below (the _GAMING_DESKTOP-less branch, which is the live one
// here) would not resolve, and the `pixlib` import target it links against does not
// exist either. PIX capture is a profiling nicety; the backend does not depend on it.
#define ENABLE_PIX_CAPTURES 0

#if ENABLE_PIX_CAPTURES
// PIX instrumentation is only enabled if one of the preprocessor symbols USE_PIX, DBG, _DEBUG, PROFILE, or PROFILE_BUILD is defined.
// ref: https://devblogs.microsoft.com/pix/winpixeventruntime/
#ifndef USE_PIX
#define USE_PIX  // Should enable it at anytime, as we already have a runtime switch for this purpose
#endif           // #ifndef USE_PIX

#ifdef _GAMING_DESKTOP
	#include <pix3.h>
#else
	#include "pix/pix3.h"
#endif // _GAMING_DESKTOP


#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wpointer-bool-conversion"
#endif

static bool s_PIXDLLLoaded = false;

typedef void(WINAPI* BeginEventOnCommandList)(ID3D12GraphicsCommandList* commandList, UINT64 color, _In_ PCSTR formatString);
typedef void(WINAPI* EndEventOnCommandList)(ID3D12GraphicsCommandList* commandList);

BeginEventOnCommandList pixBeginEventOnCommandList;
EndEventOnCommandList   pixEndEventOnCommandList;

#endif // #if ENABLE_PIX_CAPTURES

// DX12 prototypes for functions in the backend interface
FfxVersionNumber GetSDKVersionDX12(FfxInterface* backendInterface);
FfxErrorCode GetEffectGpuMemoryUsageDX12(FfxInterface* backendInterface, FfxUInt32 effectContextId, FfxEffectMemoryUsage* outVramUsage);
FfxErrorCode CreateBackendContextDX12(FfxInterface* backendInterface, FfxEffect effect, FfxEffectBindlessConfig* bindlessConfig, FfxUInt32* effectContextId);
FfxErrorCode GetDeviceCapabilitiesDX12(FfxInterface* backendInterface, FfxDeviceCapabilities* deviceCapabilities);
FfxErrorCode DestroyBackendContextDX12(FfxInterface* backendInterface, FfxUInt32 effectContextId);
FfxErrorCode CreateResourceDX12(FfxInterface* backendInterface, const FfxCreateResourceDescription* desc, FfxUInt32 effectContextId, FfxResourceInternal* outTexture);
FfxErrorCode DestroyResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource, FfxUInt32 effectContextId);
FfxErrorCode MapResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource, void** ptr);
FfxErrorCode UnmapResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode RegisterResourceDX12(FfxInterface* backendInterface, const FfxResource* inResource, FfxUInt32 effectContextId, FfxResourceInternal* outResourceInternal);
FfxResource GetResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode UnregisterResourcesDX12(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId);
FfxErrorCode RegisterStaticResourceDX12(FfxInterface* backendInterface, const FfxStaticResourceDescription* desc, FfxUInt32 effectContextId);
FfxResourceDescription GetResourceDescriptorDX12(FfxInterface* backendInterface, FfxResourceInternal resource);
FfxErrorCode StageConstantBufferDataDX12(FfxInterface* backendInterface, void* data, FfxUInt32 size, FfxConstantBuffer* constantBuffer);
FfxErrorCode CreatePipelineDX12(FfxInterface* backendInterface, FfxEffect effect, FfxPass passId, uint32_t permutationOptions, const FfxPipelineDescription*  desc, FfxUInt32 effectContextId, FfxPipelineState* outPass);
FfxErrorCode DestroyPipelineDX12(FfxInterface* backendInterface, FfxPipelineState* pipeline, FfxUInt32 effectContextId);
FfxErrorCode ScheduleGpuJobDX12(FfxInterface* backendInterface, const FfxGpuJobDescription* job);
FfxErrorCode ExecuteGpuJobsDX12(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId);
FfxErrorCode BreadcrumbsAllocBlockDX12(FfxInterface* backendInterface, uint64_t blockBytes, FfxBreadcrumbsBlockData* blockData);
void BreadcrumbsFreeBlockDX12(FfxInterface* backendInterface, FfxBreadcrumbsBlockData* blockData);
void BreadcrumbsWriteDX12(FfxInterface* backendInterface, FfxCommandList commandList, uint32_t value, uint64_t gpuLocation, void* gpuBuffer, bool isBegin);
void BreadcrumbsPrintDeviceInfoDX12(FfxInterface* backendInterface, FfxAllocationCallbacks* allocs, bool extendedInfo, char** printBuffer, size_t* printSize);
void RegisterConstantBufferAllocatorDX12(FfxInterface* backendInterface, FfxConstantBufferAllocator fpConstantAllocator);

#define FFX_MAX_RESOURCE_IDENTIFIER_COUNT   (128)
#define FFX_MAX_STATIC_DESCRIPTOR_COUNT   (65536)

// Constant buffer allocation callback
static FfxConstantBufferAllocator s_fpConstantAllocator = nullptr;

#if 0 // Now defined in header
typedef struct BackendContext_DX12 {

    // store for resources and resourceViews
    typedef struct Resource
    {
#ifdef _DEBUG
        wchar_t                 resourceName[64] = {};
#endif
        ID3D12Resource*         resourcePtr;
        FfxResourceDescription  resourceDescription;
        FfxResourceStates       initialState;
        FfxResourceStates       currentState;
        uint32_t                srvDescIndex;
        uint32_t                uavDescIndex;
        uint32_t                uavDescCount;
    } Resource;

    uint32_t refCount;
    uint32_t maxEffectContexts;

    ID3D12Device*           device = nullptr;

    FfxGpuJobDescription*   pGpuJobs;
    uint32_t                gpuJobCount;

    uint32_t                nextRtvDescriptor;
    ID3D12DescriptorHeap*   descHeapRtvCpu;

    ID3D12DescriptorHeap*   descHeapSrvCpu;
    ID3D12DescriptorHeap*   descHeapUavCpu;
    ID3D12DescriptorHeap*   descHeapUavGpu;

    uint32_t                descRingBufferSize;
    uint32_t                descRingBufferBase;
    ID3D12DescriptorHeap*   descRingBuffer;
    uint32_t                descBindlessBase;

    uint8_t*                pStagingRingBuffer;
    uint32_t                stagingRingBufferBase = 0;

    D3D12_RESOURCE_BARRIER  barriers[FFX_MAX_BARRIERS];
    uint32_t                barrierCount;

    IDXGIFactory*           dxgiFactory = nullptr;

    typedef struct alignas(32) EffectContext {

        // Effect identifier -- used for various resource callbacks to application
        FfxEffect           effectId;

        // Resource allocation
        uint32_t            nextStaticResource;
        uint32_t            nextDynamicResource;

        // UAV offsets
        uint32_t            nextStaticUavDescriptor;
        uint32_t            nextDynamicUavDescriptor;

        // Bindless heap
        uint32_t            bindlessTextureSrvHeapStart;
        uint32_t            bindlessTextureSrvHeapSize;
        uint32_t            bindlessBufferSrvHeapStart;
        uint32_t            bindlessBufferSrvHeapSize;
        uint32_t            bindlessTextureUavHeapStart;
        uint32_t            bindlessTextureUavHeapSize;
        uint32_t            bindlessBufferUavHeapStart;
        uint32_t            bindlessBufferUavHeapSize;

        uint32_t bindlessBufferHeapStart;
        uint32_t bindlessBufferHeapEnd;

        // Usage
        bool                active;

        // VRAM usage
        FfxEffectMemoryUsage vramUsage;

    } EffectContext;

    // Resource holder
    Resource*                   pResources;
    EffectContext*              pEffectContexts;

    // Allocation defaults
    FfxConstantAllocation       FallbackConstantAllocator(void* data, FfxUInt64 dataSize);
    void*                       constantBufferMem;
    ID3D12Resource*             constantBufferResource;
    uint32_t                    constantBufferSize;
    uint32_t                    constantBufferOffset;
    std::mutex                  constantBufferMutex;

} BackendContext_DX12;
#endif

static uint32_t getFreeBindlessDescriptorBlock(BackendContext_DX12 *context, uint32_t size, uint32_t effectId)
{
    uint32_t base = context->descBindlessBase;

    for (uint32_t i = 0; i < context->maxEffectContexts; ++i) {
        BackendContext_DX12::EffectContext *effectContext = &context->pEffectContexts[i];
        if (i == effectId || !effectContext->active) {
            continue;
        }

        if (!(base >= effectContext->bindlessBufferHeapEnd || base + size <= effectContext->bindlessBufferHeapStart)) {
            base = effectContext->bindlessBufferHeapEnd;
            i = 0;
        }
    }

    FFX_ASSERT(base + size <= context->descBindlessBase + FFX_MAX_STATIC_DESCRIPTOR_COUNT);

    return base;
}

FFX_API size_t ffxGetScratchMemorySizeDX12(size_t maxContexts)
{
    uint32_t resourceArraySize          = FFX_ALIGN_UP(maxContexts * FFX_MAX_RESOURCE_COUNT * sizeof(BackendContext_DX12::Resource), sizeof(uint64_t));
    uint32_t contextArraySize           = FFX_ALIGN_UP(maxContexts * sizeof(BackendContext_DX12::EffectContext), sizeof(uint32_t));
    uint32_t stagingRingBufferArraySize = FFX_ALIGN_UP(maxContexts * FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE, sizeof(uint32_t));
    uint32_t gpuJobDescArraySize        = FFX_ALIGN_UP(maxContexts * FFX_MAX_GPU_JOBS * sizeof(FfxGpuJobDescription), sizeof(uint32_t));

    /*
     * Pipeline records.
     *
     * Neither of these was in the size calculation before. pDataGraphPipelines was declared
     * in the backend context, written through by CreateDataGraphPipelineDX12, and never
     * allocated -- it stayed null, so that write was to address zero. It did not fire
     * because fpCreateDataGraphPipeline is unreachable in this build: the NFRU and NSS
     * modules drive their own dp4a hosts directly. It would have fired the moment anything
     * created a data-graph pipeline through the interface, so it is fixed here rather than
     * left as a landmine.
     *
     * The graphics records are new and are what frame interpolation's fragment passes
     * need.
     */
    uint32_t dataGraphPipelineArraySize = FFX_ALIGN_UP(maxContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_DX12::DataGraphPipelineDX12), sizeof(uint64_t));
    uint32_t graphicsPipelineArraySize  = FFX_ALIGN_UP(maxContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_DX12::GraphicsPipelineDX12), sizeof(uint64_t));

    return FFX_ALIGN_UP(sizeof(BackendContext_DX12) + resourceArraySize + contextArraySize + stagingRingBufferArraySize + gpuJobDescArraySize
                            + dataGraphPipelineArraySize + graphicsPipelineArraySize,
                        sizeof(uint64_t));
}

// Create a FfxDevice from a ID3D12Device*
FfxDevice ffxGetDeviceDX12(ID3D12Device* dx12Device)
{
    FFX_ASSERT(NULL != dx12Device);
    return reinterpret_cast<FfxDevice>(dx12Device);
}

/*
 * =============================================================================
 * Portable dp4a inference backend -- data-graph pipelines.
 * =============================================================================
 * The exact D3D12 counterpart of CreateDataGraphPipelineVK / executeGpuJobDataGraph in
 * ffx_vk.cpp. Both APIs run ONE implementation of the NFRU graph, which is why the two
 * backends can be -- and were -- verified to produce byte-identical output.
 *
 * What this replaces is the entire VK_ARM_data_graph execution path: no data-graph
 * pipeline object, no session bound to its own device memory, no tensor descriptors, no
 * vkCmdDispatchDataGraphARM. On D3D12 there was never such a path at all (the Arm fork
 * deleted this backend), so there is nothing to remove here -- only something to add.
 */
static FfxErrorCode CreateDataGraphPipelineDX12(
    FfxInterface*                 backendInterface,
    FfxEffect                     effect,
    FfxPass                       passId,
    uint32_t                      permutationOptions,
    const FfxPipelineDescription* desc,
    FfxUInt32                     effectContextId,
    FfxUInt32                     dataGraphWidth,
    FfxUInt32                     dataGraphHeight,
    FfxPipelineState*             outPipeline)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(desc != nullptr);
    FFX_ASSERT(NULL != outPipeline);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    // The blob's tensor tables are still the interface between the SDK and the graph: the
    // effect code joins them BY NAME against its own resource tables
    // (patchResourceIdentifier), so those strings and their order are copied through
    // unchanged. The weights themselves are baked into the backend, so nothing here reads
    // a model file.
    FfxDataGraphBlob dataGraphBlob = {};
    FFX_VALIDATE(backendInterface->fpGetPermutationBlobByIndex(
        effect, passId, permutationOptions, nullptr, nullptr, &dataGraphBlob));

    const FfxUInt32 srvTensorCount = dataGraphBlob.inputTensorNums;
    const FfxUInt32 uavTensorCount = dataGraphBlob.outputTensorNums;

    FFX_ASSERT(srvTensorCount > 0 && uavTensorCount > 0);
    FFX_ASSERT(srvTensorCount <= FFX_MAX_RESOURCE_COUNT && uavTensorCount <= FFX_MAX_RESOURCE_COUNT);

    for (uint32_t tensorIndex = 0; tensorIndex < srvTensorCount; ++tensorIndex)
    {
        outPipeline->srvTensorBindings[tensorIndex].slotIndex  = dataGraphBlob.inputTensorBindings[tensorIndex];
        outPipeline->srvTensorBindings[tensorIndex].arrayIndex = 0;
        strncpy(outPipeline->srvTensorBindings[tensorIndex].name,
                dataGraphBlob.inputTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->srvTensorBindings[tensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
    outPipeline->srvTensorCount = srvTensorCount;

    for (uint32_t tensorIndex = 0; tensorIndex < uavTensorCount; ++tensorIndex)
    {
        outPipeline->uavTensorBindings[tensorIndex].slotIndex  = dataGraphBlob.outputTensorBindings[tensorIndex];
        outPipeline->uavTensorBindings[tensorIndex].arrayIndex = 0;
        strncpy(outPipeline->uavTensorBindings[tensorIndex].name,
                dataGraphBlob.outputTensorNames[tensorIndex], FFX_RESOURCE_NAME_SIZE - 1);
        outPipeline->uavTensorBindings[tensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
    outPipeline->uavTensorCount = uavTensorCount;

    const char* entryPoint = (dataGraphBlob.graphEntryPoint != nullptr) ? dataGraphBlob.graphEntryPoint : "";

    FFX_ASSERT_MESSAGE(backendContext->nextDataGraphPipeline < FFX_MAX_PASS_COUNT,
                       "FFXInterface: DX12: ran out of data-graph pipeline records. Please increase FFX_MAX_PASS_COUNT");
    BackendContext_DX12::DataGraphPipelineDX12* pRecord =
        &backendContext->pDataGraphPipelines[backendContext->nextDataGraphPipeline++];

    pRecord->dp4a            = nullptr;
    pRecord->width           = dataGraphWidth;
    pRecord->height          = dataGraphHeight;
    pRecord->effectContextId = effectContextId;
    strncpy(pRecord->entryPoint, entryPoint, sizeof(pRecord->entryPoint) - 1);
    pRecord->entryPoint[sizeof(pRecord->entryPoint) - 1] = '\0';

    // -------------------------------------------------------------------------
    // Build the dp4a context for the models this backend implements.
    //
    // Only NFRU v1 is implemented. NSS's graphs are a DIFFERENT network: they consume the
    // preprocessed tensor and emit KPN coefficients plus a temporal feedback tensor.
    // Running NFRU's weights against them would silently produce garbage, so an
    // unimplemented model is refused HERE and reported again at dispatch time, rather than
    // approximated.
    // -------------------------------------------------------------------------
    /*
     * NSS. Two models, one per quality tier. The D3D12 host is driven entirely from
     * nss::Model::dispatches(), which is API-agnostic -- see nss_dp4a_dx12.cpp -- so
     * unlike the Vulkan side there is nothing to defer here: the D3D12 context creates
     * its own queue for the one-shot weight upload and completes it inside Create.
     */
    if (strcmp(entryPoint, "nss_v1_0_1_high_int8") == 0 ||
        strcmp(entryPoint, "nss_v1_0_1_mid_low_int8") == 0)
    {
        NssDx12CreateInfo ci = {};
        ci.device  = (uint64_t)backendContext->device;
        ci.quality = (strcmp(entryPoint, "nss_v1_0_1_high_int8") == 0)
                         ? NSS_DP4A_QUALITY_HIGH
                         : NSS_DP4A_QUALITY_MID_LOW;
        ci.width   = dataGraphWidth;
        ci.height  = dataGraphHeight;

        const char* nssError = nullptr;
        NssDx12* nss = NssDx12Create(&ci, &nssError);
        if (nss == nullptr)
        {
            if (nssError != nullptr) { FFX_ASSERT_MESSAGE(false, nssError); }
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        pRecord->dp4a = nss;
    }
    else if (strcmp(entryPoint, "nfru_v1_int8") == 0)
    {
        NfruDp4aCreateInfo ci = {};
        // The DX12 device reads the ID3D12Device from ::device. There is no queue to give
        // it: this backend only RECORDS into the host's command list, never submits and
        // never waits on a fence, and the SDK's DX12 backend holds no queue either.
        ci.device   = (uint64_t)backendContext->device;
        ci.instance = (uint64_t)backendContext->device;
        ci.queue    = 0;
        ci.width    = dataGraphWidth;
        ci.height   = dataGraphHeight;
        /*
         * Say which API this is. Without it the library falls back to NFRU_DP4A_BACKEND
         * and then to Vulkan, treats `instance` -- which is the ID3D12Device on purpose,
         * see the comment above -- as a VkInstance, and dies inside VkCreateDevice.
         */
        ci.backend  = NFRU_DP4A_BACKEND_DX12;

        NfruDp4aContext* dp4a = nullptr;
        if (nfruDp4aCreateContext(&ci, &dp4a) != NFRU_DP4A_OK)
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        pRecord->dp4a = dp4a;
    }

    strncpy(outPipeline->name, (desc->name != nullptr) ? desc->name : "dp4a data graph",
            FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    // No ID3D12PipelineState: the backend owns its pipelines, root signature and barriers
    // and records them itself. The record doubles as the identity the dispatch path
    // recovers from FfxPipelineState::rootSignature.
    outPipeline->pipeline      = nullptr;
    outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pRecord);

    return FFX_OK;
}

static FfxErrorCode executeGpuJobDataGraphDX12(
    BackendContext_DX12*       backendContext,
    FfxGpuJobDescription*      job,
    ID3D12GraphicsCommandList* dx12CommandList)
{
    BackendContext_DX12::DataGraphPipelineDX12* pRecord =
        reinterpret_cast<BackendContext_DX12::DataGraphPipelineDX12*>(job->dataGraphJobDescription.pipeline.rootSignature);

    /*
     * NSS first. Four outputs versus NFRU's one, and a completely different intermediate
     * tensor layout, so the only thing the two paths share is the record that carries
     * their context.
     */
    if (strcmp(pRecord->entryPoint, "nss_v1_0_1_high_int8") == 0 ||
        strcmp(pRecord->entryPoint, "nss_v1_0_1_mid_low_int8") == 0)
    {
        NssDx12* nss = reinterpret_cast<NssDx12*>(pRecord->dp4a);
        if (nss == nullptr)
        {
            FFX_ASSERT_MESSAGE(false, "this data-graph model has no dp4a implementation (see CreateDataGraphPipelineDX12)");
            return FFX_ERROR_BACKEND_API_ERROR;
        }

        FFX_ASSERT_MESSAGE(job->dataGraphJobDescription.pipeline.srvTensorCount >= 1,
                           "the NSS dp4a graph takes one input tensor");
        FFX_ASSERT_MESSAGE(job->dataGraphJobDescription.pipeline.uavTensorCount >= 2,
                           "the NSS dp4a graph produces KPN and temporal feedback");

        FfxTensor& inTensor  = job->dataGraphJobDescription.srvTensors[0];
        FfxTensor& kpnTensor = job->dataGraphJobDescription.uavTensors[0];
        FfxTensor& tmpTensor = job->dataGraphJobDescription.uavTensors[1];

        /* Same bridge as NFRU: on D3D12 a tensor was created as a plain buffer in the
         * first place (see the FFX_RESOURCE_TYPE_TENSOR case in CreateResourceDX12), so
         * resourcePtr IS the storage and nothing has to be unwrapped. */
        ID3D12Resource* inResource  = backendContext->pResources[inTensor.resource.internalIndex].resourcePtr;
        ID3D12Resource* kpnResource = backendContext->pResources[kpnTensor.resource.internalIndex].resourcePtr;
        ID3D12Resource* tmpResource = backendContext->pResources[tmpTensor.resource.internalIndex].resourcePtr;
        FFX_ASSERT_MESSAGE(inResource != nullptr && kpnResource != nullptr && tmpResource != nullptr,
                           "an NSS data-graph tensor has no D3D12 resource, so the dp4a backend cannot reach its storage");

        const uint32_t width  = pRecord->width;
        const uint32_t height = pRecord->height;
        /* KPN is 6x6 = 36 channels for HIGH and 4x4 = 16 for MID_LOW. Compared by string
         * rather than by peeking at a character offset: "nss_v1_0_1_high_int8" and
         * "nss_v1_0_1_mid_low_int8" first differ at index 11, not 10, and an off-by-one
         * there would silently give HIGH a 16-channel output buffer. */
        const uint32_t kpnChannels =
            (strcmp(pRecord->entryPoint, "nss_v1_0_1_high_int8") == 0) ? 36u : 16u;

        NssDp4aDispatchInfo di = {};
        di.input.buffer          = (uint64_t)inResource;
        di.input.offset          = 0;
        di.input.size            = (uint64_t)width * height * 12u;
        di.outputKpn.buffer      = (uint64_t)kpnResource;
        di.outputKpn.offset      = 0;
        di.outputKpn.size        = (uint64_t)(width / 4u) * (height / 4u) * kpnChannels;
        di.outputTemporal.buffer = (uint64_t)tmpResource;
        di.outputTemporal.offset = 0;
        di.outputTemporal.size   = (uint64_t)width * height * 4u;
        di.width                 = width;
        di.height                = height;

        if (!NssDx12Record(nss, (uint64_t)dx12CommandList, &di))
        {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        return FFX_OK;
    }

    NfruDp4aContext* dp4a = reinterpret_cast<NfruDp4aContext*>(pRecord->dp4a);
    if (dp4a == nullptr)
    {
        // The model was refused at pipeline creation because this backend does not
        // implement it. Fail loudly: falling through would run the wrong weights.
        FFX_ASSERT_MESSAGE(false, "This data-graph model has no dp4a implementation (see CreateDataGraphPipelineDX12)");
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    FFX_ASSERT_MESSAGE(job->dataGraphJobDescription.pipeline.srvTensorCount >= 1,
                       "the dp4a data graph takes at least one input tensor");
    FFX_ASSERT_MESSAGE(job->dataGraphJobDescription.pipeline.uavTensorCount >= 1,
                       "the dp4a data graph produces at least one output tensor");

    FfxTensor& inTensor  = job->dataGraphJobDescription.srvTensors[0];
    FfxTensor& outTensor = job->dataGraphJobDescription.uavTensors[0];

    FFX_ASSERT(inTensor.resource.internalIndex != 0 && outTensor.resource.internalIndex != 0);

    /*
     * The bridge. On Vulkan the tensors are VK_ARM tensor objects and the int8 payload is
     * reached through the buffer they alias; here a tensor was created as a plain buffer
     * in the first place (see the FFX_RESOURCE_TYPE_TENSOR case in CreateResourceDX12), so
     * resourcePtr IS the storage. The dp4a module builds its own raw SRV/UAV and does its
     * own transitions, so nothing else has to be prepared.
     */
    ID3D12Resource* inResource  = backendContext->pResources[inTensor.resource.internalIndex].resourcePtr;
    ID3D12Resource* outResource = backendContext->pResources[outTensor.resource.internalIndex].resourcePtr;
    FFX_ASSERT_MESSAGE(inResource != nullptr && outResource != nullptr,
                       "a data-graph tensor has no D3D12 resource, so the dp4a backend cannot reach its storage");

    const uint32_t width  = pRecord->width;
    const uint32_t height = pRecord->height;

    NfruDp4aDispatchInfo di = {};
    di.input.buffer  = (uint64_t)inResource;
    di.input.offset  = 0;
    // NFRU v1 consumes 16 int8 channels and emits 4 logits, in NHWC. The D3D12 path builds
    // its views from these sizes, so they are stated rather than left zero.
    di.input.size    = (uint64_t)width * height * 16u;
    di.output.buffer = (uint64_t)outResource;
    di.output.offset = 0;
    di.output.size   = (uint64_t)width * height * 4u;
    di.width         = width;
    di.height        = height;

    if (nfruDp4aRecord(dp4a, (uint64_t)dx12CommandList, &di) != NFRU_DP4A_OK)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    return FFX_OK;
}
/*
 * Graphics pipelines.
 *
 * The imported AMD 1.1.3 backend creates only COMPUTE pipeline states -- there is no
 * CreateGraphicsPipelineState call anywhere in it -- and it reached the shader-blob
 * provider with a hardcoded FFX_BIND_COMPUTE_SHADER_STAGE, so it could not have built one.
 * The Arm fork, meanwhile, split the interface into fpCreateComputePipeline and
 * fpCreateGraphicsPipeline, and frame interpolation's eight fragment passes need the
 * second.
 *
 * This shares its body with the compute path (CreatePipelineInternal, defined further
 * down) because the two differ only in which shader blobs are fetched and in what is built
 * at the end -- and for graphics, what is built at the end is nothing, until a job runs and
 * the render-target formats are known. See GraphicsPipelineDX12 in ffx_dx12.h.
 */
static FfxErrorCode CreatePipelineInternal(
    FfxInterface*                 backendInterface,
    FfxEffect                     effect,
    FfxPass                       pass,
    uint32_t                      permutationOptions,
    const FfxPipelineDescription* pipelineDescription,
    FfxUInt32                     effectContextId,
    FfxPipelineState*             outPipeline,
    bool                          graphics);

static FfxErrorCode CreateGraphicsPipelineDX12(
    FfxInterface*                 backendInterface,
    FfxEffect                     effect,
    FfxPass                       pass,
    uint32_t                      permutationOptions,
    const FfxPipelineDescription* pipelineDescription,
    FfxUInt32                     effectContextId,
    FfxPipelineState*             outPipeline)
{
    return CreatePipelineInternal(backendInterface, effect, pass, permutationOptions, pipelineDescription,
                                  effectContextId, outPipeline, /*graphics=*/true);
}

/*
 * The Arm fork added fpSetMessageCallback to FfxInterface and wired it in ffx_vk.cpp. The
 * D3D12 backend in this tree came from AMD 1.1.3, which predates that member, so it was
 * never assigned -- and the frame interpolation component calls it unconditionally during
 * context creation (ffx_frameinterpolation.cpp, in frameinterpolationCreate). An unassigned
 * function pointer is not a soft failure: it is an access violation, and it took the
 * execution harness down before a single pipeline was created.
 *
 * The member count made this easy to miss: 28 of the interface's 30 entries were assigned,
 * and the two that were not were invisible without enumerating them.
 */
FfxErrorCode SetupMessageCallbackDX12(FfxInterface* backendInterface, FfxBackendMessage ffxMessageCallback)
{
    FFX_ASSERT(NULL != backendInterface);

    backendInterface->fpMessage = ffxMessageCallback;

    return FFX_OK;
}

// populate interface with DX12 pointers.
FfxErrorCode ffxGetInterfaceDX12(
    FfxInterface* backendInterface,
    FfxDevice device,
    void* scratchBuffer,
    size_t scratchBufferSize,
    size_t maxContexts) {


    FFX_RETURN_ON_ERROR(
        backendInterface,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        scratchBuffer,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        scratchBufferSize >= ffxGetScratchMemorySizeDX12(maxContexts),
        FFX_ERROR_INSUFFICIENT_MEMORY);
    backendInterface->fpSetMessageCallback = SetupMessageCallbackDX12;

    backendInterface->fpGetSDKVersion = GetSDKVersionDX12;
    backendInterface->fpGetEffectGpuMemoryUsage = GetEffectGpuMemoryUsageDX12;
    backendInterface->fpCreateBackendContext = CreateBackendContextDX12;
    backendInterface->fpGetDeviceCapabilities = GetDeviceCapabilitiesDX12;
    backendInterface->fpDestroyBackendContext = DestroyBackendContextDX12;
    backendInterface->fpCreateResource = CreateResourceDX12;
    backendInterface->fpDestroyResource = DestroyResourceDX12;
    backendInterface->fpMapResource = MapResourceDX12;
    backendInterface->fpUnmapResource = UnmapResourceDX12;
    backendInterface->fpGetResource = GetResourceDX12;
    backendInterface->fpRegisterResource = RegisterResourceDX12;
    backendInterface->fpUnregisterResources = UnregisterResourcesDX12;
    backendInterface->fpRegisterStaticResource      = RegisterStaticResourceDX12;
    backendInterface->fpGetResourceDescription = GetResourceDescriptorDX12;
    backendInterface->fpStageConstantBufferDataFunc = StageConstantBufferDataDX12;
    /*
     * The Arm fork split the single 1.1.3 `fpCreatePipeline` entry point into
     * fpCreateComputePipeline / fpCreateGraphicsPipeline / fpCreateDataGraphPipeline /
     * fpCreateOpticalFlowPipeline. CreatePipelineDX12 implements the compute half; the
     * 1.1.3 function it came from was compute-only anyway (it always passed
     * FFX_BIND_COMPUTE_SHADER_STAGE to the blob provider and only ever called
     * CreateComputePipelineState). Graphics gets an explicit refusal rather than a
     * compute PSO, and the data-graph / optical-flow entry points are left alone: on
     * D3D12 the inference graph runs through the portable dp4a backend, wired in
     * separately, and optical flow has no D3D12 implementation in this tree.
     */
    backendInterface->fpCreateComputePipeline  = CreatePipelineDX12;
    backendInterface->fpCreateGraphicsPipeline = CreateGraphicsPipelineDX12;
    backendInterface->fpCreateDataGraphPipeline = CreateDataGraphPipelineDX12;
    backendInterface->fpGetPermutationBlobByIndex = ffxGetPermutationBlobByIndex;
    backendInterface->fpDestroyPipeline = DestroyPipelineDX12;
    backendInterface->fpScheduleGpuJob = ScheduleGpuJobDX12;
    backendInterface->fpExecuteGpuJobs = ExecuteGpuJobsDX12;
    backendInterface->fpBreadcrumbsAllocBlock = BreadcrumbsAllocBlockDX12;
    backendInterface->fpBreadcrumbsFreeBlock = BreadcrumbsFreeBlockDX12;
    backendInterface->fpBreadcrumbsWrite = BreadcrumbsWriteDX12;
    backendInterface->fpBreadcrumbsPrintDeviceInfo = BreadcrumbsPrintDeviceInfoDX12;
#if defined(FFX_FI) || defined(FFX_ALL)
    backendInterface->fpSwapChainConfigureFrameGeneration = ffxSetFrameGenerationConfigToSwapchainDX12;
#else
    backendInterface->fpSwapChainConfigureFrameGeneration = 0;
#endif // defined(FFX_FI) || defined(FFX_ALL)
    backendInterface->fpRegisterConstantBufferAllocator = RegisterConstantBufferAllocatorDX12;

    // Memory assignments
    backendInterface->scratchBuffer = scratchBuffer;
    backendInterface->scratchBufferSize = scratchBufferSize;

    // The data-graph pipeline record cursor. Reset here rather than in
    // CreateBackendContextDX12 because this runs once, before any context exists, and the
    // host is not required to hand us zeroed scratch memory.
    {
        BackendContext_DX12* pBackendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
        pBackendContext->nextDataGraphPipeline = 0;
    }

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    FFX_RETURN_ON_ERROR(
        !backendContext->refCount,
        FFX_ERROR_BACKEND_API_ERROR);

    // Clear everything out
    memset(backendContext, 0, sizeof(*backendContext));

    // Set the device
    backendInterface->device = device;


    // Assign the max number of contexts we'll be using
    backendContext->maxEffectContexts = (uint32_t)maxContexts;

    return FFX_OK;
}

FfxCommandList ffxGetCommandListDX12(ID3D12CommandList* cmdList)
{
    FFX_ASSERT(NULL != cmdList);
    return reinterpret_cast<FfxCommandList>(cmdList);
}

FfxPipeline ffxGetPipelineDX12(ID3D12PipelineState* pipelineState)
{
    FFX_ASSERT(NULL != pipelineState);
    return reinterpret_cast<FfxPipeline>(pipelineState);
}

// register a DX12 resource to the backend
FfxResource ffxGetResourceDX12(const ID3D12Resource* dx12Resource,
    FfxResourceDescription                     ffxResDescription,
    const wchar_t* ffxResName,
    FfxResourceStates                          state /*=FFX_RESOURCE_STATE_COMPUTE_READ*/)
{
    FfxResource resource = {};
    resource.resource    = reinterpret_cast<void*>(const_cast<ID3D12Resource*>(dx12Resource));
    resource.state = state;
    resource.description = ffxResDescription;

#ifdef _DEBUG
    if (ffxResName) {
        strncpy(resource.name, ffxResName, FFX_RESOURCE_NAME_SIZE - 1);
        resource.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
#else
    (void)ffxResName;
#endif

    return resource;
}

FfxErrorCode ffxLoadPixDll(const wchar_t* pixDllPath)
{
#if ENABLE_PIX_CAPTURES
    // Only do this once
    if (s_PIXDLLLoaded)
        return FFX_OK;

    HMODULE module = LoadLibrary(pixDllPath);

    if (!module)
    {
        return FFX_ERROR_INVALID_PATH;
    }

    // Get handles to PIXBeginEvent and PIXEndEvent
    pixBeginEventOnCommandList = (BeginEventOnCommandList)GetProcAddress(module, "PIXBeginEventOnCommandList");
    pixEndEventOnCommandList   = (EndEventOnCommandList)GetProcAddress(module, "PIXEndEventOnCommandList");
    if (!pixBeginEventOnCommandList || !pixEndEventOnCommandList)
    {
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    s_PIXDLLLoaded = true;
#endif // #if ENABLE_PIX_CAPTURES

    return FFX_OK;
}

void TIF(HRESULT result)
{
    if (FAILED(result)) {

        wchar_t errorMessage[256];
        memset(errorMessage, 0, 256);
        FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, NULL, result, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), errorMessage, 255, NULL);
        char errA[256];
        size_t returnSize;
        wcstombs_s(&returnSize, errA, 255, errorMessage, 255);
#ifdef _DEBUG
        std::ignore = MessageBoxW(NULL, errorMessage, L"Error", MB_OK);
#endif
        throw 1;
    }
}

FfxConstantAllocation BackendContext_DX12::FallbackConstantAllocator(void* data, FfxUInt64 dataSize)
{
    FfxConstantAllocation allocation;
    std::lock_guard<std::mutex> cbLock{ constantBufferMutex };

    if (!constantBufferMem)
    {
        // create dynamic ring buffer for constant uploads
        constantBufferSize = FFX_ALIGN_UP(FFX_BUFFER_SIZE, 256) * maxEffectContexts * FFX_MAX_PASS_COUNT * FFX_MAX_QUEUED_FRAMES; // Size aligned to 256

        CD3DX12_RESOURCE_DESC constDesc = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);
        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        TIF(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &constDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&constantBufferResource)));
        constantBufferResource->SetName(L"FFX_DX12_DynamicRingBuffer");

        // map it
        TIF(constantBufferResource->Map(0, nullptr,
            (void**)&constantBufferMem));
        constantBufferOffset = 0;
    }

    FFX_ASSERT(constantBufferMem);

    uint32_t size = FFX_ALIGN_UP(dataSize, 256);

    // wrap as needed 
    if (constantBufferOffset + size >= constantBufferSize)
        constantBufferOffset = 0;

    void* pBuffer = (void*)((uint8_t*)(constantBufferMem)+constantBufferOffset);
    memcpy(pBuffer, data, (size_t)dataSize);

    D3D12_GPU_VIRTUAL_ADDRESS bufferViewDesc = constantBufferResource->GetGPUVirtualAddress() + constantBufferOffset;

    // update the offset
    constantBufferOffset += size;

    allocation.resource = FfxResource(); // Not needed for directx
    allocation.handle = FfxUInt64(bufferViewDesc);

    return allocation;
}

// fix up format in case resource passed for UAV cannot be mapped
static DXGI_FORMAT convertFormatUav(DXGI_FORMAT format)
{
    switch (format) 
    {
        // Handle Depth
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case DXGI_FORMAT_D32_FLOAT:
            return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_D16_UNORM:
            return DXGI_FORMAT_R16_UNORM;

        // Handle color: assume FLOAT for 16 and 32 bit channels, else UNORM
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R32G32B32_TYPELESS:
            return DXGI_FORMAT_R32G32B32_FLOAT;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R32G32_TYPELESS:
            return DXGI_FORMAT_R32G32_FLOAT;
        case DXGI_FORMAT_R16G16_TYPELESS:
            return DXGI_FORMAT_R16G16_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8X8_TYPELESS:
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        case DXGI_FORMAT_R32_TYPELESS:
            return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R8G8_TYPELESS:
            return DXGI_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_R16_TYPELESS:
            return DXGI_FORMAT_R16_FLOAT;
        case DXGI_FORMAT_R8_TYPELESS:
            return DXGI_FORMAT_R8_UNORM;
        default:
            return format;
    }
}

// fix up format in case resource passed for SRV cannot be mapped
static DXGI_FORMAT convertFormatSrv(DXGI_FORMAT format)
{
    switch (format) 
    {
        // Handle Depth
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case DXGI_FORMAT_D32_FLOAT:
            return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_D16_UNORM:
            return DXGI_FORMAT_R16_UNORM;

        // Handle color: assume FLOAT for 16 and 32 bit channels, else UNORM
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R32G32B32_TYPELESS:
            return DXGI_FORMAT_R32G32B32_FLOAT;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R32G32_TYPELESS:
            return DXGI_FORMAT_R32G32_FLOAT;
        case DXGI_FORMAT_R16G16_TYPELESS:
            return DXGI_FORMAT_R16G16_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8X8_TYPELESS:
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        case DXGI_FORMAT_R32_TYPELESS:
            return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R8G8_TYPELESS:
            return DXGI_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_R16_TYPELESS:
            return DXGI_FORMAT_R16_FLOAT;
        case DXGI_FORMAT_R8_TYPELESS:
            return DXGI_FORMAT_R8_UNORM;
        default:
            return format;
    }
}

D3D12_RESOURCE_STATES ffxGetDX12StateFromResourceState(FfxResourceStates state)
{
    switch (state) {

        case FFX_RESOURCE_STATE_GENERIC_READ:
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        case FFX_RESOURCE_STATE_UNORDERED_ACCESS:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
            return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case FFX_RESOURCE_STATE_COMPUTE_READ:
            return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case FFX_RESOURCE_STATE_PIXEL_READ:
            return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case FFX_RESOURCE_STATE_COPY_SRC:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case FFX_RESOURCE_STATE_COPY_DEST:
            return D3D12_RESOURCE_STATE_COPY_DEST;
        case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
            return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        case FFX_RESOURCE_STATE_PRESENT:
            return D3D12_RESOURCE_STATE_PRESENT;
        case FFX_RESOURCE_STATE_COMMON:
            return D3D12_RESOURCE_STATE_COMMON;
        case FFX_RESOURCE_STATE_RENDER_TARGET:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        default:
            FFX_ASSERT_MESSAGE(false, "Resource state not yet supported");
            return D3D12_RESOURCE_STATE_COMMON;
    }
}

DXGI_FORMAT ffxGetDX12FormatFromSurfaceFormat(FfxSurfaceFormat surfaceFormat)
{
    switch (surfaceFormat) 
    {
        case (FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS):
            return DXGI_FORMAT_R32G32B32A32_TYPELESS;
        case (FFX_SURFACE_FORMAT_R32G32B32A32_UINT):
            return DXGI_FORMAT_R32G32B32A32_UINT;
        case (FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT):
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case (FFX_SURFACE_FORMAT_R16G16B16A16_TYPELESS):
            return DXGI_FORMAT_R16G16B16A16_TYPELESS;
        case (FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT):
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case (FFX_SURFACE_FORMAT_R32G32B32_FLOAT):
            return DXGI_FORMAT_R32G32B32_FLOAT;
        case (FFX_SURFACE_FORMAT_R32G32_TYPELESS):
            return DXGI_FORMAT_R32G32_TYPELESS;
        case (FFX_SURFACE_FORMAT_R32G32_FLOAT):
            return DXGI_FORMAT_R32G32_FLOAT;
        case (FFX_SURFACE_FORMAT_R32_UINT):
            return DXGI_FORMAT_R32_UINT;
        case(FFX_SURFACE_FORMAT_R10G10B10A2_TYPELESS):
            return DXGI_FORMAT_R10G10B10A2_TYPELESS;
        case(FFX_SURFACE_FORMAT_R10G10B10A2_UNORM):
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        case (FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS):
            return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case (FFX_SURFACE_FORMAT_R8G8B8A8_UNORM):
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case (FFX_SURFACE_FORMAT_R8G8B8A8_SRGB):
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case (FFX_SURFACE_FORMAT_R8G8B8A8_SNORM):
            return DXGI_FORMAT_R8G8B8A8_SNORM;
        case (FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS):
            return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        case (FFX_SURFACE_FORMAT_B8G8R8A8_UNORM):
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case (FFX_SURFACE_FORMAT_B8G8R8A8_SRGB):
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case (FFX_SURFACE_FORMAT_R11G11B10_FLOAT):
            return DXGI_FORMAT_R11G11B10_FLOAT;
        case (FFX_SURFACE_FORMAT_R16G16_TYPELESS):
            return DXGI_FORMAT_R16G16_TYPELESS;
        case (FFX_SURFACE_FORMAT_R16G16_FLOAT):
            return DXGI_FORMAT_R16G16_FLOAT;
        case (FFX_SURFACE_FORMAT_R16G16_UINT):
            return DXGI_FORMAT_R16G16_UINT;
        case (FFX_SURFACE_FORMAT_R16G16_SINT):
            return DXGI_FORMAT_R16G16_SINT;
        case (FFX_SURFACE_FORMAT_R16_TYPELESS):
            return DXGI_FORMAT_R16_TYPELESS;
        case (FFX_SURFACE_FORMAT_R16_FLOAT):
            return DXGI_FORMAT_R16_FLOAT;
        case (FFX_SURFACE_FORMAT_R16_UINT):
            return DXGI_FORMAT_R16_UINT;
        case (FFX_SURFACE_FORMAT_R16_UNORM):
            return DXGI_FORMAT_R16_UNORM;
        case (FFX_SURFACE_FORMAT_R16_SNORM):
            return DXGI_FORMAT_R16_SNORM;
        case (FFX_SURFACE_FORMAT_R8_TYPELESS):
            return DXGI_FORMAT_R8_TYPELESS;
        case (FFX_SURFACE_FORMAT_R8_UNORM):
            return DXGI_FORMAT_R8_UNORM;
        case (FFX_SURFACE_FORMAT_R8_UINT):
            return DXGI_FORMAT_R8_UINT;
        case (FFX_SURFACE_FORMAT_R8G8_UINT):
            return DXGI_FORMAT_R8G8_UINT;
        case (FFX_SURFACE_FORMAT_R8G8_TYPELESS):
            return DXGI_FORMAT_R8G8_TYPELESS;
        case (FFX_SURFACE_FORMAT_R8G8_UNORM):
            return DXGI_FORMAT_R8G8_UNORM;
        case (FFX_SURFACE_FORMAT_R32_TYPELESS):
            return DXGI_FORMAT_R32_TYPELESS;
        case (FFX_SURFACE_FORMAT_R32_FLOAT):
            return DXGI_FORMAT_R32_FLOAT;
        case (FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP):
            return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        case (FFX_SURFACE_FORMAT_UNKNOWN):
            return DXGI_FORMAT_UNKNOWN;

        default:
            FFX_ASSERT_MESSAGE(false, "Format not yet supported");
            return DXGI_FORMAT_UNKNOWN;
    }
}

D3D12_RESOURCE_FLAGS ffxGetDX12ResourceFlags(FfxResourceUsage flags)
{
    D3D12_RESOURCE_FLAGS dx12ResourceFlags = D3D12_RESOURCE_FLAG_NONE;
    if (flags & FFX_RESOURCE_USAGE_RENDERTARGET) dx12ResourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (flags & FFX_RESOURCE_USAGE_UAV) dx12ResourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return dx12ResourceFlags;
}

FfxSurfaceFormat ffxGetSurfaceFormatDX12(DXGI_FORMAT format)
{
    switch (format) {

        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
            return FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R32G32B32_FLOAT:
            return FFX_SURFACE_FORMAT_R32G32B32_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_UINT:
            return FFX_SURFACE_FORMAT_R32G32B32A32_UINT;

        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;

        case DXGI_FORMAT_R32G32_TYPELESS:
        case DXGI_FORMAT_R32G32_FLOAT:
            return FFX_SURFACE_FORMAT_R32G32_FLOAT;

        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
            return FFX_SURFACE_FORMAT_R32_FLOAT;

        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
            return FFX_SURFACE_FORMAT_R32_UINT;

        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
            return FFX_SURFACE_FORMAT_R8_UINT;

        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return FFX_SURFACE_FORMAT_R10G10B10A2_UNORM;
        
        case DXGI_FORMAT_R11G11B10_FLOAT:
            return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;

        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            return FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return FFX_SURFACE_FORMAT_R8G8B8A8_SRGB;
        case DXGI_FORMAT_R8G8B8A8_SNORM:
            return FFX_SURFACE_FORMAT_R8G8B8A8_SNORM;

        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            return FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return FFX_SURFACE_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return FFX_SURFACE_FORMAT_B8G8R8A8_SRGB;

        case DXGI_FORMAT_R16G16_TYPELESS:
        case DXGI_FORMAT_R16G16_FLOAT:
            return FFX_SURFACE_FORMAT_R16G16_FLOAT;
        case DXGI_FORMAT_R16G16_UINT:
            return FFX_SURFACE_FORMAT_R16G16_UINT;
        case DXGI_FORMAT_R16G16_SINT:
            return FFX_SURFACE_FORMAT_R16G16_SINT;
        case DXGI_FORMAT_R32_UINT:
            return FFX_SURFACE_FORMAT_R32_UINT;
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R32_FLOAT:
            return FFX_SURFACE_FORMAT_R32_FLOAT;

        case DXGI_FORMAT_R8G8_TYPELESS:
        case DXGI_FORMAT_R8G8_UINT:
            return FFX_SURFACE_FORMAT_R8G8_UINT;
        case DXGI_FORMAT_R8G8_UNORM:
            return FFX_SURFACE_FORMAT_R8G8_UNORM;

        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R16_FLOAT:
            return FFX_SURFACE_FORMAT_R16_FLOAT;
        case DXGI_FORMAT_R16_UINT:
            return FFX_SURFACE_FORMAT_R16_UINT;
        case DXGI_FORMAT_D16_UNORM:
        case DXGI_FORMAT_R16_UNORM:
            return FFX_SURFACE_FORMAT_R16_UNORM;
        case DXGI_FORMAT_R16_SNORM:
            return FFX_SURFACE_FORMAT_R16_SNORM;

        case DXGI_FORMAT_R8_TYPELESS:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_A8_UNORM:
            return FFX_SURFACE_FORMAT_R8_UNORM;
        case DXGI_FORMAT_R8_UINT:
            return FFX_SURFACE_FORMAT_R8_UINT;

        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
            return FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP;

        case DXGI_FORMAT_UNKNOWN:
            return FFX_SURFACE_FORMAT_UNKNOWN;
        default:
            FFX_ASSERT_MESSAGE(false, "Format not yet supported");
            return FFX_SURFACE_FORMAT_UNKNOWN;
    }
}

bool IsDepthDX12(DXGI_FORMAT format)
{
    return (format == DXGI_FORMAT_D16_UNORM) || 
           (format == DXGI_FORMAT_D32_FLOAT) || 
           (format == DXGI_FORMAT_D24_UNORM_S8_UINT) ||
           (format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
}

bool IsStencilDX12(DXGI_FORMAT format)
{
    return (format == DXGI_FORMAT_D24_UNORM_S8_UINT) || (format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
}

FfxResourceDescription ffxGetResourceDescriptionDX12(const ID3D12Resource* pResource, FfxResourceUsage additionalUsages /*=FFX_RESOURCE_USAGE_READ_ONLY*/)
{
    FfxResourceDescription resourceDescription = {};

    // This is valid
    if (!pResource)
        return resourceDescription;

    if (pResource)
    {
        D3D12_RESOURCE_DESC desc = const_cast<ID3D12Resource*>(pResource)->GetDesc();
        
        if( desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        {
            resourceDescription.flags  = FFX_RESOURCE_FLAGS_NONE;
            resourceDescription.usage  = FFX_RESOURCE_USAGE_UAV;
            resourceDescription.size  = (uint32_t)desc.Width;
            resourceDescription.stride = (uint32_t)desc.Height;
            resourceDescription.format = ffxGetSurfaceFormatDX12(desc.Format);

            // What should we initialize this to?? No case for this yet
            resourceDescription.depth    = 0;
            resourceDescription.mipCount = 0;

            // Set the type
            resourceDescription.type = FFX_RESOURCE_TYPE_BUFFER;
        }
        else
        {
            // Set flags properly for resource registration
            resourceDescription.flags     = FFX_RESOURCE_FLAGS_NONE;
           
            // Check for depth use
            resourceDescription.usage     = IsDepthDX12(desc.Format) ? FFX_RESOURCE_USAGE_DEPTHTARGET : FFX_RESOURCE_USAGE_READ_ONLY;
            
            if (IsStencilDX12(desc.Format))
                resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_STENCILTARGET);

            // Unordered access use
            if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
                resourceDescription.usage = (FfxResourceUsage)(resourceDescription.usage | FFX_RESOURCE_USAGE_UAV);

            // Resource-specific supplemental use flags
            resourceDescription.usage    = (FfxResourceUsage)(resourceDescription.usage | additionalUsages);

            resourceDescription.width    = (uint32_t)desc.Width;
            resourceDescription.height   = (uint32_t)desc.Height;
            resourceDescription.depth    = desc.DepthOrArraySize;
            resourceDescription.mipCount = desc.MipLevels;
            resourceDescription.format   = ffxGetSurfaceFormatDX12(desc.Format);

            switch (desc.Dimension)
            {
            case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
                resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE1D;
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
                if (FFX_CONTAINS_FLAG(additionalUsages, FFX_RESOURCE_USAGE_ARRAYVIEW))
                    resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
                else if (desc.DepthOrArraySize == 1)
                    resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
                else if (desc.DepthOrArraySize == 6)
                    resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE_CUBE;
                else
                    resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE3D;
                break;
            default:
                FFX_ASSERT_MESSAGE(false, "FFXInterface: DX12: Unsupported texture dimension requested. Please implement.");
                break;
            }
        }
    }

    return resourceDescription;
}


ID3D12Resource* getDX12ResourcePtr(BackendContext_DX12* backendContext, int32_t resourceIndex)
{
    FFX_ASSERT(NULL != backendContext);
    return reinterpret_cast<ID3D12Resource*>(backendContext->pResources[resourceIndex].resourcePtr);
}

void beginMarkerDX12(BackendContext_DX12* backendContext, ID3D12GraphicsCommandList* pCmdList, const char* label)
{
    FFX_ASSERT(nullptr != backendContext);
    FFX_ASSERT(nullptr != pCmdList);

#if ENABLE_PIX_CAPTURES
    if (s_PIXDLLLoaded)
    {
        char strLabel[FFX_RESOURCE_NAME_SIZE];
        WideCharToMultiByte(CP_UTF8, 0, label, -1, strLabel, int(std::size(strLabel)), nullptr, nullptr);
        pixBeginEventOnCommandList(pCmdList, 0, strLabel);
    }
#endif // #if ENABLE_PIX_CAPTURES
}

void endMarkerDX12(BackendContext_DX12* backendContext, ID3D12GraphicsCommandList* pCmdList)
{
    FFX_ASSERT(nullptr != backendContext);
    FFX_ASSERT(nullptr != pCmdList);

#if ENABLE_PIX_CAPTURES
    if (s_PIXDLLLoaded)
    {
        pixEndEventOnCommandList(pCmdList);
    }
#endif // #if ENABLE_PIX_CAPTURES
}

void addBarrier(BackendContext_DX12* backendContext, FfxResourceInternal* resource, FfxResourceStates newState)
{
    FFX_ASSERT(NULL != backendContext);
    FFX_ASSERT(NULL != resource);

    ID3D12Resource* dx12Resource = getDX12ResourcePtr(backendContext, resource->internalIndex);
    D3D12_RESOURCE_BARRIER* barrier = &backendContext->barriers[backendContext->barrierCount];

    FFX_ASSERT(backendContext->barrierCount < FFX_MAX_BARRIERS);

    FfxResourceStates* currentState = &backendContext->pResources[resource->internalIndex].currentState;

    if ((*currentState & newState) != newState) {

        *barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            dx12Resource,
            ffxGetDX12StateFromResourceState(*currentState),
            ffxGetDX12StateFromResourceState(newState));

        *currentState = newState;
        ++backendContext->barrierCount;

    }
    else if (newState == FFX_RESOURCE_STATE_UNORDERED_ACCESS) {

        *barrier = CD3DX12_RESOURCE_BARRIER::UAV(dx12Resource);
        ++backendContext->barrierCount;
    }
}

void flushBarriers(BackendContext_DX12* backendContext, ID3D12GraphicsCommandList* dx12CommandList)
{
    FFX_ASSERT(NULL != backendContext);
    FFX_ASSERT(NULL != dx12CommandList);

    if (backendContext->barrierCount > 0) {

        dx12CommandList->ResourceBarrier(backendContext->barrierCount, backendContext->barriers);
        backendContext->barrierCount = 0;
    }
}

//////////////////////////////////////////////////////////////////////////
// DX12 back end implementation

FfxUInt32 GetSDKVersionDX12(FfxInterface*)
{
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
}

uint64_t GetResourceGpuMemorySizeDX12(ID3D12Resource* resource)
{
    uint64_t      size = 0;
    D3D12_RESOURCE_ALLOCATION_INFO allocInfo = {};
    if (resource)
    {
        D3D12_RESOURCE_DESC desc = resource->GetDesc();
        ID3D12Device4* pDevice4 = nullptr;
        if (SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(&pDevice4))))
        {
            allocInfo = pDevice4->GetResourceAllocationInfo(0, 1, &desc);
            size = allocInfo.SizeInBytes;
            pDevice4->Release();
        }
    }

    return size;
}

FfxErrorCode GetEffectGpuMemoryUsageDX12(FfxInterface* backendInterface, FfxUInt32 effectContextId, FfxEffectMemoryUsage* outVramUsage)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != outVramUsage);

    BackendContext_DX12*                backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
    BackendContext_DX12::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];

    *outVramUsage = effectContext.vramUsage;

    return FFX_OK;
}

// initialize the DX12 backend
FfxErrorCode CreateBackendContextDX12(FfxInterface* backendInterface, FfxEffect effect, FfxEffectBindlessConfig* bindlessConfig, FfxUInt32* effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != backendInterface->device);

    HRESULT result = S_OK;
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendInterface->device);

    // set up some internal resources we need (space for resource views and constant buffers)
    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    // Set things up if this is the first invocation
    if (!backendContext->refCount) {

        new (&backendContext->constantBufferMutex) std::mutex();

        if (dx12Device != NULL) {

            dx12Device->AddRef();
            backendContext->device = dx12Device;
        }

        // Map all of our pointers
        uint32_t gpuJobDescArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_GPU_JOBS * sizeof(FfxGpuJobDescription), sizeof(uint32_t));
        uint32_t resourceArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_RESOURCE_COUNT * sizeof(BackendContext_DX12::Resource), sizeof(uint64_t));
        uint32_t stagingRingBufferArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE, sizeof(uint32_t));
        uint32_t contextArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * sizeof(BackendContext_DX12::EffectContext), sizeof(uint32_t));

        uint8_t* pMem = (uint8_t*)((BackendContext_DX12*)(backendContext + 1));

        // Map gpu job array
        backendContext->pGpuJobs = (FfxGpuJobDescription*)pMem;
        memset(backendContext->pGpuJobs, 0, gpuJobDescArraySize);
        pMem += gpuJobDescArraySize;

        // Map the resources
        backendContext->pResources = (BackendContext_DX12::Resource*)(pMem);
        memset(backendContext->pResources, 0, resourceArraySize);
        pMem += resourceArraySize;

        // Map the staging buffer
        backendContext->pStagingRingBuffer = (uint8_t*)(pMem);
        memset(backendContext->pStagingRingBuffer, 0, stagingRingBufferArraySize);
        pMem += stagingRingBufferArraySize;

        // Map the effect contexts
        backendContext->pEffectContexts = reinterpret_cast<BackendContext_DX12::EffectContext*>(pMem);
        memset(backendContext->pEffectContexts, 0, contextArraySize);
        pMem += contextArraySize;

        /*
         * Map the pipeline records.
         *
         * pDataGraphPipelines was previously left null while CreateDataGraphPipelineDX12
         * wrote through it. ffxGetScratchMemorySizeDX12 now reserves both arrays, and they
         * are mapped here. The total there is a sum, so order does not affect correctness --
         * but keeping the two lists in step is what keeps this readable.
         */
        uint32_t dataGraphPipelineArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_DX12::DataGraphPipelineDX12), sizeof(uint64_t));
        backendContext->pDataGraphPipelines = reinterpret_cast<BackendContext_DX12::DataGraphPipelineDX12*>(pMem);
        memset(backendContext->pDataGraphPipelines, 0, dataGraphPipelineArraySize);
        pMem += dataGraphPipelineArraySize;

        uint32_t graphicsPipelineArraySize = FFX_ALIGN_UP(backendContext->maxEffectContexts * FFX_MAX_PASS_COUNT * sizeof(BackendContext_DX12::GraphicsPipelineDX12), sizeof(uint64_t));
        backendContext->pGraphicsPipelines = reinterpret_cast<BackendContext_DX12::GraphicsPipelineDX12*>(pMem);
        memset(backendContext->pGraphicsPipelines, 0, graphicsPipelineArraySize);
        pMem += graphicsPipelineArraySize;

        // CPUVisible
        D3D12_DESCRIPTOR_HEAP_DESC descHeap;
        descHeap.NumDescriptors = FFX_MAX_RESOURCE_COUNT * backendContext->maxEffectContexts;
        descHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        descHeap.NodeMask = 0;

        result = dx12Device->CreateDescriptorHeap(&descHeap, IID_PPV_ARGS(&backendContext->descHeapSrvCpu));
        result = dx12Device->CreateDescriptorHeap(&descHeap, IID_PPV_ARGS(&backendContext->descHeapUavCpu));

        // GPU
        descHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        result = dx12Device->CreateDescriptorHeap(&descHeap, IID_PPV_ARGS(&backendContext->descHeapUavGpu));

        // descriptor ring buffer
        descHeap.NumDescriptors            = FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts + FFX_MAX_STATIC_DESCRIPTOR_COUNT;
        backendContext->descRingBufferSize = descHeap.NumDescriptors;
        backendContext->descRingBufferBase = 0;
        result = dx12Device->CreateDescriptorHeap(&descHeap, IID_PPV_ARGS(&backendContext->descRingBuffer));

        // RTV descriptor heap to raster jobs
        descHeap.NumDescriptors = 8;
        descHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        descHeap.NodeMask = 0;
        descHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        dx12Device->CreateDescriptorHeap(&descHeap, IID_PPV_ARGS(&backendContext->descHeapRtvCpu));

        // initialize the bindless offset to *after* the ring-buffer 
        backendContext->descBindlessBase = FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts;

        // DXGI factory used for memory usage tracking
        result = CreateDXGIFactory2(0, IID_PPV_ARGS(&backendContext->dxgiFactory));

        FFX_UNUSED(result);
    }

    // Increment the ref count
    ++backendContext->refCount;

    // Get an available context id
    for (uint32_t i = 0; i < backendContext->maxEffectContexts; ++i) {
        if (!backendContext->pEffectContexts[i].active) {
            *effectContextId = i;

            // Reset everything accordingly
            BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[i];
            effectContext.active = true;
            effectContext.effectId = effect;

            effectContext.nextStaticResource = (i * FFX_MAX_RESOURCE_COUNT) + 1;
            effectContext.nextDynamicResource = (i * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;
            effectContext.nextStaticUavDescriptor = (i * FFX_MAX_RESOURCE_COUNT);
            effectContext.nextDynamicUavDescriptor = (i * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;

            if (bindlessConfig)
            {
                uint32_t numDescriptors = bindlessConfig->maxTextureSrvs + bindlessConfig->maxBufferSrvs + bindlessConfig->maxTextureUavs + bindlessConfig->maxBufferUavs;

                uint32_t bindlessBase = getFreeBindlessDescriptorBlock(backendContext, numDescriptors, i);

                effectContext.bindlessBufferHeapStart = bindlessBase;
                effectContext.bindlessBufferHeapEnd = bindlessBase + numDescriptors;

                effectContext.bindlessTextureSrvHeapStart = bindlessBase;
                effectContext.bindlessTextureSrvHeapSize  = bindlessConfig->maxTextureSrvs;

                bindlessBase += bindlessConfig->maxTextureSrvs;

                effectContext.bindlessBufferSrvHeapStart = bindlessBase;
                effectContext.bindlessBufferSrvHeapSize  = bindlessConfig->maxBufferSrvs;

                bindlessBase += bindlessConfig->maxBufferSrvs;

                effectContext.bindlessTextureUavHeapStart = bindlessBase;
                effectContext.bindlessTextureUavHeapSize  = bindlessConfig->maxTextureUavs;

                bindlessBase += bindlessConfig->maxTextureUavs;

                effectContext.bindlessBufferUavHeapStart = bindlessBase;
                effectContext.bindlessBufferUavHeapSize  = bindlessConfig->maxBufferUavs;

                bindlessBase += bindlessConfig->maxBufferUavs;
            }
            else
            {
                effectContext.bindlessTextureSrvHeapStart = 0;
                effectContext.bindlessTextureSrvHeapSize  = 0;
                effectContext.bindlessBufferSrvHeapStart  = 0;
                effectContext.bindlessBufferSrvHeapSize   = 0;
                effectContext.bindlessTextureUavHeapStart = 0;
                effectContext.bindlessTextureUavHeapSize  = 0;
                effectContext.bindlessBufferUavHeapStart  = 0;
                effectContext.bindlessBufferUavHeapSize   = 0;
                effectContext.bindlessBufferHeapStart     = 0;
                effectContext.bindlessBufferHeapEnd       = 0;
            }   

            break;
        }
    }

    return FFX_OK;
}

// query device capabilities to select the optimal shader permutation
FfxErrorCode GetDeviceCapabilitiesDX12(FfxInterface* backendInterface, FfxDeviceCapabilities* deviceCapabilities)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != backendInterface->device);
    FFX_ASSERT(NULL != deviceCapabilities);
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendInterface->device);

    /*
     * Zero first, then set the fields this backend knows about.
     *
     * FfxDeviceCapabilities carries members the Arm fork added for its tensor and
     * data-graph paths -- tensorSupported, computeSupportTensor, fragmentSupportTensor,
     * dataGraphSupported, dataGraphOFSupported, the grid-size sets and the four
     * dataGraphOF* extents. This function came from AMD 1.1.3 and knows none of them, so
     * without this the caller's struct keeps whatever happened to be on its stack.
     *
     * The frame interpolation component is exactly such a caller: it declares
     * `FfxDeviceCapabilities capabilities;` uninitialised, derives `inTensorBufferAliased`
     * from computeSupportTensor / fragmentSupportTensor, and feeds the resulting
     * FfxResourceFlags into fpCreateResource. Garbage there is how the execution harness
     * died with STATUS_STACK_BUFFER_OVERRUN before creating a single pipeline -- and it is
     * the kind of failure that produces no diagnostic at all.
     *
     * The Vulkan backend sets every one of these. D3D12 cannot honour any of them: there
     * are no ARM tensors and no data graph on this API, so false and zero are the correct
     * values rather than placeholders. The effect then takes its buffer-aliased path, which
     * this backend does implement.
     */
    memset(deviceCapabilities, 0, sizeof(FfxDeviceCapabilities));
    deviceCapabilities->tensorSupported       = false;
    deviceCapabilities->computeSupportTensor  = false;
    deviceCapabilities->fragmentSupportTensor = false;
    deviceCapabilities->dataGraphSupported    = false;
    deviceCapabilities->dataGraphOFSupported  = false;

    // Check if we have shader model 6.6
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_6 };
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(D3D12_FEATURE_DATA_SHADER_MODEL)))) {

        switch (shaderModel.HighestShaderModel) {

        case D3D_SHADER_MODEL_5_1:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_5_1;
            break;

        case D3D_SHADER_MODEL_6_0:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_0;
            break;

        case D3D_SHADER_MODEL_6_1:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_1;
            break;

        case D3D_SHADER_MODEL_6_2:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_2;
            break;

        case D3D_SHADER_MODEL_6_3:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_3;
            break;

        case D3D_SHADER_MODEL_6_4:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_4;
            break;

        case D3D_SHADER_MODEL_6_5:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_5;
            break;

        case D3D_SHADER_MODEL_6_6:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_6;
            break;

        default:
            deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_6;
            break;
        }
    }
    else {

        deviceCapabilities->maximumSupportedShaderModel = FFX_SHADER_MODEL_5_1;
    }

    // check if we can force wave64 mode.
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 d3d12Options1 = {};
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &d3d12Options1, sizeof(d3d12Options1)))) {

        const uint32_t waveLaneCountMin = d3d12Options1.WaveLaneCountMin;
        const uint32_t waveLaneCountMax = d3d12Options1.WaveLaneCountMax;
        deviceCapabilities->waveLaneCountMin = waveLaneCountMin;
        deviceCapabilities->waveLaneCountMax = waveLaneCountMax;
    }

    // check if we have 16bit floating point.
    D3D12_FEATURE_DATA_D3D12_OPTIONS d3d12Options = {};
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3d12Options, sizeof(d3d12Options)))) {

        deviceCapabilities->fp16Supported = bool(d3d12Options.MinPrecisionSupport & D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT);
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 d3d12Options4 = {};
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &d3d12Options4, sizeof(d3d12Options4)))) {

        deviceCapabilities->fp16Supported &= bool(d3d12Options4.Native16BitShaderOpsSupported);
    }

    // check if we have raytracing support
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 d3d12Options5 = {};
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &d3d12Options5, sizeof(d3d12Options5)))) {

        deviceCapabilities->raytracingSupported = (d3d12Options5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED);
    }

    deviceCapabilities->deviceCoherentMemorySupported = false;
    deviceCapabilities->dedicatedAllocationSupported = true; // committed resources are always available
    deviceCapabilities->bufferMarkerSupported = false;
    deviceCapabilities->extendedSynchronizationSupported = false;
    deviceCapabilities->shaderStorageBufferArrayNonUniformIndexing = true;

    return FFX_OK;
}

// deinitialize the DX12 backend
FfxErrorCode DestroyBackendContextDX12(FfxInterface* backendInterface, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
    FFX_ASSERT(backendContext->refCount > 0);

    // Delete any resources allocated by this context
    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];
    for (uint32_t currentStaticResourceIndex = effectContextId * FFX_MAX_RESOURCE_COUNT; currentStaticResourceIndex < (uint32_t)effectContext.nextStaticResource; ++currentStaticResourceIndex) {
        if (backendContext->pResources[currentStaticResourceIndex].resourcePtr) {
            FFX_ASSERT_MESSAGE(false, "FFXInterface: DX12: SDK Resource was not destroyed prior to destroying the backend context. There is a resource leak.");
            FfxResourceInternal internalResource = { (int32_t)currentStaticResourceIndex };
            DestroyResourceDX12(backendInterface, internalResource, effectContextId);
        }
    }

    // Free up for use by another context
    effectContext.nextStaticResource = 0;
    effectContext.active = false;

    // Decrement ref count
    --backendContext->refCount;

    if (!backendContext->refCount) {

        // release constant buffer pool if it was allocated
        if (backendContext->constantBufferMem)
        {
            backendContext->constantBufferResource->Unmap(0, nullptr);
            backendContext->constantBufferResource->Release();
            backendContext->constantBufferMem = nullptr;
            backendContext->constantBufferOffset = 0;
            backendContext->constantBufferSize = 0;
        }

        backendContext->gpuJobCount             = 0;
        backendContext->barrierCount            = 0;

        // release heaps
        backendContext->descHeapRtvCpu->Release();
        backendContext->descHeapSrvCpu->Release();
        backendContext->descHeapUavCpu->Release();
        backendContext->descHeapUavGpu->Release();
        backendContext->descRingBuffer->Release();

        if (backendContext->device != NULL) {
            backendContext->device->Release();
            backendContext->device = NULL;
        }

        if (backendContext->dxgiFactory != NULL) {
            backendContext->dxgiFactory->Release();
            backendContext->dxgiFactory = NULL;
        }
    }

    return FFX_OK;
}

// create a internal resource that will stay alive until effect gets shut down
FfxErrorCode CreateResourceDX12(
    FfxInterface* backendInterface,
    const FfxCreateResourceDescription* createResourceDescription,
    FfxUInt32 effectContextId,
    FfxResourceInternal* outTexture
)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != createResourceDescription);
    FFX_ASSERT(NULL != outTexture);
    FFX_ASSERT_MESSAGE(createResourceDescription->initData.type != FFX_RESOURCE_INIT_DATA_TYPE_INVALID,
                       "InitData type cannot be FFX_RESOURCE_INIT_DATA_TYPE_INVALID. Please explicitly specify the resource initialization type.");


    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];
    ID3D12Device* dx12Device = backendContext->device;

    uint64_t resourceSize = 0;
    FFX_ASSERT(NULL != dx12Device);

    D3D12_HEAP_PROPERTIES dx12HeapProperties = {};

    switch (createResourceDescription->heapType)
    {
    case FFX_HEAP_TYPE_DEFAULT:
        dx12HeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
        break;
    case FFX_HEAP_TYPE_UPLOAD:
        dx12HeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        break;
    case FFX_HEAP_TYPE_READBACK:
        dx12HeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
        break;
    default:
        dx12HeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
        break;
    }

    FFX_ASSERT(effectContext.nextStaticResource + 1 < effectContext.nextDynamicResource);

    outTexture->internalIndex = effectContext.nextStaticResource++;
    BackendContext_DX12::Resource* backendResource = &backendContext->pResources[outTexture->internalIndex];
    backendResource->resourceDescription = createResourceDescription->resourceDescription;

    const auto& initData = createResourceDescription->initData;

    D3D12_RESOURCE_DESC dx12ResourceDescription = {};
    dx12ResourceDescription.Format              = DXGI_FORMAT_UNKNOWN;
    dx12ResourceDescription.Width               = 1;
    dx12ResourceDescription.Height              = 1;
    dx12ResourceDescription.MipLevels           = 1;
    dx12ResourceDescription.DepthOrArraySize    = 1;
    dx12ResourceDescription.SampleDesc.Count    = 1;
    dx12ResourceDescription.Flags               = ffxGetDX12ResourceFlags(backendResource->resourceDescription.usage);

    switch (createResourceDescription->resourceDescription.type) {

    case FFX_RESOURCE_TYPE_BUFFER:
        dx12ResourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        dx12ResourceDescription.Width = createResourceDescription->resourceDescription.width;
        dx12ResourceDescription.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        break;

    case FFX_RESOURCE_TYPE_TENSOR:
        /*
         * 1.1.3 had no tensors, so there is no case for them here; the ARM fork's data
         * graph expressed every tensor as a VK_ARM_tensors object, which does not exist on
         * D3D12 at all.
         *
         * The dp4a backend does not care about the tensor abstraction: it reads and writes
         * int8 payloads, so a tensor is exactly a raw buffer whose width is the byte size
         * the effect declared in shapeSize. No typed view is created here -- the dp4a
         * module makes its own raw SRV/UAV -- which is why this needs nothing beyond the
         * buffer case.
         */
        dx12ResourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        /*
         * The byte size is NOT shapeSize. shapeSize is the RANK of the shape -- the
         * interpolation effect sets tensorShapeSize = 4, meaning four dimensions -- while
         * the extent lives in width and height and the innermost dimension in channel.
         * Using shapeSize as the width produced a four-byte buffer for a tensor that needs
         * 160*96*16 = 245,760 bytes. The first such call happened to survive and the second
         * aborted inside CreateCommittedResource, which is what the execution harness
         * caught: an abort with no diagnostic, because the failing HRESULT is swallowed by
         * the TIF macro.
         */
        dx12ResourceDescription.Width = static_cast<UINT64>(createResourceDescription->resourceDescription.width)
                                      * createResourceDescription->resourceDescription.height
                                      * (createResourceDescription->resourceDescription.channel > 0
                                             ? createResourceDescription->resourceDescription.channel
                                             : 1u)
                                      * sizeof(int8_t);
        dx12ResourceDescription.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        break;

    case FFX_RESOURCE_TYPE_TEXTURE1D:
        dx12ResourceDescription.Format = ffxGetDX12FormatFromSurfaceFormat(createResourceDescription->resourceDescription.format);
        dx12ResourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        dx12ResourceDescription.Width = createResourceDescription->resourceDescription.width;
        dx12ResourceDescription.DepthOrArraySize = UINT16(createResourceDescription->resourceDescription.depth);
        dx12ResourceDescription.MipLevels = UINT16(createResourceDescription->resourceDescription.mipCount);
        break;

    case FFX_RESOURCE_TYPE_TEXTURE2D:
        dx12ResourceDescription.Format = ffxGetDX12FormatFromSurfaceFormat(createResourceDescription->resourceDescription.format);
        dx12ResourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dx12ResourceDescription.Width = createResourceDescription->resourceDescription.width;
        dx12ResourceDescription.Height = createResourceDescription->resourceDescription.height;
        dx12ResourceDescription.DepthOrArraySize = UINT16(createResourceDescription->resourceDescription.depth);
        dx12ResourceDescription.MipLevels = UINT16(createResourceDescription->resourceDescription.mipCount);
        break;

    case FFX_RESOURCE_TYPE_TEXTURE_CUBE:
    case FFX_RESOURCE_TYPE_TEXTURE3D:
        dx12ResourceDescription.Format = ffxGetDX12FormatFromSurfaceFormat(createResourceDescription->resourceDescription.format);
        dx12ResourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        dx12ResourceDescription.Width = createResourceDescription->resourceDescription.width;
        dx12ResourceDescription.Height = createResourceDescription->resourceDescription.height;
        dx12ResourceDescription.DepthOrArraySize = UINT16(createResourceDescription->resourceDescription.depth);
        dx12ResourceDescription.MipLevels = UINT16(createResourceDescription->resourceDescription.mipCount);
        break;

    default:
        break;
    }

    ID3D12Resource* dx12Resource = nullptr;
    if (createResourceDescription->heapType == FFX_HEAP_TYPE_UPLOAD) {

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT dx12Footprint = {};

        UINT rowCount;
        UINT64 rowSizeInBytes;
        UINT64 totalBytes;

        dx12Device->GetCopyableFootprints(&dx12ResourceDescription, 0, 1, 0, &dx12Footprint, &rowCount, &rowSizeInBytes, &totalBytes);

        D3D12_HEAP_PROPERTIES dx12UploadHeapProperties = {};
        dx12UploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC dx12UploadBufferDescription = {};

        dx12UploadBufferDescription.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        dx12UploadBufferDescription.Width = totalBytes;
        dx12UploadBufferDescription.Height = 1;
        dx12UploadBufferDescription.DepthOrArraySize = 1;
        dx12UploadBufferDescription.MipLevels = 1;
        dx12UploadBufferDescription.Format = DXGI_FORMAT_UNKNOWN;
        dx12UploadBufferDescription.SampleDesc.Count = 1;
        dx12UploadBufferDescription.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        TIF(dx12Device->CreateCommittedResource(&dx12HeapProperties, D3D12_HEAP_FLAG_NONE, &dx12UploadBufferDescription, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&dx12Resource)));
        resourceSize = GetResourceGpuMemorySizeDX12(dx12Resource);

        backendResource->initialState = FFX_RESOURCE_STATE_GENERIC_READ;
        backendResource->currentState = FFX_RESOURCE_STATE_GENERIC_READ;

        D3D12_RANGE dx12EmptyRange = {};
        void* uploadBufferData = nullptr;
        TIF(dx12Resource->Map(0, &dx12EmptyRange, &uploadBufferData));

        const uint8_t* src = static_cast<uint8_t*>(initData.buffer);
        uint8_t* dst = static_cast<uint8_t*>(uploadBufferData);
        for (uint32_t currentRowIndex = 0; currentRowIndex < createResourceDescription->resourceDescription.height; ++currentRowIndex) {

            if (initData.type == FFX_RESOURCE_INIT_DATA_TYPE_BUFFER)
            {
                memcpy(dst, src, (size_t)rowSizeInBytes);
                src += rowSizeInBytes;
            }
            else if (initData.type == FFX_RESOURCE_INIT_DATA_TYPE_VALUE)
            {
                memset(dst, initData.value, (size_t)rowSizeInBytes);
            }
            dst += dx12Footprint.Footprint.RowPitch;
        }

        dx12Resource->Unmap(0, nullptr);
        { wchar_t ffxWideName[FFX_RESOURCE_NAME_SIZE];
            ffxNarrowToWide(ffxWideName, std::size(ffxWideName), createResourceDescription->name);
            dx12Resource->SetName(ffxWideName); }
        backendResource->resourcePtr = dx12Resource;

#ifdef _DEBUG
        ffxNarrowToWide(backendResource->resourceName, std::size(backendResource->resourceName), createResourceDescription->name);
#endif
        return FFX_OK;

    }
    else {

        const FfxResourceStates resourceStates =
            ((initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED) && (createResourceDescription->heapType != FFX_HEAP_TYPE_UPLOAD))
                ? FFX_RESOURCE_STATE_COPY_DEST
                : createResourceDescription->initialState;
        // Buffers ignore any input state and create in common (but issue a warning)
        const D3D12_RESOURCE_STATES dx12ResourceStates = dx12ResourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? D3D12_RESOURCE_STATE_COMMON : ffxGetDX12StateFromResourceState(resourceStates);

        TIF(dx12Device->CreateCommittedResource(&dx12HeapProperties, D3D12_HEAP_FLAG_NONE, &dx12ResourceDescription, dx12ResourceStates, nullptr, IID_PPV_ARGS(&dx12Resource)));
        resourceSize = GetResourceGpuMemorySizeDX12(dx12Resource);
        backendResource->initialState = resourceStates;
        backendResource->currentState = resourceStates;

        { wchar_t ffxWideName[FFX_RESOURCE_NAME_SIZE];
            ffxNarrowToWide(ffxWideName, std::size(ffxWideName), createResourceDescription->name);
            dx12Resource->SetName(ffxWideName); }
        backendResource->resourcePtr = dx12Resource;

#ifdef _DEBUG
        ffxNarrowToWide(backendResource->resourceName, std::size(backendResource->resourceName), createResourceDescription->name);
#endif

        // Create SRVs and UAVs
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC dx12UavDescription = {};
            D3D12_SHADER_RESOURCE_VIEW_DESC dx12SrvDescription = {};
            D3D12_RESOURCE_DESC dx12Desc = dx12Resource->GetDesc();
            dx12UavDescription.Format = convertFormatUav(dx12Desc.Format);
            dx12SrvDescription.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            dx12SrvDescription.Format = convertFormatSrv(dx12Desc.Format);

            bool requestArrayView = FFX_CONTAINS_FLAG(createResourceDescription->resourceDescription.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

            switch (dx12Desc.Dimension) {

            case D3D12_RESOURCE_DIMENSION_BUFFER:
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                break;

            case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
                if (dx12Desc.DepthOrArraySize > 1 || requestArrayView)
                {
                    dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                    dx12UavDescription.Texture1DArray.ArraySize = dx12Desc.DepthOrArraySize;
                    dx12UavDescription.Texture1DArray.FirstArraySlice = 0;
                    dx12UavDescription.Texture1DArray.MipSlice = 0;

                    dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                    dx12SrvDescription.Texture1DArray.ArraySize = dx12Desc.DepthOrArraySize;
                    dx12SrvDescription.Texture1DArray.FirstArraySlice = 0;
                    dx12SrvDescription.Texture1DArray.MipLevels = dx12Desc.MipLevels;
                    dx12SrvDescription.Texture1DArray.MostDetailedMip = 0;
                }
                else
                {
                    dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                    dx12UavDescription.Texture1D.MipSlice = 0;

                    dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                    dx12SrvDescription.Texture1D.MipLevels = dx12Desc.MipLevels;
                    dx12SrvDescription.Texture1D.MostDetailedMip = 0;
                }
                break;

            case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
            {
                if (dx12Desc.DepthOrArraySize > 1 || requestArrayView)
                {
                    dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                    dx12UavDescription.Texture2DArray.ArraySize = dx12Desc.DepthOrArraySize;
                    dx12UavDescription.Texture2DArray.FirstArraySlice = 0;
                    dx12UavDescription.Texture2DArray.MipSlice = 0;
                    dx12UavDescription.Texture2DArray.PlaneSlice = 0;

                    dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                    dx12SrvDescription.Texture2DArray.ArraySize = dx12Desc.DepthOrArraySize;
                    dx12SrvDescription.Texture2DArray.FirstArraySlice = 0;
                    dx12SrvDescription.Texture2DArray.MipLevels = dx12Desc.MipLevels;
                    dx12SrvDescription.Texture2DArray.MostDetailedMip = 0;
                    dx12SrvDescription.Texture2DArray.PlaneSlice = 0;
                }
                else
                {
                    dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    dx12UavDescription.Texture2D.MipSlice = 0;
                    dx12UavDescription.Texture2D.PlaneSlice = 0;

                    dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    dx12SrvDescription.Texture2D.MipLevels = dx12Desc.MipLevels;
                    dx12SrvDescription.Texture2D.MostDetailedMip = 0;
                    dx12SrvDescription.Texture2D.PlaneSlice = 0;
                }
                break;
            }

            case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                dx12SrvDescription.Texture3D.MipLevels = dx12Resource->GetDesc().MipLevels;
                dx12SrvDescription.Texture3D.MostDetailedMip = 0;
                break;

            default:
                break;
            }

            if (dx12Desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {

                dx12SrvDescription.Buffer.FirstElement        = 0;
                dx12SrvDescription.Buffer.StructureByteStride = backendResource->resourceDescription.stride;
                dx12SrvDescription.Buffer.NumElements         = backendResource->resourceDescription.size / backendResource->resourceDescription.stride;
                D3D12_CPU_DESCRIPTOR_HANDLE dx12CpuHandle     = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
                dx12CpuHandle.ptr += outTexture->internalIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, dx12CpuHandle);

                // UAV
                if (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {

                    FFX_ASSERT(effectContext.nextStaticUavDescriptor + 1 < effectContext.nextDynamicUavDescriptor);
                    backendResource->uavDescCount = 1;
                    backendResource->uavDescIndex = effectContext.nextStaticUavDescriptor++;

                    dx12UavDescription.Buffer.FirstElement = 0;
                    dx12UavDescription.Buffer.StructureByteStride = backendResource->resourceDescription.stride;
                    dx12UavDescription.Buffer.NumElements = backendResource->resourceDescription.size / backendResource->resourceDescription.stride;
                    dx12UavDescription.Buffer.CounterOffsetInBytes = 0;

                    dx12CpuHandle = backendContext->descHeapUavGpu->GetCPUDescriptorHandleForHeapStart();
                    dx12CpuHandle.ptr += (backendResource->uavDescIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);

                    dx12CpuHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                    dx12CpuHandle.ptr += (backendResource->uavDescIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);

                    effectContext.nextStaticUavDescriptor++;
                }
            }
            else {
                // CPU readable
                D3D12_CPU_DESCRIPTOR_HANDLE dx12CpuHandle = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
                dx12CpuHandle.ptr += outTexture->internalIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, dx12CpuHandle);

                // UAV
                if (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {

                    const int32_t uavDescriptorCount = (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? dx12Desc.MipLevels : 1;
                    FFX_ASSERT(effectContext.nextStaticUavDescriptor + uavDescriptorCount < effectContext.nextDynamicUavDescriptor);

                    backendResource->uavDescCount = uavDescriptorCount;
                    backendResource->uavDescIndex = effectContext.nextStaticUavDescriptor;

                    for (int32_t currentMipIndex = 0; currentMipIndex < uavDescriptorCount; ++currentMipIndex) {

                        if (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE3D)
                        {
                            dx12UavDescription.Texture3D.MipSlice    = currentMipIndex;
                            dx12UavDescription.Texture3D.FirstWSlice = currentMipIndex;
                            dx12UavDescription.Texture3D.WSize       = createResourceDescription->resourceDescription.depth;
                        }
                        else if (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE2D)
                            dx12UavDescription.Texture2D.MipSlice = currentMipIndex;
                        else if (createResourceDescription->resourceDescription.type == FFX_RESOURCE_TYPE_TEXTURE1D)
                            dx12UavDescription.Texture1D.MipSlice = currentMipIndex;

                        dx12CpuHandle = backendContext->descHeapUavGpu->GetCPUDescriptorHandleForHeapStart();
                        dx12CpuHandle.ptr += (backendResource->uavDescIndex + currentMipIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                        dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);

                        dx12CpuHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                        dx12CpuHandle.ptr += (backendResource->uavDescIndex + currentMipIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                        dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);
                    }

                    effectContext.nextStaticUavDescriptor += uavDescriptorCount;
                }
            }
        }

        // create upload resource and upload job
        if (initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED) {

            FfxResourceInternal copySrc;
            FfxCreateResourceDescription uploadDescription = { *createResourceDescription };
            uploadDescription.heapType = FFX_HEAP_TYPE_UPLOAD;
            uploadDescription.resourceDescription.usage = FFX_RESOURCE_USAGE_READ_ONLY;
            uploadDescription.initialState = FFX_RESOURCE_STATE_GENERIC_READ;

            backendInterface->fpCreateResource(backendInterface, &uploadDescription, effectContextId, &copySrc);

            // setup the upload job
            FfxGpuJobDescription copyJob  = { FFX_GPU_JOB_COPY };
#ifdef FFX_DEBUG
            strncpy(copyJob.jobLabel, "Resource Initialization Copy", FFX_RESOURCE_NAME_SIZE - 1);
            copyJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
            copyJob.copyJobDescriptor.src = copySrc;
            copyJob.copyJobDescriptor.dst = *outTexture;
            copyJob.copyJobDescriptor.srcOffset = 0;
            copyJob.copyJobDescriptor.dstOffset = 0;
            copyJob.copyJobDescriptor.size      = 0;

            backendInterface->fpScheduleGpuJob(backendInterface, &copyJob);
        }
    }
    
    effectContext.vramUsage.totalUsageInBytes += resourceSize;
    if ((createResourceDescription->resourceDescription.flags & FFX_RESOURCE_FLAGS_ALIASABLE) == FFX_RESOURCE_FLAGS_ALIASABLE)
    {
        effectContext.vramUsage.aliasableUsageInBytes += resourceSize;
    }

    return FFX_OK;
}

FfxErrorCode DestroyResourceDX12(
    FfxInterface* backendInterface,
    FfxResourceInternal resource,
	FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
	BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];
	if ((resource.internalIndex >= int32_t(effectContextId * FFX_MAX_RESOURCE_COUNT)) && (resource.internalIndex < int32_t(effectContext.nextStaticResource))) {
		ID3D12Resource* dx12Resource = getDX12ResourcePtr(backendContext, resource.internalIndex);

		if (dx12Resource) {

            uint64_t resourceSize = GetResourceGpuMemorySizeDX12(dx12Resource);

			dx12Resource->Release();

            // update effect memory usage
            effectContext.vramUsage.totalUsageInBytes -= resourceSize;
            if ((backendContext->pResources[resource.internalIndex].resourceDescription.flags & FFX_RESOURCE_FLAGS_ALIASABLE) == FFX_RESOURCE_FLAGS_ALIASABLE)
            {
                effectContext.vramUsage.aliasableUsageInBytes -= resourceSize;
            }

			backendContext->pResources[resource.internalIndex].resourcePtr = nullptr;
		}
        
        return FFX_OK;
	}

	return FFX_ERROR_OUT_OF_RANGE;
}

DXGI_FORMAT patchDxgiFormatWithFfxUsage(DXGI_FORMAT dxResFmt, FfxSurfaceFormat ffxFmt)
{
    DXGI_FORMAT fromFfx = ffxGetDX12FormatFromSurfaceFormat(ffxFmt);
    DXGI_FORMAT fmt = dxResFmt;

    switch (fmt)
    {
    case DXGI_FORMAT_UNKNOWN:
        return fromFfx;

    // fixup RGBA8 with SRGB flag passed in the ffxSurfaceFormat
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return fromFfx;
    
    // fixup depth formats as ffxGetDX12FormatFromSurfaceFormat will result in wrong format
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;

    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;

    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;

    case DXGI_FORMAT_D16_UNORM:
        return DXGI_FORMAT_R16_UNORM;

    default:
        break;
    }
    return fmt;
}

FfxErrorCode MapResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource, void** ptr)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    if (FAILED(backendContext->pResources[resource.internalIndex].resourcePtr->Map(0, NULL, ptr)))
        return FFX_ERROR_BACKEND_API_ERROR;

    return FFX_OK;
}

FfxErrorCode UnmapResourceDX12(FfxInterface* backendInterface, FfxResourceInternal resource)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    backendContext->pResources[resource.internalIndex].resourcePtr->Unmap(0, NULL);

    return FFX_OK;
}

FfxErrorCode RegisterResourceDX12(
    FfxInterface* backendInterface,
    const FfxResource* inFfxResource,
    FfxUInt32 effectContextId,
    FfxResourceInternal* outFfxResourceInternal
)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)(backendInterface->scratchBuffer);
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendContext->device);
    ID3D12Resource* dx12Resource = reinterpret_cast<ID3D12Resource*>(inFfxResource->resource);
    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    FfxResourceStates state = inFfxResource->state;

    if (dx12Resource == nullptr) {

        outFfxResourceInternal->internalIndex = 0; // Always maps to FFX_<feature>_RESOURCE_IDENTIFIER_NULL;
        return FFX_OK;
    }

    FFX_ASSERT(effectContext.nextDynamicResource > effectContext.nextStaticResource);
    outFfxResourceInternal->internalIndex = effectContext.nextDynamicResource--;

    BackendContext_DX12::Resource* backendResource = &backendContext->pResources[outFfxResourceInternal->internalIndex];
    backendResource->resourcePtr = dx12Resource;
    backendResource->initialState = state;
    backendResource->currentState = state;

#ifdef _DEBUG
    const wchar_t* name = inFfxResource->name;
    if (name) {
        ffxNarrowToWide(backendResource->resourceName, std::size(backendResource->resourceName), name);
    }
#endif

    // create resource views
    if (dx12Resource) {

        D3D12_UNORDERED_ACCESS_VIEW_DESC dx12UavDescription = {};
        D3D12_SHADER_RESOURCE_VIEW_DESC dx12SrvDescription = {};
        D3D12_RESOURCE_DESC dx12Desc = dx12Resource->GetDesc();

        // we still want to respect the format provided in the description for SRGB or TYPELESS resources
        DXGI_FORMAT descFormat = patchDxgiFormatWithFfxUsage(dx12Desc.Format, inFfxResource->description.format);

        dx12UavDescription.Format = convertFormatUav(descFormat);
        // Will support something other than this only where there is an actual need for it
        dx12SrvDescription.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        dx12SrvDescription.Format = convertFormatSrv(descFormat);

        bool requestArrayView = FFX_CONTAINS_FLAG(inFfxResource->description.usage, FFX_RESOURCE_USAGE_ARRAYVIEW);

        switch (dx12Desc.Dimension) {

        case D3D12_RESOURCE_DIMENSION_BUFFER:
            dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            backendResource->resourceDescription.type = FFX_RESOURCE_TYPE_BUFFER;
            backendResource->resourceDescription.size = inFfxResource->description.size;
            backendResource->resourceDescription.stride = inFfxResource->description.stride;
            backendResource->resourceDescription.alignment = 0;
            break;

        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            if (dx12Desc.DepthOrArraySize > 1 || requestArrayView)
            {
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                dx12UavDescription.Texture1DArray.ArraySize = dx12Desc.DepthOrArraySize;
                dx12UavDescription.Texture1DArray.FirstArraySlice = 0;
                dx12UavDescription.Texture1DArray.MipSlice = 0;

                dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                dx12SrvDescription.Texture1DArray.ArraySize = dx12Desc.DepthOrArraySize;
                dx12SrvDescription.Texture1DArray.FirstArraySlice = 0;
                dx12SrvDescription.Texture1DArray.MipLevels = dx12Desc.MipLevels;
                dx12SrvDescription.Texture1DArray.MostDetailedMip = 0;
            }
            else
            {
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                dx12UavDescription.Texture1D.MipSlice = 0;

                dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                dx12SrvDescription.Texture1D.MipLevels = dx12Desc.MipLevels;
                dx12SrvDescription.Texture1D.MostDetailedMip = 0;
            }

            backendResource->resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE1D;
            backendResource->resourceDescription.format = inFfxResource->description.format;
            backendResource->resourceDescription.width = inFfxResource->description.width;
            backendResource->resourceDescription.mipCount = inFfxResource->description.mipCount;
            backendResource->resourceDescription.depth = inFfxResource->description.depth;
            break;

        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        {
            if (dx12Desc.DepthOrArraySize > 1 || requestArrayView)
            {
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                dx12UavDescription.Texture2DArray.ArraySize = dx12Desc.DepthOrArraySize;
                dx12UavDescription.Texture2DArray.FirstArraySlice = 0;
                dx12UavDescription.Texture2DArray.MipSlice = 0;
                dx12UavDescription.Texture2DArray.PlaneSlice = 0;

                dx12SrvDescription.ViewDimension = inFfxResource->description.type == FFX_RESOURCE_TYPE_TEXTURE_CUBE ? D3D12_SRV_DIMENSION_TEXTURECUBE : D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                dx12SrvDescription.Texture2DArray.ArraySize = dx12Desc.DepthOrArraySize;
                dx12SrvDescription.Texture2DArray.FirstArraySlice = 0;
                dx12SrvDescription.Texture2DArray.MipLevels = dx12Desc.MipLevels;
                dx12SrvDescription.Texture2DArray.MostDetailedMip = 0;
                dx12SrvDescription.Texture2DArray.PlaneSlice = 0;
            }
            else
            {
                dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                dx12UavDescription.Texture2D.MipSlice = 0;
                dx12UavDescription.Texture2D.PlaneSlice = 0;

                dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                dx12SrvDescription.Texture2D.MipLevels = dx12Desc.MipLevels;
                dx12SrvDescription.Texture2D.MostDetailedMip = 0;
                dx12SrvDescription.Texture2D.PlaneSlice = 0;
            }

            backendResource->resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE2D;
            backendResource->resourceDescription.format = inFfxResource->description.format;
            backendResource->resourceDescription.width = inFfxResource->description.width;
            backendResource->resourceDescription.height = inFfxResource->description.height;
            backendResource->resourceDescription.mipCount = inFfxResource->description.mipCount;
            backendResource->resourceDescription.depth = inFfxResource->description.depth;
            break;
        }

        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            dx12UavDescription.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            dx12UavDescription.Texture3D.FirstWSlice      = 0; // Bind all W slices
            dx12UavDescription.Texture3D.WSize            = std::numeric_limits<UINT>::max();
            dx12SrvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            dx12SrvDescription.Texture3D.MipLevels = dx12Desc.MipLevels;
            dx12SrvDescription.Texture3D.MostDetailedMip = 0;

            backendResource->resourceDescription.type = FFX_RESOURCE_TYPE_TEXTURE3D;
            backendResource->resourceDescription.format = inFfxResource->description.format;
            backendResource->resourceDescription.width = inFfxResource->description.width;
            backendResource->resourceDescription.height = inFfxResource->description.height;
            backendResource->resourceDescription.mipCount = inFfxResource->description.mipCount;
            backendResource->resourceDescription.depth = inFfxResource->description.depth;
            break;

        default:
            break;
        }

        if (dx12Desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {

            // UAV
            if (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {

                FFX_ASSERT(effectContext.nextDynamicUavDescriptor > effectContext.nextStaticUavDescriptor);
                backendResource->uavDescCount = 1;
                backendResource->uavDescIndex = effectContext.nextDynamicUavDescriptor--;

                dx12UavDescription.Format = DXGI_FORMAT_UNKNOWN;
                dx12UavDescription.Buffer.FirstElement = 0;
                dx12UavDescription.Buffer.StructureByteStride = backendResource->resourceDescription.stride;
                dx12UavDescription.Buffer.NumElements = backendResource->resourceDescription.size / backendResource->resourceDescription.stride;
                dx12UavDescription.Buffer.CounterOffsetInBytes = 0;

                D3D12_CPU_DESCRIPTOR_HANDLE dx12CpuHandle = backendContext->descHeapUavGpu->GetCPUDescriptorHandleForHeapStart();
                dx12CpuHandle.ptr += (backendResource->uavDescIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);

                dx12CpuHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                dx12CpuHandle.ptr += (backendResource->uavDescIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, dx12CpuHandle);
            }

            {
                dx12SrvDescription.Format = DXGI_FORMAT_UNKNOWN;
                dx12SrvDescription.Buffer.FirstElement        = 0;
                dx12SrvDescription.Buffer.StructureByteStride = backendResource->resourceDescription.stride;
                dx12SrvDescription.Buffer.NumElements         = backendResource->resourceDescription.size / backendResource->resourceDescription.stride;
                D3D12_CPU_DESCRIPTOR_HANDLE dx12CpuHandle     = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
                dx12CpuHandle.ptr += outFfxResourceInternal->internalIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, dx12CpuHandle);
                backendResource->srvDescIndex = outFfxResourceInternal->internalIndex;
            }
        }
        else {

            // CPU readable
            D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
            cpuHandle.ptr += outFfxResourceInternal->internalIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, cpuHandle);
            backendResource->srvDescIndex = outFfxResourceInternal->internalIndex;

            // UAV
            if (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {

                const int32_t uavDescriptorsCount = (dx12Desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? dx12Desc.MipLevels : 1;
                FFX_ASSERT(effectContext.nextDynamicUavDescriptor - uavDescriptorsCount + 1 > effectContext.nextStaticUavDescriptor);

                backendResource->uavDescCount = uavDescriptorsCount;
                backendResource->uavDescIndex = effectContext.nextDynamicUavDescriptor - uavDescriptorsCount + 1;

                for (int32_t currentMipIndex = 0; currentMipIndex < uavDescriptorsCount; ++currentMipIndex) {

                    switch (dx12Desc.Dimension)
                    {
                    case D3D12_RESOURCE_DIMENSION_BUFFER:
                        break;

                    case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
                    case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
                    case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                        // TextureXD<Y>.MipSlice values map to same mem
                        dx12UavDescription.Texture2D.MipSlice = currentMipIndex;
                        break;

                    default:
                        FFX_ASSERT_MESSAGE(false, "Invalid View Dimension");
                        break;
                    }

                    cpuHandle = backendContext->descHeapUavGpu->GetCPUDescriptorHandleForHeapStart();
                    cpuHandle.ptr += (backendResource->uavDescIndex + currentMipIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, cpuHandle);

                    cpuHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                    cpuHandle.ptr += (backendResource->uavDescIndex + currentMipIndex) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    dx12Device->CreateUnorderedAccessView(dx12Resource, 0, &dx12UavDescription, cpuHandle);
                }

                effectContext.nextDynamicUavDescriptor -= uavDescriptorsCount;
            }
        }
    }

    return FFX_OK;
}

FfxResource GetResourceDX12(FfxInterface* backendInterface, FfxResourceInternal inResource)
{
    FFX_ASSERT(nullptr != backendInterface);
    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    FfxResourceDescription ffxResDescription = backendInterface->fpGetResourceDescription(backendInterface, inResource);

    FfxResource resource = {};
    resource.resource = resource.resource = reinterpret_cast<void*>(backendContext->pResources[inResource.internalIndex].resourcePtr);
    resource.state = backendContext->pResources[inResource.internalIndex].currentState;
    resource.description = ffxResDescription;

#ifdef _DEBUG
    if (backendContext->pResources[inResource.internalIndex].resourceName)
    {
        strncpy(resource.name, backendContext->pResources[inResource.internalIndex].resourceName, FFX_RESOURCE_NAME_SIZE - 1);
        resource.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }
#endif

    return resource;
}

// dispose dynamic resources: This should be called at the end of the frame
FfxErrorCode UnregisterResourcesDX12(FfxInterface* backendInterface, FfxCommandList commandList, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_DX12* backendContext = (BackendContext_DX12*)(backendInterface->scratchBuffer);
    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    // Walk back all the resources that don't belong to us and reset them to their initial state
    for (uint32_t resourceIndex = ++effectContext.nextDynamicResource; resourceIndex < (effectContextId * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT; ++resourceIndex)
    {
        FfxResourceInternal internalResource;
        internalResource.internalIndex = resourceIndex;

        BackendContext_DX12::Resource* backendResource = &backendContext->pResources[resourceIndex];
        addBarrier(backendContext, &internalResource, backendResource->initialState);
    }

    FFX_ASSERT(nullptr != commandList);
    ID3D12GraphicsCommandList* pCmdList = reinterpret_cast<ID3D12GraphicsCommandList*>(commandList);

    flushBarriers(backendContext, pCmdList);

    effectContext.nextDynamicResource      = (effectContextId * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;
    effectContext.nextDynamicUavDescriptor = (effectContextId * FFX_MAX_RESOURCE_COUNT) + FFX_MAX_RESOURCE_COUNT - 1;

    return FFX_OK;
}

FfxErrorCode registerStaticTextureSrv(BackendContext_DX12* backendContext, const FfxResource* inResource, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_DX12::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    ID3D12Device*                       dx12Device     = reinterpret_cast<ID3D12Device*>(backendContext->device);
    ID3D12Resource*                     dx12Resource   = reinterpret_cast<ID3D12Resource*>(inResource->resource);

    if (effectContext.bindlessTextureSrvHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    if (dx12Resource == nullptr)
        return FFX_OK;

    // create resource views
    if (dx12Resource)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC dx12SrvDescription = {};
        dx12SrvDescription.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        dx12SrvDescription.Format                          = convertFormatSrv(dx12Resource->GetDesc().Format);

        uint32_t depthArraySize = dx12Resource->GetDesc().DepthOrArraySize;

        switch (dx12Resource->GetDesc().Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            if (depthArraySize > 1)
            {
                dx12SrvDescription.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                dx12SrvDescription.Texture1DArray.MipLevels = dx12Resource->GetDesc().MipLevels;
                dx12SrvDescription.Texture1DArray.ArraySize = std::numeric_limits<UINT>::max();
            }
            else
            {
                dx12SrvDescription.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE1D;
                dx12SrvDescription.Texture1D.MipLevels = dx12Resource->GetDesc().MipLevels;
            }
            break;

        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
            if (depthArraySize > 1)
            {
                dx12SrvDescription.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                dx12SrvDescription.Texture2DArray.MipLevels = dx12Resource->GetDesc().MipLevels;
                dx12SrvDescription.Texture2DArray.ArraySize = depthArraySize;
                dx12SrvDescription.Texture2DArray.ArraySize = std::numeric_limits<UINT>::max();
            }
            else
            {
                dx12SrvDescription.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                dx12SrvDescription.Texture2D.MipLevels = dx12Resource->GetDesc().MipLevels;
            }
            break;

        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            dx12SrvDescription.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE3D;
            dx12SrvDescription.Texture3D.MipLevels = dx12Resource->GetDesc().MipLevels;
            break;

        default:
            // Only texture resources are allowed here.
            FFX_ASSERT(false);
            break;
        }

        // CPU readable
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = backendContext->descRingBuffer->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr +=
            (effectContext.bindlessTextureSrvHeapStart + index) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, cpuHandle);
    }

    return FFX_OK;
}

FfxErrorCode registerStaticBufferSrv(BackendContext_DX12* backendContext, const FfxResource* inResource, uint32_t offset, uint32_t size, uint32_t stride, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_DX12::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    ID3D12Device*                       dx12Device     = reinterpret_cast<ID3D12Device*>(backendContext->device);
    ID3D12Resource*                     dx12Resource   = reinterpret_cast<ID3D12Resource*>(inResource->resource);

    if (effectContext.bindlessBufferSrvHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    constexpr uint32_t ShaderComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    if (dx12Resource == nullptr)
        return FFX_OK;

    // create resource views
    if (dx12Resource)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC dx12SrvDescription = {};
        dx12SrvDescription.Shader4ComponentMapping         = ShaderComponentMapping;

        switch (dx12Resource->GetDesc().Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_BUFFER:
        {
            FFX_ASSERT(stride > 0);

            const uint32_t actualSize = size > 0 ? size : uint32_t(dx12Resource->GetDesc().Width);

            dx12SrvDescription.Format                     = DXGI_FORMAT_UNKNOWN;
            dx12SrvDescription.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
            dx12SrvDescription.Buffer.FirstElement        = offset / stride;
            dx12SrvDescription.Buffer.NumElements         = actualSize / stride;
            dx12SrvDescription.Buffer.StructureByteStride = stride;
            dx12SrvDescription.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
            dx12SrvDescription.Shader4ComponentMapping    = ShaderComponentMapping;
            break;
        }
        default:
            // Only buffer resources are allowed here.
            FFX_ASSERT(false);
            break;
        }

        // CPU readable
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = backendContext->descRingBuffer->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr +=
            (effectContext.bindlessBufferSrvHeapStart + index) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12Device->CreateShaderResourceView(dx12Resource, &dx12SrvDescription, cpuHandle);
    }

    return FFX_OK;
}

FfxErrorCode registerStaticTextureUav(BackendContext_DX12* backendContext, const FfxResource* inResource, uint32_t mip, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_DX12::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    ID3D12Device*                       dx12Device     = reinterpret_cast<ID3D12Device*>(backendContext->device);
    ID3D12Resource*                     dx12Resource   = reinterpret_cast<ID3D12Resource*>(inResource->resource);

    if (effectContext.bindlessTextureUavHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    if (dx12Resource == nullptr)
        return FFX_OK;

    // create resource views
    if (dx12Resource)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC dx12UavDescription = {};
        dx12UavDescription.Format                           = convertFormatSrv(dx12Resource->GetDesc().Format);

        uint32_t depthArraySize = dx12Resource->GetDesc().DepthOrArraySize;

        switch (dx12Resource->GetDesc().Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
        {
            if (depthArraySize > 1)
            {
                dx12UavDescription.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                dx12UavDescription.Texture1DArray.ArraySize       = depthArraySize;
                dx12UavDescription.Texture1DArray.FirstArraySlice = 0;
                dx12UavDescription.Texture1DArray.MipSlice        = mip;
            }
            else
            {
                dx12UavDescription.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE1D;
                dx12UavDescription.Texture1D.MipSlice = mip;
            }
            break;
        }
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        {
            if (depthArraySize > 1)
            {
                dx12UavDescription.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                dx12UavDescription.Texture2DArray.ArraySize       = depthArraySize;
                dx12UavDescription.Texture2DArray.FirstArraySlice = 0;
                dx12UavDescription.Texture2DArray.MipSlice        = mip;
                dx12UavDescription.Texture2DArray.PlaneSlice      = 0;
            }
            else
            {
                dx12UavDescription.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE2D;
                dx12UavDescription.Texture2D.MipSlice   = mip;
                dx12UavDescription.Texture2D.PlaneSlice = 0;
            }
            break;
        }
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
        {
            dx12UavDescription.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
            dx12UavDescription.Texture3D.FirstWSlice = 0;  // Bind all W slices
            dx12UavDescription.Texture3D.WSize       = std::numeric_limits<UINT>::max();
            break;
        }
        default:
            // Only texture resources are allowed here.
            FFX_ASSERT(false);
            break;
        }

        // CPU readable
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = backendContext->descRingBuffer->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr +=
            (effectContext.bindlessTextureUavHeapStart + index) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12Device->CreateUnorderedAccessView(dx12Resource, nullptr, &dx12UavDescription, cpuHandle);
    }

    return FFX_OK;
}

FfxErrorCode registerStaticBufferUav(BackendContext_DX12* backendContext, const FfxResource* inResource, uint32_t offset, uint32_t size, uint32_t stride, uint32_t index, FfxUInt32 effectContextId)
{
    BackendContext_DX12::EffectContext& effectContext  = backendContext->pEffectContexts[effectContextId];
    ID3D12Device*                       dx12Device     = reinterpret_cast<ID3D12Device*>(backendContext->device);
    ID3D12Resource*                     dx12Resource   = reinterpret_cast<ID3D12Resource*>(inResource->resource);

    if (effectContext.bindlessBufferUavHeapSize <= index)
    {
        FFX_ASSERT(false);
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    }

    if (dx12Resource == nullptr)
        return FFX_OK;

    // create resource views
    if (dx12Resource)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC dx12UavDescription = {};

        switch (dx12Resource->GetDesc().Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_BUFFER:
        {
            FFX_ASSERT(stride > 0);

            const uint32_t actualSize = size > 0 ? size : uint32_t(dx12Resource->GetDesc().Width);

            dx12UavDescription.Format                      = DXGI_FORMAT_UNKNOWN;
            dx12UavDescription.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
            dx12UavDescription.Buffer.FirstElement         = offset / stride;
            dx12UavDescription.Buffer.NumElements          = actualSize / stride;
            dx12UavDescription.Buffer.StructureByteStride  = stride;
            dx12UavDescription.Buffer.CounterOffsetInBytes = 0;
            dx12UavDescription.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_NONE;
            break;
        }
        default:
            // Only buffer resources are allowed here.
            FFX_ASSERT(false);
            break;
        }

        // CPU readable
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = backendContext->descRingBuffer->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr +=
            (effectContext.bindlessBufferUavHeapStart + index) * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12Device->CreateUnorderedAccessView(dx12Resource, nullptr, &dx12UavDescription, cpuHandle);
    }

    return FFX_OK;
}

FfxErrorCode RegisterStaticResourceDX12(FfxInterface* backendInterface, const FfxStaticResourceDescription* desc, FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != desc);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    switch (desc->descriptorType)
    {
    case FFX_DESCRIPTOR_TEXTURE_SRV:
        return registerStaticTextureSrv(backendContext, desc->resource, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_BUFFER_SRV:
        return registerStaticBufferSrv(backendContext, desc->resource, desc->bufferOffset, desc->bufferSize, desc->bufferStride, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_TEXTURE_UAV:
        return registerStaticTextureUav(backendContext, desc->resource, desc->textureUavMip, desc->descriptorIndex, effectContextId);
    case FFX_DESCRIPTOR_BUFFER_UAV:
        return registerStaticBufferUav(backendContext, desc->resource, desc->bufferOffset, desc->bufferSize, desc->bufferStride, desc->descriptorIndex, effectContextId);
    default:
        return FFX_ERROR_INVALID_ARGUMENT;
    }
}

FfxResourceDescription GetResourceDescriptorDX12(
    FfxInterface* backendInterface,
    FfxResourceInternal resource)
{
    FFX_ASSERT(NULL != backendInterface);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    FfxResourceDescription resourceDescription = backendContext->pResources[resource.internalIndex].resourceDescription;
    return resourceDescription;
}

FfxErrorCode StageConstantBufferDataDX12(FfxInterface* backendInterface, void* data, FfxUInt32 size, FfxConstantBuffer* constantBuffer)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    if (data && constantBuffer)
    {
        if ((backendContext->stagingRingBufferBase + FFX_ALIGN_UP(size, 256)) >= FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE)
            backendContext->stagingRingBufferBase = 0;

        uint32_t* dstPtr = (uint32_t*)(backendContext->pStagingRingBuffer + backendContext->stagingRingBufferBase);

        memcpy(dstPtr, data, size);

        constantBuffer->data            = dstPtr;
        constantBuffer->num32BitEntries = size / sizeof(uint32_t);

        backendContext->stagingRingBufferBase += FFX_ALIGN_UP(size, 256);

        return FFX_OK;
    }
    else
        return FFX_ERROR_INVALID_POINTER;
}

D3D12_TEXTURE_ADDRESS_MODE FfxGetAddressModeDX12(const FfxAddressMode& addressMode)
{
    switch (addressMode)
    {
    case FFX_ADDRESS_MODE_WRAP:
        return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case FFX_ADDRESS_MODE_MIRROR:
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case FFX_ADDRESS_MODE_CLAMP:
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case FFX_ADDRESS_MODE_BORDER:
        return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case FFX_ADDRESS_MODE_MIRROR_ONCE:
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    default:
        FFX_ASSERT_MESSAGE(false, "Unsupported addressing mode requested. Please implement");
        return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        break;
    }
}

/*
 * Shared body of the compute and graphics pipeline creators.
 *
 * The two differ in exactly two places, and neither is the root signature: a D3D12 root
 * signature is stage-agnostic (visibility is ALL), and the binding flattening that fills
 * outPipeline->*Bindings is identical for both. What differs is
 *
 *   1. which blobs are fetched -- a graphics pipeline needs the vertex blob as well as the
 *      pixel one, and the widened callback carries it in the second out-parameter; and
 *   2. what is built at the end -- a compute PSO there, and for graphics no PSO at all,
 *      because a graphics PSO bakes its render-target formats and those are only known
 *      when the job runs. See GraphicsPipelineDX12 in ffx_dx12.h.
 *
 * Duplicating the root signature code to express that would have been ~250 copied lines
 * whose only job is to stay in step with the original. One function and a flag is smaller,
 * and there is one place for the two to stay consistent.
 */
static FfxErrorCode CreatePipelineInternal(
    FfxInterface* backendInterface,
    FfxEffect effect,
    FfxPass pass,
    uint32_t permutationOptions,
    const FfxPipelineDescription* pipelineDescription,
    FfxUInt32                     effectContextId,
    FfxPipelineState* outPipeline,
    bool                          graphics)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != pipelineDescription);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;
    ID3D12Device* dx12Device = backendContext->device;

    FfxShaderBlob shaderBlob     = { };
    FfxShaderBlob vertShaderBlob = { };
    /*
     * The Arm fork widened this callback. 1.1.3 was
     * (effect, pass, FfxBindStage bindStage, permutationOptions, outBlob) -- it carried
     * the shader stage in the call and had no notion of a vertex blob or a data-graph
     * blob. The current signature is
     * (effect, pass, permutationOptions, outBlob, outVertBlob, outDataGraphBlob).
     *
     * A compute pipeline has no vertex shader, so the compute path asks for nothing in the
     * second out-parameter -- which is what the hardcoded FFX_BIND_COMPUTE_SHADER_STAGE
     * used to say. A graphics pipeline needs both halves.
     */
    backendInterface->fpGetPermutationBlobByIndex(effect, pass, permutationOptions, &shaderBlob,
                                                  graphics ? &vertShaderBlob : nullptr, nullptr);
    FFX_ASSERT(shaderBlob.data && shaderBlob.size);
    if (graphics)
    {
        FFX_ASSERT(vertShaderBlob.data && vertShaderBlob.size);
    }

    int32_t staticTextureSrvCount = 0;
    int32_t staticBufferSrvCount  = 0;
    int32_t staticTextureUavCount = 0;
    int32_t staticBufferUavCount  = 0;

    int32_t staticTextureSrvSpace = -1;
    int32_t staticBufferSrvSpace  = -1;
    int32_t staticTextureUavSpace = -1;
    int32_t staticBufferUavSpace  = -1;

    // set up root signature
    // easiest implementation: simply create one root signature per pipeline
    // should add some management later on to avoid unnecessarily re-binding the root signature
    {
        FFX_ASSERT(pipelineDescription->samplerCount <= FFX_MAX_SAMPLERS);
        const size_t samplerCount = pipelineDescription->samplerCount;
        D3D12_STATIC_SAMPLER_DESC dx12SamplerDescriptions[FFX_MAX_SAMPLERS];
        for (uint32_t currentSamplerIndex = 0; currentSamplerIndex < samplerCount; ++currentSamplerIndex) {

            D3D12_STATIC_SAMPLER_DESC dx12SamplerDesc = {};

            dx12SamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            dx12SamplerDesc.MinLOD = 0.f;
            dx12SamplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
            dx12SamplerDesc.MipLODBias = 0.f;
            dx12SamplerDesc.MaxAnisotropy = 16;
            dx12SamplerDesc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
            dx12SamplerDesc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            dx12SamplerDesc.AddressU = FfxGetAddressModeDX12(pipelineDescription->samplers[currentSamplerIndex].addressModeU);
            dx12SamplerDesc.AddressV = FfxGetAddressModeDX12(pipelineDescription->samplers[currentSamplerIndex].addressModeV);
            dx12SamplerDesc.AddressW = FfxGetAddressModeDX12(pipelineDescription->samplers[currentSamplerIndex].addressModeW);

            switch (pipelineDescription->samplers[currentSamplerIndex].filter)
            {
            case FFX_FILTER_TYPE_MINMAGMIP_POINT:
                dx12SamplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                break;
            case FFX_FILTER_TYPE_MINMAGMIP_LINEAR:
                dx12SamplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                break;
            case FFX_FILTER_TYPE_MINMAGLINEARMIP_POINT:
                dx12SamplerDesc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                break;

            default:
                FFX_ASSERT_MESSAGE(false, "Unsupported filter type requested. Please implement");
                break;
            }

            dx12SamplerDescriptions[currentSamplerIndex] = dx12SamplerDesc;
            dx12SamplerDescriptions[currentSamplerIndex].ShaderRegister = (UINT)currentSamplerIndex;
        }

        // storage for maximum number of descriptor ranges.
        const int32_t maximumDescriptorRangeSize = 3;
        D3D12_DESCRIPTOR_RANGE dx12Ranges[maximumDescriptorRangeSize] = {};
        int32_t currentDescriptorRangeIndex = 0;

        // storage for maximum number of root parameters.
        const int32_t maximumRootParameters = 10;
        D3D12_ROOT_PARAMETER dx12RootParameters[maximumRootParameters] = {};
        int32_t currentRootParameterIndex = 0;

        int32_t uavCount              = 0;
        int32_t maxUavSlotIndex       = 0;

        for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; uavIndex++)
        {
            // count static uavs separately.
            if (shaderBlob.boundUAVBufferSpaces[uavIndex] != 0)
            {
                if (staticBufferUavCount > 0)
                    FFX_ASSERT(uint32_t(staticBufferUavSpace) != shaderBlob.boundUAVBufferSpaces[uavIndex]);

                staticBufferUavCount += shaderBlob.boundUAVBufferCounts[uavIndex];
                staticBufferUavSpace = shaderBlob.boundUAVBufferSpaces[uavIndex];
                continue;
            }

            uint32_t bindCount = shaderBlob.boundUAVBufferCounts[uavIndex];

            uavCount += bindCount;

            if (shaderBlob.boundUAVBuffers[uavIndex] > uint32_t(maxUavSlotIndex))
                maxUavSlotIndex = shaderBlob.boundUAVBuffers[uavIndex] + (bindCount - 1);
        }

        for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; uavIndex++)
        {
            // count static uavs separately.
            if (shaderBlob.boundUAVTextureSpaces[uavIndex] != 0)
            {
                if (staticTextureUavCount > 0)
                    FFX_ASSERT(uint32_t(staticTextureUavSpace) != shaderBlob.boundUAVTextureSpaces[uavIndex]);

                staticTextureUavCount += shaderBlob.boundUAVTextureCounts[uavIndex];
                staticTextureUavSpace = shaderBlob.boundUAVTextureSpaces[uavIndex];
                continue;
            }

            uint32_t bindCount = shaderBlob.boundUAVTextureCounts[uavIndex];

            uavCount += bindCount;

            if (shaderBlob.boundUAVTextures[uavIndex] > uint32_t(maxUavSlotIndex))
                maxUavSlotIndex = shaderBlob.boundUAVTextures[uavIndex] + (bindCount - 1);
        }

        if (uavCount > 0)
            uavCount = (maxUavSlotIndex + 1) > uavCount ? (maxUavSlotIndex + 1) : uavCount;

        int32_t srvCount              = 0;
        int32_t maxSrvSlotIndex       = 0;

        for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; srvIndex++)
        {
            // count static srvs separately.
            if (shaderBlob.boundSRVBufferSpaces[srvIndex] != 0)
            {
                if (staticBufferSrvCount > 0)
                    FFX_ASSERT(uint32_t(staticBufferSrvSpace) != shaderBlob.boundSRVBufferSpaces[srvIndex]);

                staticBufferSrvCount += shaderBlob.boundSRVBufferCounts[srvIndex];
                staticBufferSrvSpace = shaderBlob.boundSRVBufferSpaces[srvIndex];
                continue;
            }

            uint32_t bindCount = shaderBlob.boundSRVBufferCounts[srvIndex];

            srvCount += bindCount;

            if (shaderBlob.boundSRVBuffers[srvIndex] > uint32_t(maxSrvSlotIndex))
                maxSrvSlotIndex = shaderBlob.boundSRVBuffers[srvIndex] + (bindCount - 1);
        }

        for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; srvIndex++)
        {
            // count static srvs separately.
            if (shaderBlob.boundSRVTextureSpaces[srvIndex] != 0)
            {
                if (staticTextureSrvCount > 0)
                    FFX_ASSERT(uint32_t(staticTextureSrvSpace) != shaderBlob.boundSRVTextureSpaces[srvIndex]);

                staticTextureSrvCount += shaderBlob.boundSRVTextureCounts[srvIndex];
                staticTextureSrvSpace = shaderBlob.boundSRVTextureSpaces[srvIndex];
                continue;
            }

            uint32_t bindCount = shaderBlob.boundSRVTextureCounts[srvIndex];

            srvCount += bindCount;

            if (shaderBlob.boundSRVTextures[srvIndex] > uint32_t(maxSrvSlotIndex))
                maxSrvSlotIndex = shaderBlob.boundSRVTextures[srvIndex] + (bindCount - 1);
        }

        if (srvCount > 0)
            srvCount = (maxSrvSlotIndex + 1) > srvCount ? (maxSrvSlotIndex + 1) : srvCount;

        if (uavCount > 0) {

            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            dx12DescriptorRange->BaseShaderRegister = 0;
            dx12DescriptorRange->NumDescriptors = uavCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        if (srvCount > 0) {

            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            dx12DescriptorRange->BaseShaderRegister = 0;
            dx12DescriptorRange->NumDescriptors = srvCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        if (staticTextureSrvCount > 0)
        {
            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            dx12DescriptorRange->BaseShaderRegister                = 0;
            dx12DescriptorRange->RegisterSpace                     = staticTextureSrvSpace;
            dx12DescriptorRange->NumDescriptors                    = staticTextureSrvCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        if (staticBufferSrvCount > 0)
        {
            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            dx12DescriptorRange->BaseShaderRegister                = 0;
            dx12DescriptorRange->RegisterSpace                     = staticBufferSrvSpace;
            dx12DescriptorRange->NumDescriptors                    = staticBufferSrvCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        if (staticTextureUavCount > 0)
        {
            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            dx12DescriptorRange->BaseShaderRegister                = 0;
            dx12DescriptorRange->RegisterSpace                     = staticTextureUavSpace;
            dx12DescriptorRange->NumDescriptors                    = staticTextureUavCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        if (staticBufferUavCount > 0)
        {
            FFX_ASSERT(currentDescriptorRangeIndex < maximumDescriptorRangeSize);
            D3D12_DESCRIPTOR_RANGE* dx12DescriptorRange = &dx12Ranges[currentDescriptorRangeIndex];
            memset(dx12DescriptorRange, 0, sizeof(D3D12_DESCRIPTOR_RANGE));
            dx12DescriptorRange->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            dx12DescriptorRange->RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            dx12DescriptorRange->BaseShaderRegister                = 0;
            dx12DescriptorRange->RegisterSpace                     = staticBufferUavSpace;
            dx12DescriptorRange->NumDescriptors                    = staticBufferUavCount;
            currentDescriptorRangeIndex++;

            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* dx12RootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(dx12RootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            dx12RootParameterSlot->ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            dx12RootParameterSlot->DescriptorTable.NumDescriptorRanges = 1;
            currentRootParameterIndex++;
        }

        // Setup the descriptor table bindings for the above
        for (int32_t currentRangeIndex = 0; currentRangeIndex < currentDescriptorRangeIndex; currentRangeIndex++) {

            dx12RootParameters[currentRangeIndex].DescriptorTable.pDescriptorRanges = &dx12Ranges[currentRangeIndex];
        }

        for (int32_t currentRootConstantIndex = 0; currentRootConstantIndex < (int32_t)shaderBlob.cbvCount; currentRootConstantIndex++)
        {
            FFX_ASSERT(currentRootParameterIndex < maximumRootParameters);
            D3D12_ROOT_PARAMETER* rootParameterSlot = &dx12RootParameters[currentRootParameterIndex];
            memset(rootParameterSlot, 0, sizeof(D3D12_ROOT_PARAMETER));
            rootParameterSlot->ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameterSlot->Constants.ShaderRegister = shaderBlob.boundConstantBuffers[currentRootConstantIndex];
            currentRootParameterIndex++;
        }

        D3D12_ROOT_SIGNATURE_DESC dx12RootSignatureDescription = {};
        dx12RootSignatureDescription.NumParameters = currentRootParameterIndex;
        dx12RootSignatureDescription.pParameters = dx12RootParameters;
        dx12RootSignatureDescription.NumStaticSamplers = (UINT)samplerCount;
        dx12RootSignatureDescription.pStaticSamplers = dx12SamplerDescriptions;

        ID3DBlob* outBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;

        //Query D3D12SerializeRootSignature from d3d12.dll handle
        typedef HRESULT(__stdcall* D3D12SerializeRootSignatureType)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

        //Do not pass hD3D12 handle to the FreeLibrary function, as GetModuleHandle will not increment refcount
        HMODULE d3d12ModuleHandle = GetModuleHandleW(L"D3D12.dll");

        if (NULL != d3d12ModuleHandle) {

            D3D12SerializeRootSignatureType dx12SerializeRootSignatureType = (D3D12SerializeRootSignatureType)GetProcAddress(d3d12ModuleHandle, "D3D12SerializeRootSignature");

            if (nullptr != dx12SerializeRootSignatureType) {
                
                HRESULT result = dx12SerializeRootSignatureType(&dx12RootSignatureDescription, D3D_ROOT_SIGNATURE_VERSION_1, &outBlob, &errorBlob);
                if (errorBlob != nullptr) {

                    errorBlob->Release();
                } 
                if (FAILED(result)) {

                    if (outBlob != nullptr) {
                        
                        outBlob->Release();
                    }  
                    return FFX_ERROR_BACKEND_API_ERROR;
                }

                result = dx12Device->CreateRootSignature(0, outBlob->GetBufferPointer(), outBlob->GetBufferSize(), IID_PPV_ARGS(reinterpret_cast<ID3D12RootSignature**>(&outPipeline->rootSignature)));
                if (outBlob != nullptr) {
                    
                    outBlob->Release();
                }  
                if (FAILED(result)) {

                    return FFX_ERROR_BACKEND_API_ERROR;
                }
            } else {
                return FFX_ERROR_BACKEND_API_ERROR;
            }
        } else {
            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }

    ID3D12RootSignature* dx12RootSignature = reinterpret_cast<ID3D12RootSignature*>(outPipeline->rootSignature);

    // Only set the command signature if this is setup as an indirect workload
    if (pipelineDescription->indirectWorkload)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argumentDescs = { D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH };
        D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
        commandSignatureDesc.pArgumentDescs = &argumentDescs;
        commandSignatureDesc.NumArgumentDescs = 1;
        commandSignatureDesc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);

        HRESULT result = dx12Device->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(reinterpret_cast<ID3D12CommandSignature**>(&outPipeline->cmdSignature)));
        if (FAILED(result)) {

            return FFX_ERROR_BACKEND_API_ERROR;
        }
    }
    else
    {
        outPipeline->cmdSignature = nullptr;
    }

    uint32_t flattenedSrvTextureCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvTextureCount; ++srvIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundSRVTextures[srvIndex];
        uint32_t spaceIndex = shaderBlob.boundSRVTextureSpaces[srvIndex];
        uint32_t bindCount = shaderBlob.boundSRVTextureCounts[srvIndex];

        // Skip static resources
        if (spaceIndex == uint32_t(staticTextureSrvSpace))
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvTextureCount++;

            outPipeline->srvTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvTextureBindings[bindingIndex].name, shaderBlob.boundSRVTextureNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvTextureCount = flattenedSrvTextureCount;
    FFX_ASSERT(outPipeline->srvTextureCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavTextureCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavTextureCount; ++uavIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundUAVTextures[uavIndex];
        uint32_t spaceIndex = shaderBlob.boundUAVTextureSpaces[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVTextureCounts[uavIndex];

        // Skip static resources
        if (spaceIndex == uint32_t(staticTextureUavSpace))
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavTextureCount++;

            outPipeline->uavTextureBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavTextureBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavTextureBindings[bindingIndex].name, shaderBlob.boundUAVTextureNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavTextureBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavTextureCount = flattenedUavTextureCount;
    FFX_ASSERT(outPipeline->uavTextureCount < FFX_MAX_NUM_UAVS);

    uint32_t flattenedSrvBufferCount = 0;

    for (uint32_t srvIndex = 0; srvIndex < shaderBlob.srvBufferCount; ++srvIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundSRVBuffers[srvIndex];
        uint32_t spaceIndex = shaderBlob.boundSRVBufferSpaces[srvIndex];
        uint32_t bindCount  = shaderBlob.boundSRVBufferCounts[srvIndex];

        // Skip static resources
        if (spaceIndex == uint32_t(staticBufferSrvSpace))
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedSrvBufferCount++;

            outPipeline->srvBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->srvBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->srvBufferBindings[bindingIndex].name, shaderBlob.boundSRVBufferNames[srvIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->srvBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->srvBufferCount = flattenedSrvBufferCount;
    FFX_ASSERT(outPipeline->srvBufferCount < FFX_MAX_NUM_SRVS);

    uint32_t flattenedUavBufferCount = 0;

    for (uint32_t uavIndex = 0; uavIndex < shaderBlob.uavBufferCount; ++uavIndex)
    {
        uint32_t slotIndex  = shaderBlob.boundUAVBuffers[uavIndex];
        uint32_t spaceIndex = shaderBlob.boundUAVBufferSpaces[uavIndex];
        uint32_t bindCount = shaderBlob.boundUAVBufferCounts[uavIndex];

        // Skip static resources
        if (spaceIndex == uint32_t(staticBufferUavSpace))
            continue;

        for (uint32_t arrayIndex = 0; arrayIndex < bindCount; arrayIndex++)
        {
            uint32_t bindingIndex = flattenedUavBufferCount++;

            outPipeline->uavBufferBindings[bindingIndex].slotIndex  = slotIndex;
            outPipeline->uavBufferBindings[bindingIndex].arrayIndex = arrayIndex;
            strncpy(outPipeline->uavBufferBindings[bindingIndex].name, shaderBlob.boundUAVBufferNames[uavIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->uavBufferBindings[bindingIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
        }
    }

    outPipeline->uavBufferCount = flattenedUavBufferCount;
    FFX_ASSERT(outPipeline->uavBufferCount < FFX_MAX_NUM_UAVS);

    for (uint32_t cbIndex = 0; cbIndex < shaderBlob.cbvCount; ++cbIndex)
    {
        outPipeline->constantBufferBindings[cbIndex].slotIndex  = shaderBlob.boundConstantBuffers[cbIndex];
        outPipeline->constantBufferBindings[cbIndex].arrayIndex = 1;
        strncpy(outPipeline->constantBufferBindings[cbIndex].name, shaderBlob.boundConstantBufferNames[cbIndex], FFX_RESOURCE_NAME_SIZE - 1);
            outPipeline->constantBufferBindings[cbIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
    }

    outPipeline->constCount = shaderBlob.cbvCount;
    FFX_ASSERT(outPipeline->constCount < FFX_MAX_NUM_CONST_BUFFERS);

    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    outPipeline->staticTextureSrvCount = staticTextureSrvCount;
    FFX_ASSERT(outPipeline->staticTextureSrvCount <= effectContext.bindlessTextureSrvHeapSize);

    outPipeline->staticBufferSrvCount = staticBufferSrvCount;
    FFX_ASSERT(outPipeline->staticBufferSrvCount <= effectContext.bindlessBufferSrvHeapSize);

    outPipeline->staticTextureUavCount = staticTextureUavCount;
    FFX_ASSERT(outPipeline->staticTextureUavCount <= effectContext.bindlessTextureUavHeapSize);

    outPipeline->staticBufferUavCount = staticBufferUavCount;
    FFX_ASSERT(outPipeline->staticBufferUavCount <= effectContext.bindlessBufferUavHeapSize);

    // Todo when needed
    //outPipeline->samplerCount      = shaderBlob.samplerCount;
    //outPipeline->rtAccelStructCount= shaderBlob.rtAccelStructCount;
    // create the PSO
    if (graphics)
    {
        /*
         * No PSO here, deliberately.
         *
         * A D3D12 graphics pipeline bakes its render-target formats, and this function runs
         * before any job has been scheduled -- the targets it will write are not known yet.
         * The Vulkan backend has the same problem and solves it the same way: its
         * CreateGraphicsPipelineVK builds only the layout, and getOrCreateGraphicsPipeline
         * creates the PSO from inside the job once the render pass (and so the formats)
         * exist.
         *
         * So the blobs are parked in a record, whose address goes into outPipeline->
         * rootSignature -- the same recovery route DataGraphPipelineDX12 uses -- and
         * executeGpuJobFragmentDX12 builds and caches the PSO on first execution.
         */
        FFX_ASSERT_MESSAGE(backendContext->nextGraphicsPipeline < backendContext->maxEffectContexts * FFX_MAX_PASS_COUNT,
                           "FFXInterface: DX12: ran out of graphics pipeline records. Please increase FFX_MAX_PASS_COUNT");

        BackendContext_DX12::GraphicsPipelineDX12* pRecord =
            &backendContext->pGraphicsPipelines[backendContext->nextGraphicsPipeline++];

        pRecord->vertBlob         = vertShaderBlob.data;
        pRecord->vertBlobSize     = vertShaderBlob.size;
        pRecord->pixelBlob        = shaderBlob.data;
        pRecord->pixelBlobSize    = shaderBlob.size;
        pRecord->effectContextId  = effectContextId;
        pRecord->rootSignature    = dx12RootSignature;
        pRecord->cachedPso        = nullptr;
        pRecord->cachedFormatHash = 0;
        pRecord->hasCachedPso     = false;

        outPipeline->rootSignature = reinterpret_cast<FfxRootSignature>(pRecord);
        outPipeline->pipeline      = nullptr;
    }
    else
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC dx12PipelineStateDescription = {};
        dx12PipelineStateDescription.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        dx12PipelineStateDescription.pRootSignature = dx12RootSignature;
        dx12PipelineStateDescription.CS.pShaderBytecode = shaderBlob.data;
        dx12PipelineStateDescription.CS.BytecodeLength = shaderBlob.size;

        if (FAILED(dx12Device->CreateComputePipelineState(&dx12PipelineStateDescription, IID_PPV_ARGS(reinterpret_cast<ID3D12PipelineState**>(&outPipeline->pipeline)))))
            return FFX_ERROR_BACKEND_API_ERROR;

        // Set the pipeline name
        { wchar_t ffxWideName[FFX_RESOURCE_NAME_SIZE];
        ffxNarrowToWide(ffxWideName, std::size(ffxWideName), pipelineDescription->name);
        reinterpret_cast<ID3D12PipelineState*>(outPipeline->pipeline)->SetName(ffxWideName); }
    }

    strncpy(outPipeline->name, pipelineDescription->name, FFX_RESOURCE_NAME_SIZE - 1);
    outPipeline->name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    return FFX_OK;
}

FfxErrorCode CreatePipelineDX12(
    FfxInterface* backendInterface,
    FfxEffect effect,
    FfxPass pass,
    uint32_t permutationOptions,
    const FfxPipelineDescription* pipelineDescription,
    FfxUInt32                     effectContextId,
    FfxPipelineState* outPipeline)
{
    return CreatePipelineInternal(backendInterface, effect, pass, permutationOptions, pipelineDescription,
                                  effectContextId, outPipeline, /*graphics=*/false);
}

FfxErrorCode DestroyPipelineDX12(
    FfxInterface* backendInterface,
    FfxPipelineState* pipeline,
    FfxUInt32)
{
    FFX_ASSERT(backendInterface != nullptr);
    if (!pipeline) {
        return FFX_OK;
    }

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    /*
     * Two kinds of pipeline do not put an ID3D12RootSignature in
     * FfxPipelineState::rootSignature. Both put the address of a backend-side record there
     * instead, because neither can be represented by a D3D12 pipeline object at all:
     *
     *   - Graphics pipelines, because a graphics PSO cannot be built until a job supplies
     *     render targets, so something has to outlive the create call.
     *   - Data-graph pipelines, which have no ID3D12PipelineState in any form: the dp4a
     *     host owns its own pipelines, root signature and barriers and records them
     *     itself.
     *
     * That field is therefore NOT a COM pointer for either kind, and releasing it is an
     * access violation.
     *
     * This is not hypothetical. The frame interpolation component releases a pipeline
     * before every create (ffxSafeReleasePipeline at the top of its CreatePipeline lambda
     * in ffx_frameinterpolation.cpp). The graphics case is what the execution harness hit
     * first. The data-graph case is reached on the next createPipelineStates, because by
     * then the NFRU pipeline is holding a data-graph record.
     *
     * Each record is identified by its address lying inside the array the backend reserved
     * for it, which is exact rather than a guess about which fields are null.
     */
    {
        const uintptr_t rootSig = reinterpret_cast<uintptr_t>(pipeline->rootSignature);

        const uintptr_t gfxBegin = reinterpret_cast<uintptr_t>(backendContext->pGraphicsPipelines);
        const uintptr_t gfxEnd   = gfxBegin + static_cast<uintptr_t>(backendContext->maxEffectContexts) * FFX_MAX_PASS_COUNT
                                               * sizeof(BackendContext_DX12::GraphicsPipelineDX12);

        if (backendContext->pGraphicsPipelines != nullptr && rootSig >= gfxBegin && rootSig < gfxEnd)
        {
            BackendContext_DX12::GraphicsPipelineDX12* pRecord =
                reinterpret_cast<BackendContext_DX12::GraphicsPipelineDX12*>(pipeline->rootSignature);

            if (pRecord->cachedPso != nullptr)
            {
                pRecord->cachedPso->Release();
                pRecord->cachedPso = nullptr;
            }
            if (pRecord->rootSignature != nullptr)
            {
                pRecord->rootSignature->Release();
                pRecord->rootSignature = nullptr;
            }
            pRecord->hasCachedPso     = false;
            pRecord->cachedFormatHash = 0;

            pipeline->rootSignature = nullptr;
            pipeline->pipeline      = nullptr;
            return FFX_OK;
        }

        const uintptr_t dgBegin = reinterpret_cast<uintptr_t>(backendContext->pDataGraphPipelines);
        const uintptr_t dgEnd   = dgBegin + static_cast<uintptr_t>(backendContext->maxEffectContexts) * FFX_MAX_PASS_COUNT
                                              * sizeof(BackendContext_DX12::DataGraphPipelineDX12);

        if (backendContext->pDataGraphPipelines != nullptr && rootSig >= dgBegin && rootSig < dgEnd)
        {
            BackendContext_DX12::DataGraphPipelineDX12* pRecord =
                reinterpret_cast<BackendContext_DX12::DataGraphPipelineDX12*>(pipeline->rootSignature);

            /*
             * dp4a is a void* holding one of two unrelated types, discriminated by the
             * entry point the effect asked for. Releasing it here is not optional: nothing
             * else in this backend calls either destructor, so every data-graph pipeline
             * created would otherwise leak its device, weights and scratch memory.
             */
            if (pRecord->dp4a != nullptr)
            {
                if (strcmp(pRecord->entryPoint, "nfru_v1_int8") == 0)
                {
                    nfruDp4aDestroyContext(reinterpret_cast<NfruDp4aContext*>(pRecord->dp4a));
                }
                else
                {
                    NssDx12Destroy(reinterpret_cast<NssDx12*>(pRecord->dp4a));
                }
                pRecord->dp4a = nullptr;
            }
            pRecord->entryPoint[0] = '\0';
            pRecord->width         = 0;
            pRecord->height        = 0;

            pipeline->rootSignature = nullptr;
            pipeline->pipeline      = nullptr;
            return FFX_OK;
        }
    }

    // destroy Rootsignature
    ID3D12RootSignature* dx12RootSignature = reinterpret_cast<ID3D12RootSignature*>(pipeline->rootSignature);
    if (dx12RootSignature) {
        dx12RootSignature->Release();
    }
    pipeline->rootSignature = nullptr;

    // destroy CmdSignature
    ID3D12CommandSignature* dx12CmdSignature = reinterpret_cast<ID3D12CommandSignature*>(pipeline->cmdSignature);
    if (dx12CmdSignature) {
        dx12CmdSignature->Release();
    }
    pipeline->cmdSignature = nullptr;

    // destroy pipeline
    ID3D12PipelineState* dx12Pipeline = reinterpret_cast<ID3D12PipelineState*>(pipeline->pipeline);
    if (dx12Pipeline) {
        dx12Pipeline->Release();
    }
    pipeline->pipeline = nullptr;

    return FFX_OK;
}

FfxErrorCode ScheduleGpuJobDX12(
    FfxInterface* backendInterface,
    const FfxGpuJobDescription* job
)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != job);

    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    FFX_ASSERT(backendContext->gpuJobCount < FFX_MAX_GPU_JOBS);

    backendContext->pGpuJobs[backendContext->gpuJobCount] = *job;
    backendContext->gpuJobCount++;

    return FFX_OK;
}

static FfxErrorCode executeGpuJobCompute(BackendContext_DX12*       backendContext,
                                         FfxGpuJobDescription*      job,
                                         ID3D12GraphicsCommandList* dx12CommandList,
                                         FfxUInt32                  effectContextId)
{
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendContext->device);
 
    // Set descriptor head for binding
    ID3D12DescriptorHeap* dx12DescriptorHeap = reinterpret_cast<ID3D12DescriptorHeap*>(backendContext->descRingBuffer);

    // set root signature
    ID3D12RootSignature* dx12RootSignature = reinterpret_cast<ID3D12RootSignature*>(job->computeJobDescriptor.pipeline.rootSignature);
    dx12CommandList->SetComputeRootSignature(dx12RootSignature);

    // set descriptor heap
    dx12CommandList->SetDescriptorHeaps(1, &dx12DescriptorHeap);

    uint32_t descriptorTableIndex = 0;

    // bind texture & buffer UAVs (note the binding order here MUST match the root signature mapping order from CreatePipeline!)
    {
        // Set a baseline minimal value
        uint32_t maximumUavIndex = job->computeJobDescriptor.pipeline.uavTextureCount + job->computeJobDescriptor.pipeline.uavBufferCount;

        for (uint32_t uavTextureBinding = 0; uavTextureBinding < job->computeJobDescriptor.pipeline.uavTextureCount; uavTextureBinding++)
        {
            uint32_t slotIndex = job->computeJobDescriptor.pipeline.uavTextureBindings[uavTextureBinding].slotIndex +
                                 job->computeJobDescriptor.pipeline.uavTextureBindings[uavTextureBinding].arrayIndex;

            if (slotIndex > maximumUavIndex)
                maximumUavIndex = slotIndex;
        }

        for (uint32_t uavBufferBinding = 0; uavBufferBinding < job->computeJobDescriptor.pipeline.uavBufferCount; uavBufferBinding++)
        {
            uint32_t slotIndex = job->computeJobDescriptor.pipeline.uavBufferBindings[uavBufferBinding].slotIndex +
                                 job->computeJobDescriptor.pipeline.uavTextureBindings[uavBufferBinding].arrayIndex;

            if (slotIndex > maximumUavIndex)
                maximumUavIndex = slotIndex;
        }

        if (maximumUavIndex)
        {
            // check if this fits into the ringbuffer, loop if not fitting
            if (backendContext->descRingBufferBase + maximumUavIndex + 1 > FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts)
                backendContext->descRingBufferBase = 0;

            D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            gpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            // Set Texture UAVs
            for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->computeJobDescriptor.pipeline.uavTextureCount; ++currentPipelineUavIndex) {

                addBarrier(backendContext, &job->computeJobDescriptor.uavTextures[currentPipelineUavIndex].resource, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

                const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.uavTextureBindings[currentPipelineUavIndex];

                // source: UAV of resource to bind
                const uint32_t resourceIndex = job->computeJobDescriptor.uavTextures[currentPipelineUavIndex].resource.internalIndex;
                const uint32_t uavIndex = backendContext->pResources[resourceIndex].uavDescIndex + job->computeJobDescriptor.uavTextures[currentPipelineUavIndex].mip;

                // where to bind it
                const uint32_t currentUavResourceIndex = binding.slotIndex + binding.arrayIndex;

                D3D12_CPU_DESCRIPTOR_HANDLE srcHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                srcHandle.ptr += uavIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
                cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                cpuView.ptr += currentUavResourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                // Copy descriptor
                dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }

            // Set Buffer UAVs
            for (uint32_t currentPipelineUavIndex = 0; currentPipelineUavIndex < job->computeJobDescriptor.pipeline.uavBufferCount; ++currentPipelineUavIndex) {
                
                // continue if this is a null resource.
                if (job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].resource.internalIndex == 0)
                    continue;

                addBarrier(backendContext, &job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].resource, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

                const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.uavBufferBindings[currentPipelineUavIndex];

                // where to bind it
                const uint32_t currentUavResourceIndex = binding.slotIndex + binding.arrayIndex;

                D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
                cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                cpuView.ptr += currentUavResourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                if (job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].size > 0)
                {
                    // if size is non-zero create a dynamic descriptor directly on the GPU heap
                    ID3D12Resource* buffer = getDX12ResourcePtr(backendContext, job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].resource.internalIndex);
                    FFX_ASSERT(buffer != NULL);

                    D3D12_UNORDERED_ACCESS_VIEW_DESC dx12UavDescription = {};

                    bool     isStructured = job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].stride > 0;
                    uint32_t stride       = isStructured ? job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].stride : sizeof(uint32_t);

                    dx12UavDescription.Format                      = isStructured ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_TYPELESS;
                    dx12UavDescription.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
                    dx12UavDescription.Buffer.FirstElement         = job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].offset / stride;
                    dx12UavDescription.Buffer.NumElements          = job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].size / stride;
                    dx12UavDescription.Buffer.StructureByteStride  = isStructured ? stride : 0;
                    dx12UavDescription.Buffer.CounterOffsetInBytes = 0;
                    dx12UavDescription.Buffer.Flags                = isStructured ? D3D12_BUFFER_UAV_FLAG_NONE : D3D12_BUFFER_UAV_FLAG_RAW;

                    dx12Device->CreateUnorderedAccessView(buffer, 0, &dx12UavDescription, cpuView);
                }
                else
                {
                    // if size is zero assume it is a static descriptor and copy it from the CPU heap
                    const uint32_t resourceIndex = job->computeJobDescriptor.uavBuffers[currentPipelineUavIndex].resource.internalIndex;
          
                    const uint32_t uavIndex  = backendContext->pResources[resourceIndex].uavDescIndex;

                    D3D12_CPU_DESCRIPTOR_HANDLE srcHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
                    srcHandle.ptr += uavIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                    // Copy descriptor
                    dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                }
            }

            backendContext->descRingBufferBase += maximumUavIndex + 1;
            dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
        }
    }

    // bind texture & buffer SRVs
    {
        // Set a baseline minimal value
        uint32_t maximumSrvIndex = job->computeJobDescriptor.pipeline.srvTextureCount + job->computeJobDescriptor.pipeline.srvBufferCount;

        for (uint32_t srvTextureBinding = 0; srvTextureBinding < job->computeJobDescriptor.pipeline.srvTextureCount; srvTextureBinding++)
        {
            uint32_t slotIndex = job->computeJobDescriptor.pipeline.srvTextureBindings[srvTextureBinding].slotIndex +
                                 job->computeJobDescriptor.pipeline.srvTextureBindings[srvTextureBinding].arrayIndex;

            if (slotIndex > maximumSrvIndex)
                maximumSrvIndex = slotIndex;
        }

        for (uint32_t srvBufferBinding = 0; srvBufferBinding < job->computeJobDescriptor.pipeline.srvBufferCount; srvBufferBinding++)
        {
            uint32_t slotIndex = job->computeJobDescriptor.pipeline.srvBufferBindings[srvBufferBinding].slotIndex +
                                 job->computeJobDescriptor.pipeline.srvTextureBindings[srvBufferBinding].arrayIndex;

            if (slotIndex > maximumSrvIndex)
                maximumSrvIndex = slotIndex;
        }

        if (maximumSrvIndex)
        {
            // check if this fits into the ringbuffer, loop if not fitting
            if (backendContext->descRingBufferBase + maximumSrvIndex + 1 > FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts)
            {
                backendContext->descRingBufferBase = 0;
            }

            D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            gpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->computeJobDescriptor.pipeline.srvTextureCount; ++currentPipelineSrvIndex)
            {
                if (job->computeJobDescriptor.srvTextures[currentPipelineSrvIndex].resource.internalIndex == 0)
                    break;

                addBarrier(backendContext, &job->computeJobDescriptor.srvTextures[currentPipelineSrvIndex].resource, FFX_RESOURCE_STATE_COMPUTE_READ);

                const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.srvTextureBindings[currentPipelineSrvIndex];

                // source: SRV of resource to bind
                const uint32_t              resourceIndex = job->computeJobDescriptor.srvTextures[currentPipelineSrvIndex].resource.internalIndex;
                D3D12_CPU_DESCRIPTOR_HANDLE srcHandle     = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
                srcHandle.ptr += resourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                // Where to bind it
                uint32_t currentSrvResourceIndex = binding.slotIndex + binding.arrayIndex;

                D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
                cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                cpuView.ptr += currentSrvResourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }

            // Set Buffer SRVs
            for (uint32_t currentPipelineSrvIndex = 0; currentPipelineSrvIndex < job->computeJobDescriptor.pipeline.srvBufferCount; ++currentPipelineSrvIndex)
            {
                // continue if this is a null resource.
                if (job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource.internalIndex == 0)
                    continue;

                addBarrier(backendContext, &job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource, FFX_RESOURCE_STATE_COMPUTE_READ);

                const FfxResourceBinding binding = job->computeJobDescriptor.pipeline.srvBufferBindings[currentPipelineSrvIndex];

                // where to bind it
                const uint32_t currentSrvResourceIndex = binding.slotIndex + binding.arrayIndex;

                D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
                cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                cpuView.ptr += currentSrvResourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                if (job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].size > 0)
                {
                    // if size is non-zero create a dynamic descriptor directly on the GPU heap
                    ID3D12Resource* buffer = getDX12ResourcePtr(backendContext, job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource.internalIndex);
                    FFX_ASSERT(buffer != NULL);

                    D3D12_SHADER_RESOURCE_VIEW_DESC dx12SrvDescription = {};

                    bool     isStructured = job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].stride > 0;
                    uint32_t stride       = isStructured ? job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].stride : sizeof(uint32_t);

                    dx12SrvDescription.Format                     = isStructured ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_TYPELESS;
                    dx12SrvDescription.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                    dx12SrvDescription.Buffer.FirstElement        = job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].offset / stride;
                    dx12SrvDescription.Buffer.NumElements         = job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].size / stride;
                    dx12SrvDescription.Buffer.StructureByteStride = isStructured ? stride : 0;
                    dx12SrvDescription.Buffer.Flags               = isStructured ? D3D12_BUFFER_SRV_FLAG_NONE : D3D12_BUFFER_SRV_FLAG_RAW;
                    dx12SrvDescription.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

                    dx12Device->CreateShaderResourceView(buffer, &dx12SrvDescription, cpuView);
                }
                else
                {
                    // source: SRV of buffer to bind
                    const uint32_t resourceIndex = job->computeJobDescriptor.srvBuffers[currentPipelineSrvIndex].resource.internalIndex;

                    D3D12_CPU_DESCRIPTOR_HANDLE srcHandle = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
                    srcHandle.ptr += resourceIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                    // Copy descriptor
                    dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                }
            }

            backendContext->descRingBufferBase += maximumSrvIndex + 1;
            dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
        }
    }

    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    // bind static texture srv table
    if (job->computeJobDescriptor.pipeline.staticTextureSrvCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessTextureSrvHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    // bind static buffer srv table
    if (job->computeJobDescriptor.pipeline.staticBufferSrvCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessBufferSrvHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    // bind static texture uav table
    if (job->computeJobDescriptor.pipeline.staticTextureUavCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessTextureUavHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    // bind static buffer uav table
    if (job->computeJobDescriptor.pipeline.staticBufferUavCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessBufferUavHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        dx12CommandList->SetComputeRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    // If we are dispatching indirectly, transition the argument resource to indirect argument
    if (job->computeJobDescriptor.pipeline.cmdSignature)
    {
        addBarrier(backendContext, &job->computeJobDescriptor.cmdArgument, FFX_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    flushBarriers(backendContext, dx12CommandList);

    // bind pipeline
    ID3D12PipelineState* dx12PipelineStateObject = reinterpret_cast<ID3D12PipelineState*>(job->computeJobDescriptor.pipeline.pipeline);
    dx12CommandList->SetPipelineState(dx12PipelineStateObject);

    // copy data to constant buffer and bind
    {
        for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < job->computeJobDescriptor.pipeline.constCount; ++currentRootConstantIndex) {

            // If we have a constant buffer allocator, use that, otherwise use the default backend allocator
            FfxConstantAllocation allocation;
            if (s_fpConstantAllocator)
            {
                allocation = s_fpConstantAllocator(job->computeJobDescriptor.cbs[currentRootConstantIndex].data, job->computeJobDescriptor.cbs[currentRootConstantIndex].num32BitEntries * sizeof(uint32_t));
            }
            else
            {
                allocation = backendContext->FallbackConstantAllocator(job->computeJobDescriptor.cbs[currentRootConstantIndex].data, job->computeJobDescriptor.cbs[currentRootConstantIndex].num32BitEntries * sizeof(uint32_t));
            }

            D3D12_GPU_VIRTUAL_ADDRESS bufferViewDesc = D3D12_GPU_VIRTUAL_ADDRESS(allocation.handle);
            dx12CommandList->SetComputeRootConstantBufferView(descriptorTableIndex + currentRootConstantIndex, bufferViewDesc);
        }
    }

    // Dispatch (or dispatch indirect)
    if (job->computeJobDescriptor.pipeline.cmdSignature)
    {
        const uint32_t resourceIndex = job->computeJobDescriptor.cmdArgument.internalIndex;
        ID3D12Resource* pResource = backendContext->pResources[resourceIndex].resourcePtr;

        dx12CommandList->ExecuteIndirect(reinterpret_cast<ID3D12CommandSignature*>(job->computeJobDescriptor.pipeline.cmdSignature), 1, pResource, job->computeJobDescriptor.cmdArgumentOffset, nullptr, 0);
    }
    else
    {
        dx12CommandList->Dispatch(job->computeJobDescriptor.dimensions[0], job->computeJobDescriptor.dimensions[1], job->computeJobDescriptor.dimensions[2]);
    }

    return FFX_OK;
}

/*
 * =============================================================================
 * Fragment jobs -- the D3D12 half of executeGpuJobFragment in ffx_vk.cpp.
 * =============================================================================
 * Frame interpolation's eight fragment passes each draw one full-screen triangle into a
 * set of render targets. The Vulkan backend handles them with a render pass built from the
 * job's render targets and a PSO created lazily inside getOrCreateGraphicsPipeline; the
 * structure here is the same, for the same reason -- a D3D12 graphics PSO bakes its
 * render-target formats, and those only exist once the job runs.
 */
static FfxErrorCode executeGpuJobFragmentDX12(BackendContext_DX12*       backendContext,
                                              FfxGpuJobDescription*      job,
                                              ID3D12GraphicsCommandList* dx12CommandList,
                                              FfxUInt32                  effectContextId)
{
    ID3D12Device* dx12Device = backendContext->device;

    FfxPipelineState& pipeline = job->fragmentJobDescriptor.pipeline;

    BackendContext_DX12::GraphicsPipelineDX12* pRecord =
        reinterpret_cast<BackendContext_DX12::GraphicsPipelineDX12*>(pipeline.rootSignature);
    if (pRecord == nullptr)
    {
        FFX_ASSERT_MESSAGE(false, "FFXInterface: DX12: fragment job with no graphics pipeline record");
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    // -------------------------------------------------------------------------
    // Gather the render targets.
    //
    // The count and the formats come from the bound resources rather than from
    // pipeline.rtCount. They have to: a D3D12 graphics PSO bakes RTVFormats, and the
    // shader compiler's reflection cannot supply them here -- this fork's FfxShaderBlob
    // carries rtTexture* members that only its own (GLSL-only) compiler ever emitted, so
    // the accessor fills them with zero. Reading the resources is both available and
    // exact, which is what the Vulkan render pass does too.
    // -------------------------------------------------------------------------
    uint32_t                  rtCount   = 0;
    DXGI_FORMAT               rtFormats[FFX_MAX_NUM_RTS] = {};
    ID3D12Resource*           rtResources[FFX_MAX_NUM_RTS] = {};
    D3D12_CPU_DESCRIPTOR_HANDLE rtHandles[FFX_MAX_NUM_RTS] = {};

    for (uint32_t rt = 0; rt < FFX_MAX_NUM_RTS; ++rt)
    {
        const uint32_t internalIndex = job->fragmentJobDescriptor.rtTextures[rt].resource.internalIndex;
        if (internalIndex == 0)
        {
            break;
        }
        ID3D12Resource* pResource = getDX12ResourcePtr(backendContext, internalIndex);
        if (pResource == nullptr)
        {
            break;
        }
        rtResources[rt] = pResource;
        rtFormats[rt]   = pResource->GetDesc().Format;
        rtCount++;
    }

    if (rtCount == 0)
    {
        FFX_ASSERT_MESSAGE(false, "FFXInterface: DX12: fragment job with no render targets");
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    // Hash the format set so a PSO built for one set is not reused for another. The
    // Vulkan backend keys its cached pipeline on the render pass handle for the same
    // reason.
    uint64_t formatHash = 1469598103934665603ull;  // FNV-1a
    for (uint32_t rt = 0; rt < rtCount; ++rt)
    {
        formatHash = (formatHash ^ static_cast<uint64_t>(rtFormats[rt])) * 1099511628211ull;
    }
    formatHash = (formatHash ^ static_cast<uint64_t>(rtCount)) * 1099511628211ull;

    // -------------------------------------------------------------------------
    // Build the PSO on first use, or when the format set changes.
    // -------------------------------------------------------------------------
    if (!pRecord->hasCachedPso || pRecord->cachedFormatHash != formatHash)
    {
        if (pRecord->cachedPso != nullptr)
        {
            pRecord->cachedPso->Release();
            pRecord->cachedPso = nullptr;
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature        = pRecord->rootSignature;
        psoDesc.VS.pShaderBytecode    = pRecord->vertBlob;
        psoDesc.VS.BytecodeLength     = pRecord->vertBlobSize;
        psoDesc.PS.pShaderBytecode    = pRecord->pixelBlob;
        psoDesc.PS.BytecodeLength     = pRecord->pixelBlobSize;

        // No vertex input, no stream output, no tessellation: the vertex shader builds a
        // full-screen triangle from SV_VertexID, which is what the Vulkan backend does
        // too (vkCmdDraw(vkCommandBuffer, 3, 1, 0, 0) with no vertex bindings).
        psoDesc.InputLayout           = {nullptr, 0};
        psoDesc.IBStripCutValue       = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.StreamOutput          = {};

        // Rasteriser: solid fill, back-face cull, clockwise front face -- verbatim the
        // Vulkan settings (VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
        // VK_FRONT_FACE_CLOCKWISE).
        psoDesc.RasterizerState.FillMode              = D3D12_FILL_MODE_SOLID;
        psoDesc.RasterizerState.CullMode              = D3D12_CULL_MODE_BACK;
        psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
        psoDesc.RasterizerState.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
        psoDesc.RasterizerState.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        psoDesc.RasterizerState.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        psoDesc.RasterizerState.DepthClipEnable       = TRUE;
        psoDesc.RasterizerState.MultisampleEnable     = FALSE;
        psoDesc.RasterizerState.AntialiasedLineEnable = FALSE;
        psoDesc.RasterizerState.ForcedSampleCount     = 0;
        psoDesc.RasterizerState.ConservativeRaster    = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

        // Blend disabled with a full write mask on every target, matching the Vulkan
        // color blend attachments (blendEnable = VK_FALSE, all four components written).
        psoDesc.BlendState.AlphaToCoverageEnable  = FALSE;
        psoDesc.BlendState.IndependentBlendEnable = FALSE;
        for (uint32_t rt = 0; rt < FFX_MAX_NUM_RTS; ++rt)
        {
            psoDesc.BlendState.RenderTarget[rt].BlendEnable           = FALSE;
            psoDesc.BlendState.RenderTarget[rt].LogicOpEnable         = FALSE;
            psoDesc.BlendState.RenderTarget[rt].SrcBlend              = D3D12_BLEND_ONE;
            psoDesc.BlendState.RenderTarget[rt].DestBlend             = D3D12_BLEND_ZERO;
            psoDesc.BlendState.RenderTarget[rt].BlendOp               = D3D12_BLEND_OP_ADD;
            psoDesc.BlendState.RenderTarget[rt].SrcBlendAlpha         = D3D12_BLEND_ONE;
            psoDesc.BlendState.RenderTarget[rt].DestBlendAlpha        = D3D12_BLEND_ZERO;
            psoDesc.BlendState.RenderTarget[rt].BlendOpAlpha          = D3D12_BLEND_OP_ADD;
            psoDesc.BlendState.RenderTarget[rt].LogicOp               = D3D12_LOGIC_OP_NOOP;
            psoDesc.BlendState.RenderTarget[rt].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        }

        // No depth or stencil: the Vulkan render pass has a single colour attachment and
        // no depth attachment.
        psoDesc.DepthStencilState.DepthEnable    = FALSE;
        psoDesc.DepthStencilState.StencilEnable  = FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_ALWAYS;

        psoDesc.SampleMask            = UINT_MAX;
        psoDesc.NodeMask              = 0;
        psoDesc.Flags                 = D3D12_PIPELINE_STATE_FLAG_NONE;
        psoDesc.SampleDesc.Count      = 1;
        psoDesc.SampleDesc.Quality    = 0;
        psoDesc.DSVFormat             = DXGI_FORMAT_UNKNOWN;
        psoDesc.NumRenderTargets      = rtCount;
        for (uint32_t rt = 0; rt < rtCount; ++rt)
        {
            psoDesc.RTVFormats[rt] = rtFormats[rt];
        }

        if (FAILED(dx12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pRecord->cachedPso))))
        {
            FFX_ASSERT_MESSAGE(false, "FFXInterface: DX12: CreateGraphicsPipelineState failed "
                                      "(the render-target formats the job bound may not be supported "
                                      "as RTVs by this device)");
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        pRecord->cachedFormatHash = formatHash;
        pRecord->hasCachedPso     = true;
    }

    // -------------------------------------------------------------------------
    // Bind. The order here MUST match the root signature parameter order built in
    // CreatePipelineInternal -- UAV table, SRV table, static tables, then root CBVs --
    // which is the same order the compute path uses.
    // -------------------------------------------------------------------------
    ID3D12DescriptorHeap* dx12DescriptorHeap = backendContext->descRingBuffer;
    dx12CommandList->SetGraphicsRootSignature(pRecord->rootSignature);
    dx12CommandList->SetDescriptorHeaps(1, &dx12DescriptorHeap);

    uint32_t descriptorTableIndex = 0;

    const uint32_t maximumUavIndex = pipeline.uavTextureCount + pipeline.uavBufferCount;
    if (maximumUavIndex > 0)
    {
        if (backendContext->descRingBufferBase + maximumUavIndex + 1 > FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts)
        {
            backendContext->descRingBufferBase = 0;
        }
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        for (uint32_t i = 0; i < pipeline.uavTextureCount; ++i)
        {
            if (job->fragmentJobDescriptor.uavTextures[i].resource.internalIndex == 0)
            {
                continue;
            }
            addBarrier(backendContext, &job->fragmentJobDescriptor.uavTextures[i].resource, FFX_RESOURCE_STATE_PIXEL_UAV);

            const FfxResourceBinding binding    = pipeline.uavTextureBindings[i];
            const uint32_t           resIndex   = job->fragmentJobDescriptor.uavTextures[i].resource.internalIndex;
            const uint32_t           uavIndex   = backendContext->pResources[resIndex].uavDescIndex + job->fragmentJobDescriptor.uavTextures[i].mip;
            const uint32_t           dstIndex   = binding.slotIndex + binding.arrayIndex;

            D3D12_CPU_DESCRIPTOR_HANDLE srcHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
            srcHandle.ptr += uavIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            cpuView.ptr += dstIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        backendContext->descRingBufferBase += maximumUavIndex + 1;
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    const uint32_t maximumSrvIndex = pipeline.srvTextureCount + pipeline.srvBufferCount;
    if (maximumSrvIndex > 0)
    {
        if (backendContext->descRingBufferBase + maximumSrvIndex + 1 > FFX_RING_BUFFER_DESCRIPTOR_COUNT * backendContext->maxEffectContexts)
        {
            backendContext->descRingBufferBase = 0;
        }
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        for (uint32_t i = 0; i < pipeline.srvTextureCount; ++i)
        {
            if (job->fragmentJobDescriptor.srvTextures[i].resource.internalIndex == 0)
            {
                continue;
            }
            addBarrier(backendContext, &job->fragmentJobDescriptor.srvTextures[i].resource, FFX_RESOURCE_STATE_PIXEL_READ);

            const FfxResourceBinding binding  = pipeline.srvTextureBindings[i];
            const uint32_t           resIndex = job->fragmentJobDescriptor.srvTextures[i].resource.internalIndex;
            const uint32_t           dstIndex = binding.slotIndex + binding.arrayIndex;

            D3D12_CPU_DESCRIPTOR_HANDLE srcHandle = backendContext->descHeapSrvCpu->GetCPUDescriptorHandleForHeapStart();
            srcHandle.ptr += resIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            D3D12_CPU_DESCRIPTOR_HANDLE cpuView = dx12DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            cpuView.ptr += backendContext->descRingBufferBase * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            cpuView.ptr += dstIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            dx12Device->CopyDescriptorsSimple(1, cpuView, srcHandle, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        backendContext->descRingBufferBase += maximumSrvIndex + 1;
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    BackendContext_DX12::EffectContext& effectContext = backendContext->pEffectContexts[effectContextId];

    if (pipeline.staticTextureSrvCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessTextureSrvHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }
    if (pipeline.staticBufferSrvCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessBufferSrvHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }
    if (pipeline.staticTextureUavCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessTextureUavHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }
    if (pipeline.staticBufferUavCount > 0)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuView = dx12DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        gpuView.ptr += effectContext.bindlessBufferUavHeapStart * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dx12CommandList->SetGraphicsRootDescriptorTable(descriptorTableIndex++, gpuView);
    }

    // Render targets, and the barrier that makes them writable.
    for (uint32_t rt = 0; rt < rtCount; ++rt)
    {
        addBarrier(backendContext, &job->fragmentJobDescriptor.rtTextures[rt].resource, FFX_RESOURCE_STATE_RENDER_TARGET);
    }
    flushBarriers(backendContext, dx12CommandList);

    // Build an RTV per target. The heap is shared and small, so this takes the next
    // descriptor rather than trying to cache one per resource.
    for (uint32_t rt = 0; rt < rtCount; ++rt)
    {
        rtHandles[rt] = backendContext->descHeapRtvCpu->GetCPUDescriptorHandleForHeapStart();
        rtHandles[rt].ptr += backendContext->nextRtvDescriptor * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        backendContext->nextRtvDescriptor = (backendContext->nextRtvDescriptor + 1) % (FFX_MAX_NUM_RTS * 64);

        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
        rtvDesc.Format        = rtFormats[rt];
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        dx12Device->CreateRenderTargetView(rtResources[rt], &rtvDesc, rtHandles[rt]);
    }
    dx12CommandList->OMSetRenderTargets(rtCount, rtHandles, FALSE, nullptr);

    // Full-screen triangle, one draw, no vertex buffer -- as on Vulkan.
    const D3D12_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(job->fragmentJobDescriptor.viewport[0]),
                                     static_cast<float>(job->fragmentJobDescriptor.viewport[1]), 0.0f, 1.0f};
    dx12CommandList->RSSetViewports(1, &viewport);
    const D3D12_RECT scissor = {0, 0, static_cast<LONG>(job->fragmentJobDescriptor.viewport[0]),
                                static_cast<LONG>(job->fragmentJobDescriptor.viewport[1])};
    dx12CommandList->RSSetScissorRects(1, &scissor);

    dx12CommandList->SetPipelineState(pRecord->cachedPso);

    for (uint32_t cb = 0; cb < pipeline.constCount; ++cb)
    {
        FfxConstantAllocation allocation;
        if (s_fpConstantAllocator)
        {
            allocation = s_fpConstantAllocator(job->fragmentJobDescriptor.cbs[cb].data,
                                               job->fragmentJobDescriptor.cbs[cb].num32BitEntries * sizeof(uint32_t));
        }
        else
        {
            allocation = backendContext->FallbackConstantAllocator(job->fragmentJobDescriptor.cbs[cb].data,
                                                                   job->fragmentJobDescriptor.cbs[cb].num32BitEntries * sizeof(uint32_t));
        }
        dx12CommandList->SetGraphicsRootConstantBufferView(descriptorTableIndex + cb, D3D12_GPU_VIRTUAL_ADDRESS(allocation.handle));
    }

    dx12CommandList->DrawInstanced(3, 1, 0, 0);

    // Unbind so a later compute dispatch cannot be affected by these targets still being
    // bound.
    dx12CommandList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);

    return FFX_OK;
}

static FfxErrorCode executeGpuJobCopy(BackendContext_DX12* backendContext, FfxGpuJobDescription* job, ID3D12GraphicsCommandList* dx12CommandList)
{
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendContext->device);

    ID3D12Resource* dx12ResourceSrc = getDX12ResourcePtr(backendContext, job->copyJobDescriptor.src.internalIndex);
    ID3D12Resource* dx12ResourceDst = getDX12ResourcePtr(backendContext, job->copyJobDescriptor.dst.internalIndex);
    D3D12_RESOURCE_DESC dx12ResourceDescriptionDst = dx12ResourceDst->GetDesc();
    D3D12_RESOURCE_DESC dx12ResourceDescriptionSrc = dx12ResourceSrc->GetDesc();

    addBarrier(backendContext, &job->copyJobDescriptor.src, FFX_RESOURCE_STATE_COPY_SRC);
    addBarrier(backendContext, &job->copyJobDescriptor.dst, FFX_RESOURCE_STATE_COPY_DEST);
    flushBarriers(backendContext, dx12CommandList);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT dx12Footprint = {};
    UINT rowCount;
    UINT64 rowSizeInBytes;
    UINT64 totalBytes;
    dx12Device->GetCopyableFootprints(&dx12ResourceDescriptionDst, 0, 1, 0, &dx12Footprint, &rowCount, &rowSizeInBytes, &totalBytes);

    if (dx12ResourceDescriptionDst.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        dx12CommandList->CopyBufferRegion(dx12ResourceDst,
                                          job->copyJobDescriptor.dstOffset,
                                          dx12ResourceSrc,
                                          job->copyJobDescriptor.srcOffset,
                                          job->copyJobDescriptor.size > 0 ? job->copyJobDescriptor.size : totalBytes);
    }
    else if (dx12ResourceDescriptionSrc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        // TODO: account for source buffer offset
        D3D12_TEXTURE_COPY_LOCATION dx12SourceLocation = {};
        dx12SourceLocation.pResource = dx12ResourceSrc;
        dx12SourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dx12SourceLocation.PlacedFootprint = dx12Footprint;

        D3D12_TEXTURE_COPY_LOCATION dx12DestinationLocation = {};
        dx12DestinationLocation.pResource = dx12ResourceDst;
        dx12DestinationLocation.SubresourceIndex = 0;
        dx12DestinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        dx12CommandList->CopyTextureRegion(&dx12DestinationLocation, 0, 0, 0, &dx12SourceLocation, nullptr);
    }
    else
    {
        D3D12_TEXTURE_COPY_LOCATION dx12SourceLocation = {};
        dx12SourceLocation.pResource = dx12ResourceSrc;
        dx12SourceLocation.SubresourceIndex = 0;
        dx12SourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        D3D12_TEXTURE_COPY_LOCATION dx12DestinationLocation = {};
        dx12DestinationLocation.pResource = dx12ResourceDst;
        dx12DestinationLocation.SubresourceIndex = 0;
        dx12DestinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        dx12CommandList->CopyTextureRegion(&dx12DestinationLocation, 0, 0, 0, &dx12SourceLocation, nullptr);
    }

    return FFX_OK;
}

static FfxErrorCode executeGpuJobBarrier(BackendContext_DX12* backendContext, FfxGpuJobDescription* job, ID3D12GraphicsCommandList* dx12CommandList)
{
    addBarrier(backendContext, &job->barrierDescriptor.resource, job->barrierDescriptor.newState);
    flushBarriers(backendContext, dx12CommandList);

    return FFX_OK;
}

static FfxErrorCode executeGpuJobTimestamp(BackendContext_DX12*, FfxGpuJobDescription*, ID3D12GraphicsCommandList*)
{
    return FFX_OK;
}

static FfxErrorCode executeGpuJobClearFloat(BackendContext_DX12* backendContext, FfxGpuJobDescription* job, ID3D12GraphicsCommandList* dx12CommandList)
{
    ID3D12Device* dx12Device = reinterpret_cast<ID3D12Device*>(backendContext->device);

    uint32_t idx = job->clearJobDescriptor.target.internalIndex;
    BackendContext_DX12::Resource ffxResource = backendContext->pResources[idx];
    ID3D12Resource* dx12Resource = reinterpret_cast<ID3D12Resource*>(ffxResource.resourcePtr);
    uint32_t uavIndex = ffxResource.uavDescIndex;

    D3D12_CPU_DESCRIPTOR_HANDLE dx12CpuHandle = backendContext->descHeapUavCpu->GetCPUDescriptorHandleForHeapStart();
    dx12CpuHandle.ptr += uavIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_GPU_DESCRIPTOR_HANDLE dx12GpuHandle = backendContext->descHeapUavGpu->GetGPUDescriptorHandleForHeapStart();
    dx12GpuHandle.ptr += uavIndex * dx12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    dx12CommandList->SetDescriptorHeaps(1, &backendContext->descHeapUavGpu);

    addBarrier(backendContext, &job->clearJobDescriptor.target, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    flushBarriers(backendContext, dx12CommandList);

    uint32_t clearColorAsUint[4];
    clearColorAsUint[0] = reinterpret_cast<uint32_t&> (job->clearJobDescriptor.color[0]);
    clearColorAsUint[1] = reinterpret_cast<uint32_t&> (job->clearJobDescriptor.color[1]);
    clearColorAsUint[2] = reinterpret_cast<uint32_t&> (job->clearJobDescriptor.color[2]);
    clearColorAsUint[3] = reinterpret_cast<uint32_t&> (job->clearJobDescriptor.color[3]);
    dx12CommandList->ClearUnorderedAccessViewUint(dx12GpuHandle, dx12CpuHandle, dx12Resource, clearColorAsUint, 0, nullptr);

    return FFX_OK;
}

static FfxErrorCode executeGpuJobDiscard(BackendContext_DX12*       backendContext,
                                         FfxGpuJobDescription*      job,
                                         ID3D12GraphicsCommandList* dx12CommandList)
{
    uint32_t                            idx           = job->discardJobDescriptor.target.internalIndex;
    BackendContext_DX12::Resource       ffxResource   = backendContext->pResources[idx];
    ID3D12Resource*                     dx12Resource  = reinterpret_cast<ID3D12Resource*>(ffxResource.resourcePtr);

    addBarrier(backendContext, &job->discardJobDescriptor.target, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    flushBarriers(backendContext, dx12CommandList);

    dx12CommandList->DiscardResource(dx12Resource, nullptr);

    return FFX_OK;
}

FfxErrorCode ExecuteGpuJobsDX12(
    FfxInterface* backendInterface,
    FfxCommandList commandList, 
    FfxUInt32 effectContextId)
{
    FFX_ASSERT(NULL != backendInterface);
    BackendContext_DX12* backendContext = (BackendContext_DX12*)backendInterface->scratchBuffer;

    FFX_ASSERT(nullptr != commandList);
    ID3D12GraphicsCommandList* dx12CommandList = reinterpret_cast<ID3D12GraphicsCommandList*>(commandList);

    FfxErrorCode errorCode = FFX_OK;

    // execute all GpuJobs
    for (uint32_t currentGpuJobIndex = 0; currentGpuJobIndex < backendContext->gpuJobCount; ++currentGpuJobIndex) {

        FfxGpuJobDescription* GpuJob = &backendContext->pGpuJobs[currentGpuJobIndex];        
        // jobLabel only exists when FFX_DEBUG is defined -- the fork put it behind
        // that guard, and ffx_vk.cpp guards its uses the same way.
#ifdef FFX_DEBUG
        if (GpuJob->jobLabel[0]) {
            beginMarkerDX12(backendContext, dx12CommandList, GpuJob->jobLabel);
        }
#endif

        switch (GpuJob->jobType) {

            case FFX_GPU_JOB_CLEAR_FLOAT:
                errorCode = executeGpuJobClearFloat(backendContext, GpuJob, dx12CommandList);
                break;

            case FFX_GPU_JOB_COPY:
                errorCode = executeGpuJobCopy(backendContext, GpuJob, dx12CommandList);
                break;

            case FFX_GPU_JOB_COMPUTE:
                errorCode = executeGpuJobCompute(backendContext, GpuJob, dx12CommandList, effectContextId);
                break;

            case FFX_GPU_JOB_BARRIER:
                errorCode = executeGpuJobBarrier(backendContext, GpuJob, dx12CommandList);
                break;

            case FFX_GPU_JOB_DISCARD:
                errorCode = executeGpuJobDiscard(backendContext, GpuJob, dx12CommandList);
                break;

            case FFX_GPU_JOB_DATA_GRAPH:
                errorCode = executeGpuJobDataGraphDX12(backendContext, GpuJob, dx12CommandList);
                break;

            case FFX_GPU_JOB_FRAGMENT:
                errorCode = executeGpuJobFragmentDX12(backendContext, GpuJob, dx12CommandList, effectContextId);
                break;

            default:
                break;
        }
#ifdef FFX_DEBUG
        if (GpuJob->jobLabel[0]) {
            endMarkerDX12(backendContext, dx12CommandList);
        }
#endif
    }

    // check the execute function returned cleanly.
    FFX_RETURN_ON_ERROR(
        errorCode == FFX_OK,
        FFX_ERROR_BACKEND_API_ERROR);

    backendContext->gpuJobCount = 0;

    return FFX_OK;
}

// VirtualAlloc() + OpenExistingHeapFromAddress() + CreatePlacedResource() path, ensures that Breadcrumb buffer survives TDR.
static void breadcrumbsAllocBlockVirtual(ID3D12Device3* dx12Device, D3D12_RESOURCE_DESC* resDesc, FfxBreadcrumbsBlockData* blockData)
{
    // No need to lock, called on new block before placing it inside list
    D3D12_FEATURE_DATA_EXISTING_HEAPS existingHeaps = {};
    if (SUCCEEDED(dx12Device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existingHeaps, sizeof(existingHeaps))) && existingHeaps.Supported)
    {
        blockData->memory = VirtualAlloc(nullptr, (SIZE_T)resDesc->Width, MEM_COMMIT, PAGE_READWRITE);
        if (blockData->memory != nullptr)
        {
            ID3D12Heap* heap = nullptr;
            if (SUCCEEDED(dx12Device->OpenExistingHeapFromAddress(blockData->memory, IID_PPV_ARGS(&heap))))
            {
                ID3D12Resource* resource = nullptr;
                if (SUCCEEDED(dx12Device->CreatePlacedResource(heap, 0, resDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&resource))))
                {
                    resource->SetName(L"Buffer for Breadcrumbs - placed in VirtualAlloc, OpenExistingHeapFromAddress");
                    blockData->heap = (void*)heap;
                    blockData->buffer = (void*)resource;
                    return;
                }
                heap->Release();
            }
            const BOOL status = VirtualFree(blockData->memory, 0, MEM_RELEASE);
            FFX_ASSERT_MESSAGE(status != 0, "Error while releasing Breadcrumb memory!");
            blockData->memory = nullptr;
        }
    }
}

FfxErrorCode BreadcrumbsAllocBlockDX12(
    FfxInterface* backendInterface,
    uint64_t blockBytes,
    FfxBreadcrumbsBlockData* blockData)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != blockData);

    // Resource description.
    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Alignment = 0;
    resDesc.Width = blockBytes;
    resDesc.Height = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels = 1;
    resDesc.Format = DXGI_FORMAT_UNKNOWN;
    resDesc.SampleDesc.Count = 1;
    resDesc.SampleDesc.Quality = 0;
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

    ID3D12Device3* dev = nullptr;
    if (SUCCEEDED(((ID3D12Device*)backendInterface->device)->QueryInterface(IID_PPV_ARGS(&dev))))
    {
        breadcrumbsAllocBlockVirtual(dev, &resDesc, blockData);
        dev->Release();
    }

    // If VirtualAlloc path failed, try standard CreateCommittedResource().
    if (blockData->buffer == nullptr)
    {
        resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        ID3D12Resource* resource = nullptr;
        if (FAILED(((ID3D12Device*)backendInterface->device)->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&resource))))
        {
            // Cannot create breadcrumbs buffer!
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        const D3D12_RANGE range = {};
        if (FAILED(resource->Map(0, &range, &blockData->memory)))
        {
            resource->Release();
            // Cannot map breadcrumbs buffer!
            return FFX_ERROR_BACKEND_API_ERROR;
        }
        resource->SetName(L"Buffer for Breadcrumbs - committed");
        blockData->buffer = (void*)resource;
    }

    blockData->baseAddress = (uint64_t)((ID3D12Resource*)blockData->buffer)->GetGPUVirtualAddress();
    return FFX_OK;
}

void BreadcrumbsFreeBlockDX12(
    FfxInterface* backendInterface,
    FfxBreadcrumbsBlockData* blockData)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != blockData);

    if (blockData->buffer && !blockData->heap)
    {
        // CreateCommittedResource() path
        if (blockData->memory)
        {
            ((ID3D12Resource*)blockData->buffer)->Unmap(0, nullptr);
            blockData->memory = nullptr;
        }
        ((ID3D12Resource*)blockData->buffer)->Release();
        blockData->buffer = nullptr;
    }
    else
    {
        // VirutalAlloc() path
        if (blockData->buffer)
        {
            ((ID3D12Resource*)blockData->buffer)->Release();
            blockData->buffer = nullptr;
        }
        if (blockData->heap)
        {
            ((ID3D12Heap*)blockData->heap)->Release();
            blockData->heap = nullptr;
        }
        if (blockData->memory)
        {
            const BOOL status = VirtualFree(blockData->memory, 0, MEM_RELEASE);
            FFX_ASSERT_MESSAGE(status != 0, "Error while releasing Breadcrumb memory!");
            blockData->memory = nullptr;
        }
    }
}

void BreadcrumbsWriteDX12(
    FfxInterface* backendInterface,
    FfxCommandList commandList,
    uint32_t value,
    uint64_t gpuLocation,
    void* gpuBuffer,
    bool isBegin)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != gpuBuffer);

    const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = isBegin ? D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN : D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER params = { gpuLocation, value };

    ID3D12GraphicsCommandList2* cl = nullptr;
    if (SUCCEEDED(((ID3D12GraphicsCommandList*)commandList)->QueryInterface(IID_PPV_ARGS(&cl))))
    {
        cl->WriteBufferImmediate(1, &params, &mode);
        cl->Release();
    }
}

void breadcrumbsPrintDeviceInfoMemory(char** printBuffer, size_t* printSize, DXGI_MEMORY_SEGMENT_GROUP segment,
    const DXGI_QUERY_VIDEO_MEMORY_INFO* memInfo, FfxAllocationCallbacks* allocs)
{
    FFX_ASSERT(NULL != printBuffer);
    FFX_ASSERT(NULL != printSize);
    FFX_ASSERT(NULL != memInfo);

    if (segment == DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
    {
        FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, FFX_BREADCRUMBS_PRINTING_INDENT "Local memory:\n");
    }
    else if (segment == DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
    {
        FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, FFX_BREADCRUMBS_PRINTING_INDENT "Non-local memory:\n");
    }
    else
    {
        FFX_ASSERT_FAIL("Unknown segment group!");
        return;
    }

    FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, FFX_BREADCRUMBS_PRINTING_INDENT FFX_BREADCRUMBS_PRINTING_INDENT "Budget");
    FFX_BREADCRUMBS_APPEND_UINT64(*printBuffer, *printSize, (size_t)memInfo->CurrentUsage);
    FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, "/");
    FFX_BREADCRUMBS_APPEND_UINT64(*printBuffer, *printSize, (size_t)memInfo->Budget);
    FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, " B\n" FFX_BREADCRUMBS_PRINTING_INDENT FFX_BREADCRUMBS_PRINTING_INDENT "Reservation ");
    FFX_BREADCRUMBS_APPEND_UINT64(*printBuffer, *printSize, (size_t)memInfo->CurrentReservation);
    FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, "/");
    FFX_BREADCRUMBS_APPEND_UINT64(*printBuffer, *printSize, (size_t)memInfo->AvailableForReservation);
    FFX_BREADCRUMBS_APPEND_STRING(*printBuffer, *printSize, " B\n");
}

void BreadcrumbsPrintDeviceInfoDX12(
    FfxInterface* backendInterface,
    FfxAllocationCallbacks* allocs,
    bool extendedInfo,
    char** printBuffer,
    size_t* printSize)
{
    FFX_ASSERT(NULL != backendInterface);
    FFX_ASSERT(NULL != allocs);
    FFX_ASSERT(NULL != printBuffer);
    FFX_ASSERT(NULL != printSize);
    char* buff = *printBuffer;
    size_t buffSize = *printSize;

    ID3D12Device* dev = (ID3D12Device*)backendInterface->device;

    // Display as many feature info as possible.
    bool nonLocalRegionAvailable = false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture1 = {};
    architecture1.NodeIndex = 0;
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture1, sizeof(architecture1))))
    {
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[ARCHITECTURE1]\n");
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, architecture1, TileBasedRenderer);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, architecture1, UMA);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, architecture1, CacheCoherentUMA);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, architecture1, IsolatedMMU);
        nonLocalRegionAvailable = !architecture1.UMA;
    }

    // Create proper DXGI factory to get adapter
    IDXGIFactory2* oldFactory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&oldFactory))))
    {
        IDXGIFactory4* factory = nullptr;
        if (SUCCEEDED(oldFactory->QueryInterface(&factory)))
        {
            IDXGIAdapter3* adapter = nullptr;
            if (SUCCEEDED(factory->EnumAdapterByLuid(dev->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
            {
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[ADAPTER INFO]\n");

                DXGI_ADAPTER_DESC2 desc = {};
                adapter->GetDesc2(&desc);

                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "Description: ");
                const size_t descLength = wcslen(desc.Description);
                buff = (char*)ffxBreadcrumbsAppendList(buff, buffSize, 1, descLength + 1, allocs);
                for (uint8_t i = 0; i < descLength; ++i)
                    buff[buffSize++] = (char)desc.Description[i];
                buff[buffSize++] = '\n';

                FFX_BREADCRUMBS_PRINT_HEX32(buff, buffSize, desc, VendorId);
                FFX_BREADCRUMBS_PRINT_HEX32(buff, buffSize, desc, SubSysId);
                FFX_BREADCRUMBS_PRINT_HEX32(buff, buffSize, desc, Revision);
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "DedicatedVideoMemory: ");
                FFX_BREADCRUMBS_APPEND_UINT64(buff, buffSize, desc.DedicatedVideoMemory);
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT " B\nDedicatedSystemMemory: ");
                FFX_BREADCRUMBS_APPEND_UINT64(buff, buffSize, desc.DedicatedSystemMemory);
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT " B\nSharedSystemMemory: ");
                FFX_BREADCRUMBS_APPEND_UINT64(buff, buffSize, desc.SharedSystemMemory);
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT " B\nDXGI_ADAPTER_FLAG_SOFTWARE: ");
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "True\n");
                }
                else
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "False\n");
                }

                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "GraphicsPreemptionGranularity: ");
                switch (desc.GraphicsPreemptionGranularity)
                {
                case DXGI_GRAPHICS_PREEMPTION_DMA_BUFFER_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_GRAPHICS_PREEMPTION_DMA_BUFFER_BOUNDARY");
                    break;
                }
                case DXGI_GRAPHICS_PREEMPTION_PRIMITIVE_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_GRAPHICS_PREEMPTION_PRIMITIVE_BOUNDARY");
                    break;
                }
                case DXGI_GRAPHICS_PREEMPTION_TRIANGLE_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_GRAPHICS_PREEMPTION_TRIANGLE_BOUNDARY");
                    break;
                }
                case DXGI_GRAPHICS_PREEMPTION_PIXEL_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_GRAPHICS_PREEMPTION_PIXEL_BOUNDARY");
                    break;
                }
                case DXGI_GRAPHICS_PREEMPTION_INSTRUCTION_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_GRAPHICS_PREEMPTION_INSTRUCTION_BOUNDARY");
                    break;
                }
                default:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "UNKNOWN");
                    break;
                }
                }

                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n" FFX_BREADCRUMBS_PRINTING_INDENT "ComputePreemptionGranularity: ");
                switch (desc.ComputePreemptionGranularity)
                {
                case DXGI_COMPUTE_PREEMPTION_DMA_BUFFER_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_COMPUTE_PREEMPTION_DMA_BUFFER_BOUNDARY\n");
                    break;
                }
                case DXGI_COMPUTE_PREEMPTION_DISPATCH_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_COMPUTE_PREEMPTION_DISPATCH_BOUNDARY\n");
                    break;
                }
                case DXGI_COMPUTE_PREEMPTION_THREAD_GROUP_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_COMPUTE_PREEMPTION_THREAD_GROUP_BOUNDARY\n");
                    break;
                }
                case DXGI_COMPUTE_PREEMPTION_THREAD_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_COMPUTE_PREEMPTION_THREAD_BOUNDARY\n");
                    break;
                }
                case DXGI_COMPUTE_PREEMPTION_INSTRUCTION_BOUNDARY:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "DXGI_COMPUTE_PREEMPTION_INSTRUCTION_BOUNDARY\n");
                    break;
                }
                default:
                {
                    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "UNKNOWN\n");
                    break;
                }
                }

                DXGI_QUERY_VIDEO_MEMORY_INFO memInfo = {};
                if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo)))
                    breadcrumbsPrintDeviceInfoMemory(&buff, &buffSize, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo, allocs);
                if (nonLocalRegionAvailable)
                {
                    // Only on NUMA devices
                    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &memInfo)))
                        breadcrumbsPrintDeviceInfoMemory(&buff, &buffSize, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &memInfo, allocs);
                }
                adapter->Release();
            }
            factory->Release();
        }
        oldFactory->Release();
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS d3d12Options = {};
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3d12Options, sizeof(d3d12Options))))
    {
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS]\n");
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, DoublePrecisionFloatShaderOps);
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "MinPrecisionSupport: 32");
        if (d3d12Options.MinPrecisionSupport & D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT)
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "/16");
        }
        if (d3d12Options.MinPrecisionSupport & D3D12_SHADER_MIN_PRECISION_SUPPORT_10_BIT)
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "/10");
        }
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, " bit\n");

        FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options, TiledResourcesTier);
        FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options, ResourceBindingTier);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, PSSpecifiedStencilRefSupported);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, TypedUAVLoadAdditionalFormats);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, ROVsSupported);
        FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options, ConservativeRasterizationTier);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, StandardSwizzle64KBSupported);

        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "CrossNodeSharingTier: ");
        switch (d3d12Options.CrossNodeSharingTier)
        {
        case D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "0");
            break;
        }
        case D3D12_CROSS_NODE_SHARING_TIER_1_EMULATED:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "1 Emulated");
            break;
        }
        default:
        {
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options.CrossNodeSharingTier) - 1);
            break;
        }
        }
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, CrossAdapterRowMajorTextureSupported);
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options, VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation);
        FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options, ResourceHeapTier);
    }

    const D3D_FEATURE_LEVEL requestedLevels[] =
    {
          D3D_FEATURE_LEVEL_12_0,
          D3D_FEATURE_LEVEL_12_1,
#ifdef __ID3D12Device9_FWD_DEFINED__
          D3D_FEATURE_LEVEL_12_2
#endif
    };
    D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};
    featureLevels.NumFeatureLevels = sizeof(requestedLevels) / sizeof(D3D_FEATURE_LEVEL);
    featureLevels.pFeatureLevelsRequested = requestedLevels;
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &featureLevels, sizeof(featureLevels))))
    {
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[FEATURE_LEVELS]\n" FFX_BREADCRUMBS_PRINTING_INDENT "MaxSupportedFeatureLevel: ");
        switch (featureLevels.MaxSupportedFeatureLevel)
        {
        default:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "UNKNOWN\n");
            break;
        }
        case D3D_FEATURE_LEVEL_12_0:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "12_0\n");
            break;
        }
        case D3D_FEATURE_LEVEL_12_1:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "12_1\n");
            break;
        }
#ifdef __ID3D12Device9_FWD_DEFINED__
        case D3D_FEATURE_LEVEL_12_2:
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "12_2\n");
            break;
        }
#endif // #ifdef __ID3D12Device9_FWD_DEFINED__
        }
    }

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel =
    {
#if defined(__ID3D12Device10_FWD_DEFINED__)
        D3D_SHADER_MODEL_6_7
#elif defined(__ID3D12Device9_FWD_DEFINED__)
        D3D_SHADER_MODEL_6_6
#elif defined(__ID3D12Device8_FWD_DEFINED__)
        D3D_SHADER_MODEL_6_5
#else
        D3D_SHADER_MODEL_6_4
#endif
    };
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))))
    {
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[SHADER_MODEL]\n" FFX_BREADCRUMBS_PRINTING_INDENT "HighestShaderModel: ");
        FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, (shaderModel.HighestShaderModel >> 4));
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, ".");
        FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, (shaderModel.HighestShaderModel & 0x0F));
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 d3d12Options5 = {};
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &d3d12Options5, sizeof(d3d12Options5))))
    {
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS5]\n");
        FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options5, SRVOnlyTiledResourceTier3);
        FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options5, RenderPassesTier);
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "RaytracingTier: ");
        FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options5.RaytracingTier) / 10);
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, ".");
        FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options5.RaytracingTier) % 10);
        FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
    }

    if (extendedInfo)
    {
        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT addressSupport = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &addressSupport, sizeof(addressSupport))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[GPU_VIRTUAL_ADDRESS_SUPPORT]\n");
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, addressSupport, MaxGPUVirtualAddressBitsPerResource);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, addressSupport, MaxGPUVirtualAddressBitsPerProcess);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS1 d3d12Options1 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &d3d12Options1, sizeof(d3d12Options1))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS1]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options1, WaveOps);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options1, WaveLaneCountMin);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options1, WaveLaneCountMax);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options1, TotalLaneCount);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options1, ExpandedComputeResourceStates);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options1, Int64ShaderOps);
        }

        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT protectedSessionSupport = {};
        protectedSessionSupport.NodeIndex = 0;
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT, &protectedSessionSupport, sizeof(protectedSessionSupport))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[PROTECTED_RESOURCE_SESSION_SUPPORT]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, protectedSessionSupport, Support);
        }

        D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature = {};
        rootSignature.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignature, sizeof(rootSignature))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[ROOT_SIGNATURE]\n" FFX_BREADCRUMBS_PRINTING_INDENT "HighestVersion: ");
            switch (rootSignature.HighestVersion)
            {
            case D3D_ROOT_SIGNATURE_VERSION_1_0:
            {
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "1.0\n");
                break;
            }
            case D3D_ROOT_SIGNATURE_VERSION_1_1:
            {
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "1.1\n");
                break;
            }
            default:
            {
                FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "UNKNOW\n");
                break;
            }
            }
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS2 d3d12Options2 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &d3d12Options2, sizeof(d3d12Options2))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS2]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options2, DepthBoundsTestSupported);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options2, ProgrammableSamplePositionsTier);
        }

        D3D12_FEATURE_DATA_SHADER_CACHE shaderCache = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_CACHE, &shaderCache, sizeof(shaderCache))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[SHADER_CACHE]\n");
            FFX_BREADCRUMBS_PRINT_HEX32(buff, buffSize, shaderCache, SupportFlags);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS3 d3d12Options3 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &d3d12Options3, sizeof(d3d12Options3))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS3]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options3, CopyQueueTimestampQueriesSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options3, CastingFullyTypedFormatSupported);
            FFX_BREADCRUMBS_PRINT_HEX32(buff, buffSize, d3d12Options3, WriteBufferImmediateSupportFlags);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options3, ViewInstancingTier);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options3, BarycentricsSupported);
        }

        D3D12_FEATURE_DATA_EXISTING_HEAPS existingHeaps = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existingHeaps, sizeof(existingHeaps))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[EXISTING_HEAPS]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, existingHeaps, Supported);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS4 d3d12Options4 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &d3d12Options4, sizeof(d3d12Options4))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS4]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options4, MSAA64KBAlignedTextureSupported);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options4, SharedResourceCompatibilityTier);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options4, Native16BitShaderOpsSupported);
        }

        D3D12_FEATURE_DATA_SERIALIZATION serialization = {};
        serialization.NodeIndex = 0;
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_SERIALIZATION, &serialization, sizeof(serialization))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[SERIALIZATION]\n");
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, serialization, HeapSerializationTier);
        }

        D3D12_FEATURE_DATA_CROSS_NODE crossNode = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_CROSS_NODE, &crossNode, sizeof(crossNode))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[CROSS_NODE]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, crossNode, AtomicShaderInstructions);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS6 d3d12Options6 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &d3d12Options6, sizeof(d3d12Options6))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS6]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options6, AdditionalShadingRatesSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options6, PerPrimitiveShadingRateSupportedWithViewportIndexing);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options6, VariableShadingRateTier);
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, d3d12Options6, ShadingRateImageTileSize);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options6, BackgroundProcessingSupported);
        }

#ifdef __ID3D12Device8_FWD_DEFINED__
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 d3d12Options7 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &d3d12Options7, sizeof(d3d12Options7))))
        {
            const uint32_t samplerTier = static_cast<uint32_t>(d3d12Options7.SamplerFeedbackTier) / 10;
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS7]\n" FFX_BREADCRUMBS_PRINTING_INDENT "MeshShaderTier: ");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options7.MeshShaderTier) / 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, ".");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options7.MeshShaderTier) % 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n" FFX_BREADCRUMBS_PRINTING_INDENT "SamplerFeedbackTier: ");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, samplerTier / 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, ".");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, samplerTier % 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
        }

        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT protectedSessionTypeCount = {};
        protectedSessionTypeCount.NodeIndex = 0;
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT, &protectedSessionTypeCount, sizeof(protectedSessionTypeCount))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[PROTECTED_RESOURCE_SESSION_TYPE_COUNT]\n");
            FFX_BREADCRUMBS_PRINT_UINT(buff, buffSize, protectedSessionTypeCount, Count);
        }
#endif // #ifdef __ID3D12Device8_FWD_DEFINED__

#ifdef __ID3D12Device9_FWD_DEFINED__
        D3D12_FEATURE_DATA_D3D12_OPTIONS8 d3d12Options8 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &d3d12Options8, sizeof(d3d12Options8))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS8]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options8, UnalignedBlockTexturesSupported);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS9 d3d12Options9 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &d3d12Options9, sizeof(d3d12Options9))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS9]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options9, MeshShaderPipelineStatsSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options9, MeshShaderSupportsFullRangeRenderTargetArrayIndex);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options9, AtomicInt64OnTypedResourceSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options9, AtomicInt64OnGroupSharedSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options9, DerivativesInMeshAndAmplificationShadersSupported);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, FFX_BREADCRUMBS_PRINTING_INDENT "WaveMMATier: ");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options9.WaveMMATier) / 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, ".");
            FFX_BREADCRUMBS_APPEND_UINT(buff, buffSize, ((uint32_t)d3d12Options9.WaveMMATier) % 10);
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
        }
#endif // #ifdef __ID3D12Device9_FWD_DEFINED__

#ifdef __ID3D12Device10_FWD_DEFINED__
        D3D12_FEATURE_DATA_D3D12_OPTIONS10 d3d12Options10 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS10, &d3d12Options10, sizeof(d3d12Options10))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS10]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options10, VariableRateShadingSumCombinerSupported);
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options10, MeshShaderPerPrimitiveShadingRateSupported);
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS11 d3d12Options11 = {};
        if (SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &d3d12Options11, sizeof(d3d12Options11))))
        {
            FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "[D3D12_OPTIONS11]\n");
            FFX_BREADCRUMBS_PRINT_BOOL(buff, buffSize, d3d12Options11, AtomicInt64OnDescriptorHeapResourceSupported);
        }
#endif // #ifdef __ID3D12Device10_FWD_DEFINED__
    }
    FFX_BREADCRUMBS_APPEND_STRING(buff, buffSize, "\n");
    *printBuffer = buff;
    *printSize = buffSize;
}

void RegisterConstantBufferAllocatorDX12(FfxInterface*, FfxConstantBufferAllocator fpConstantAllocator)
{
    s_fpConstantAllocator = fpConstantAllocator;
}

FfxCommandQueue ffxGetCommandQueueDX12(ID3D12CommandQueue* pCommandQueue)
{
    FFX_ASSERT(nullptr != pCommandQueue);
    return reinterpret_cast<FfxCommandQueue>(pCommandQueue);
}

FfxSwapchain ffxGetSwapchainDX12(IDXGISwapChain4* pSwapchain)
{
    FFX_ASSERT(nullptr != pSwapchain);
    return reinterpret_cast<FfxSwapchain>(pSwapchain);
}

IDXGISwapChain4* ffxGetDX12SwapchainPtr(FfxSwapchain ffxSwapchain)
{
    return reinterpret_cast<IDXGISwapChain4*>(ffxSwapchain);
}
