// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/ffx_interface.h>
#include <stdlib.h>
#include <string.h>

/*
 * Bridge between this fork's optical flow API and AMD FidelityFX SDK 1.1.3's
 * implementation of it.
 *
 * Why this exists
 * ---------------
 * The fork replaced optical flow's seven compute passes with a single VK_ARM data-graph
 * pipeline, and in doing so redesigned the effect's public API: the context types became
 * FfxOpticalFlowContext / FfxOpticalFlowContextDescription (capital F), and three
 * functions appeared that 1.1.3 has no equivalent for -- ffxOpticalFlowGetSize,
 * ffxOpticalFlowGetGridSize and the two grid-size probes below, all of which exist to
 * negotiate against the data-graph backend.
 *
 * That API has a real consumer. ffx-api/src/ffx_provider_framegeneration.cpp -- the FSR3
 * frame generation provider, and the reason this SDK has a swapchain at all -- creates an
 * optical flow context, sizes it, queries its shared resources and dispatches it. 1.1.3's
 * spelling of the same API is lowercase, with leaner description structs.
 *
 * Rewriting the provider was never on the table, and renaming inside a 987-line component
 * would have hidden the seam. So the seam is here instead: the fork's API is implemented
 * on top of 1.1.3's component, in one file, where it can be read in full.
 *
 * What this is not: a second implementation. Every call below forwards to 1.1.3's
 * component. The seven compute passes, their shaders and their blobs are that component's.
 */

/*
 * 1.1.3's half of the API, declared here rather than pulled in from its header.
 *
 * The fork's header is the one in the tree -- it carries the fork's types and the C++
 * functions the frame generation provider was written against -- so 1.1.3's declarations
 * are not visible. These are defined by ffx_opticalflow.cpp, restored from 1.1.3, and by
 * nothing else. Deliberately not FFX_API: the component's own definitions carry the
 * export decoration, and redeclaring it here would be a second, weaker claim on the same
 * symbol rather than a description of it.
 */
extern FfxErrorCode ffxOpticalflowContextCreate(
    FfxOpticalflowContext* context, FfxOpticalflowContextDescription* contextDescription);
extern FfxErrorCode ffxOpticalflowContextDestroy(FfxOpticalflowContext* context);
extern FfxErrorCode ffxOpticalflowGetSharedResourceDescriptions(
    FfxOpticalflowContext* context, FfxOpticalflowSharedResourceDescriptions* sharedResources);
extern FfxErrorCode ffxOpticalflowContextDispatch(
    FfxOpticalflowContext* context, const FfxOpticalflowDispatchDescription* dispatchDescription);
extern FfxVersionNumber ffxOpticalflowGetEffectVersion();

namespace {

/*
 * 1.1.3's context does not fit in the fork's opaque buffer -- the fork sized that buffer
 * for its own, smaller context -- so the shim holds a pointer rather than the context
 * itself. A pointer fits any buffer worth the name, and the allocation is one per optical
 * flow context, so heap is the honest place for it.
 */
constexpr uint32_t kShimMagic = 0x4F464C4Du;  // 'OFLM'

struct OpticalFlowShim {
    uint32_t              magic;
    FfxOpticalflowContext of;  // the 1.1.3 context, which does all the work
};

inline OpticalFlowShim** ShimSlot(FfxOpticalFlowContext* context)
{
    return reinterpret_cast<OpticalFlowShim**>(context->data);
}

inline OpticalFlowShim* ShimOf(FfxOpticalFlowContext* context)
{
    OpticalFlowShim* shim = *ShimSlot(context);
    return (shim != nullptr && shim->magic == kShimMagic) ? shim : nullptr;
}

}  // namespace

/*
 * The fork negotiated an optical flow grid size with the data-graph backend, because the
 * graph could be built for several. The compute implementation has a fixed 8-pixel block
 * (FFX_OPTICALFLOW_BLOCK_SIZE in the callbacks header) and does not negotiate.
 *
 * Reported as UNKNOWN rather than inventing a value: the caller initialises to UNKNOWN and
 * treats it as "not specified", and a made-up concrete value would be a claim the compute
 * path cannot honour. If a consumer needs a concrete grid size, this is the one place to
 * change, and the block size above is what it should report.
 */
FFX_API FfxOpticalFlowGridSize ffxGetDefaultDataGraphOpticalFlowGridSize(FfxInterface& backendInterface)
{
    (void)backendInterface;
    return FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
}

FFX_API bool ffxOpticalFlowGridSizeSupported(FfxInterface& backendInterface, const FfxOpticalFlowGridSize gridSize)
{
    (void)backendInterface;
    /* Only "unspecified" is true: see the note above. */
    return gridSize == FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
}

FFX_API FfxErrorCode ffxOpticalFlowContextCreate(
    FfxOpticalFlowContext* context, FfxOpticalFlowContextDescription* contextDescription)
{
    if (context == nullptr || contextDescription == nullptr)
    {
        return FFX_ERROR_INVALID_POINTER;
    }

    OpticalFlowShim* shim = static_cast<OpticalFlowShim*>(malloc(sizeof(OpticalFlowShim)));
    if (shim == nullptr)
    {
        return FFX_ERROR_OUT_OF_MEMORY;
    }
    memset(shim, 0, sizeof(*shim));
    shim->magic = kShimMagic;

    /*
     * 1.1.3's context description is three fields where the fork's is eight: it derives
     * the render size, the view projection, the back buffer format and the performance
     * level itself. The fork's gridSize and performanceLevel exist only to configure the
     * data-graph pipeline and have nothing to configure here.
     *
     * maxRenderSize, initialViewProjection and backBufferFormat are therefore dropped
     * rather than forwarded. That is the substantive difference between the two APIs and
     * it is worth being explicit about: 1.1.3 sizes the effect from `resolution` alone.
     */
    FfxOpticalflowContextDescription ofDescription = {};
    ofDescription.backendInterface = contextDescription->backendInterface;
    ofDescription.flags            = contextDescription->flags;
    ofDescription.resolution       = contextDescription->resolution;

    const FfxErrorCode result = ffxOpticalflowContextCreate(&shim->of, &ofDescription);
    if (result != FFX_OK)
    {
        free(shim);
        return result;
    }

    *ShimSlot(context) = shim;
    return FFX_OK;
}

FFX_API FfxErrorCode ffxOpticalFlowContextDestroy(FfxOpticalFlowContext* context)
{
    if (context == nullptr)
    {
        return FFX_ERROR_INVALID_POINTER;
    }
    OpticalFlowShim* shim = ShimOf(context);
    if (shim == nullptr)
    {
        return FFX_ERROR_INVALID_ARGUMENT;
    }
    const FfxErrorCode result = ffxOpticalflowContextDestroy(&shim->of);
    shim->magic = 0;
    free(shim);
    *ShimSlot(context) = nullptr;
    return result;
}

FFX_API FfxErrorCode ffxOpticalFlowGetSharedResourceDescriptions(
    FfxOpticalFlowContext* context, FfxOpticalFlowSharedResourceDescriptions* sharedResources)
{
    if (context == nullptr || sharedResources == nullptr)
    {
        return FFX_ERROR_INVALID_POINTER;
    }
    OpticalFlowShim* shim = ShimOf(context);
    if (shim == nullptr)
    {
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    FfxOpticalflowSharedResourceDescriptions ofResources = {};
    const FfxErrorCode result = ffxOpticalflowGetSharedResourceDescriptions(&shim->of, &ofResources);
    if (result != FFX_OK)
    {
        return result;
    }

    memset(sharedResources, 0, sizeof(*sharedResources));
    sharedResources->opticalFlowVector = ofResources.opticalFlowVector;
    sharedResources->opticalFlowSCD    = ofResources.opticalFlowSCD;

    /*
     * depthTm1 / depthTm1Next / colorTm1 are left zeroed, and that is a real difference
     * rather than an omission. The fork's implementation consumed a previous-frame depth
     * and colour pair, which the provider allocates from these descriptions; 1.1.3's
     * optical flow is fed by a single colour resource and keeps whatever history it needs
     * internally, in the pyramid and flow buffers it allocates itself.
     *
     * A caller that dereferences these will get a null resource. The frame generation
     * provider does not: it forwards opticalFlowVector to frame interpolation and nothing
     * else.
     */
    return FFX_OK;
}

FFX_API FfxErrorCode ffxOpticalFlowContextDispatch(
    FfxOpticalFlowContext* context, FfxOpticalFlowDispatchDescription* dispatchDescription)
{
    if (context == nullptr || dispatchDescription == nullptr)
    {
        return FFX_ERROR_INVALID_POINTER;
    }
    OpticalFlowShim* shim = ShimOf(context);
    if (shim == nullptr)
    {
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    /*
     * Field by field, and the ones that do not carry over are the interesting part:
     *
     *   depth, viewProjection, depthTm1, colorTm1, meanFlowL1NormHint
     *       the fork's MV-hints path. 1.1.3's optical flow estimates flow from colour
     *       alone, so there is nothing to forward these to. Not a degradation relative to
     *       the compute implementation -- that implementation has no such inputs.
     *
     *   opticalFlowSCD, backbufferTransferFunction, minMaxLuminance
     *       1.1.3 fields, forwarded.
     */
    FfxOpticalflowDispatchDescription ofDispatch = {};
    ofDispatch.commandList                = dispatchDescription->commandList;
    ofDispatch.color                      = dispatchDescription->color;
    ofDispatch.opticalFlowVector          = dispatchDescription->opticalFlowVector;
    ofDispatch.opticalFlowSCD             = dispatchDescription->opticalFlowSCD;
    ofDispatch.reset                      = dispatchDescription->reset;
    ofDispatch.backbufferTransferFunction = dispatchDescription->backbufferTransferFunction;
    ofDispatch.minMaxLuminance            = dispatchDescription->minMaxLuminance;

    return ffxOpticalflowContextDispatch(&shim->of, &ofDispatch);
}

FFX_API FfxDimensions2D ffxOpticalFlowGetSize(FfxOpticalFlowContext* context)
{
    FfxDimensions2D size = {0, 0};
    if (context == nullptr)
    {
        return size;
    }
    OpticalFlowShim* shim = ShimOf(context);
    if (shim == nullptr)
    {
        return size;
    }

    /*
     * Read out of the flow-vector resource description rather than recomputed. The geometry
     * (block size, pyramid rounding) belongs to the component that allocates the resource;
     * duplicating its arithmetic here is how the two would drift apart.
     *
     * The dimensions live on the FfxResourceDescription the create-description wraps.
     */
    FfxOpticalflowSharedResourceDescriptions ofResources = {};
    if (ffxOpticalflowGetSharedResourceDescriptions(&shim->of, &ofResources) != FFX_OK)
    {
        return size;
    }
    size.width  = static_cast<uint32_t>(ofResources.opticalFlowVector.resourceDescription.width);
    size.height = static_cast<uint32_t>(ofResources.opticalFlowVector.resourceDescription.height);
    return size;
}

FFX_API FfxOpticalFlowGridSize ffxOpticalFlowGetGridSize(FfxOpticalFlowContext* context)
{
    (void)context;
    /* See ffxGetDefaultDataGraphOpticalFlowGridSize. */
    return FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
}

FFX_API FfxVersionNumber ffxOpticalFlowGetEffectVersion()
{
    return ffxOpticalflowGetEffectVersion();
}
