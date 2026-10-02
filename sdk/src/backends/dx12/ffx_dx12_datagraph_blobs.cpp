/*
 * ffx_dx12_datagraph_blobs.cpp -- data-graph model descriptors for the D3D12 backend.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * WHY THIS FILE EXISTS
 * =============================================================================
 *
 * `sdk/src/backends/shared/ffx_shader_blobs.cpp` provides
 * `ffxGetPermutationBlobByIndex` for the Vulkan backend by dispatching to the three
 * per-effect accessors under `shared/blob_accessors/`, each of which includes a pile of
 * GENERATED `*_permutations.h` headers full of shader blobs.
 *
 * That route is closed on D3D12 in this tree, and not for a small reason:
 *
 *   * `compile_shaders_with_depfile` picks its target language from the shader file
 *     extension -- `.glsl` goes through glslang to SPIR-V, `.hlsl` through dxc to DXIL.
 *   * This fork converted every shader to GLSL and deleted all HLSL. There are zero
 *     `.hlsl` files in the tree.
 *   * D3D12 consumes DXIL, so SPIR-V permutation headers would compile and then produce
 *     pipeline states that cannot be created.
 *
 * The dp4a inference backend does not have this problem, because it carries its own
 * premade DXIL inside `nfru_dp4a/nfru_shaders_dxil.h` -- it never asks the SDK for a
 * shader. So this file supplies exactly what the dp4a path needs and nothing else: the
 * `FfxDataGraphBlob` for each model, taken from the same hand-written descriptors the
 * Vulkan backend uses. Keeping them shared means the effect code's name-based tensor
 * joining (`patchResourceIdentifier`) behaves identically on both APIs.
 *
 * Consequently this backend is COMPUTE-ONLY and DATA-GRAPH-ONLY. A compute pipeline
 * request gets an explicit error rather than an empty blob that would be asserted on or
 * dereferenced. See SDK-INTEGRATION-NOTES.md.
 */
#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_nss.h>

#include <ffx_shader_blobs.h>

#include <string.h>

/* The same descriptors the Vulkan backend serves. */
#include <nfru_v1_int8.h>
#include <nss_v1_0_1_high_int8.h>
#include <nss_v1_0_1_mid_low_int8.h>

/* Declared in ffx_shader_blobs.h, which wraps it in extern "C" itself; the definition
 * must not repeat the linkage specification. The two trailing parameters carry default
 * arguments in the declaration, which also must not be repeated here. */
FfxErrorCode ffxGetPermutationBlobByIndex(
    FfxEffect         effectId,
    FfxPass           passId,
    uint32_t          permutationOptions,
    FfxShaderBlob*    outBlob,
    FfxShaderBlob*    outVertBlob,
    FfxDataGraphBlob* outDataGraphBlob)
{
    (void)permutationOptions;

    if (outBlob != nullptr)
    {
        memset(outBlob, 0, sizeof(FfxShaderBlob));
    }
    if (outVertBlob != nullptr)
    {
        memset(outVertBlob, 0, sizeof(FfxShaderBlob));
    }
    if (outDataGraphBlob != nullptr)
    {
        memset(outDataGraphBlob, 0, sizeof(FfxDataGraphBlob));
    }
    else
    {
        /*
         * A compute-shader request. There are no compute shaders in this backend, so say
         * so instead of handing back a zeroed blob that the caller would then assert on
         * or dereference.
         */
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    switch (effectId)
    {
    case ARM_EFFECT_FRAMEINTERPOLATION:
        if (passId == FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION)
        {
            memcpy(outDataGraphBlob, &g_nfru_v1_int8_Info, sizeof(FfxDataGraphBlob));
            return FFX_OK;
        }
        break;

    case FFX_EFFECT_NSS:
        /*
         * The NSS graphs are described so that the pipeline plumbing -- tensor counts,
         * names and slots -- is well defined, and so that the same
         * CreateDataGraphPipelineVK-style gate can refuse them cleanly. The dp4a backend
         * implements NFRU v1 only; running NFRU's weights against an NSS graph would
         * silently produce garbage, so an unimplemented model must not look implemented.
         *
         * The quality split is not resolvable here: ffxGetPermutationBlobByIndex has no
         * shader-quality parameter, and unlike the Vulkan accessor this file does not
         * receive the permutation options that encode it. Both NSS graphs are therefore
         * handed out by pass only, and the backend refuses whichever one it is asked for.
         */
        if (passId == FFX_NSS_PASS_DATA_GRAPH)
        {
            memcpy(outDataGraphBlob, &g_nss_v1_0_1_high_int8_Info, sizeof(FfxDataGraphBlob));
            return FFX_OK;
        }
        break;

    default:
        break;
    }

    return FFX_ERROR_BACKEND_API_ERROR;
}
