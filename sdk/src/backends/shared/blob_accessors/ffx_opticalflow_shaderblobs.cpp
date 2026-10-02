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

#include <FidelityFX/host/ffx_util.h>
#include "ffx_opticalflow_shaderblobs.h"
#include "opticalflow/ffx_opticalflow_private.h"

/*
 * FfxShaderBlob compatibility with the 1.1.3 shader compiler.
 *
 * The macro this system header defines lists fifteen reflection members that the Arm fork
 * added for its data-graph path -- numRTTextures, numSRVTensors, numUAVTensors and the
 * rtTexture* / srvTensor* / uavTensor* pointers. AMD 1.1.3's compiler does not emit them,
 * and it is the only compiler here that can build HLSL at all, which is what these passes
 * are. Neither optical flow nor SPD has a ray-tracing texture or a tensor, so zero is the
 * correct reflection rather than a placeholder.
 *
 * The three counts sit between the acceleration-structure count and the name pointers; the
 * twelve pointers sit at the end. Order is fixed by FfxShaderBlob.
 */
#undef POPULATE_SHADER_BLOB_FFX
#define POPULATE_SHADER_BLOB_FFX(info, index)                                                    \
    {                                                                                            \
        info[index].blobData,                                                                    \
        info[index].blobSize,                                                                    \
        info[index].numConstantBuffers,                                                          \
        info[index].numSRVTextures,                                                              \
        info[index].numUAVTextures,                                                              \
        info[index].numSRVBuffers,                                                               \
        info[index].numUAVBuffers,                                                               \
        info[index].numSamplers,                                                                 \
        info[index].numRTAccelerationStructures,                                                 \
        0u,                                                                                      \
        0u,                                                                                      \
        0u,                                                                                      \
        info[index].constantBufferNames,                                                         \
        info[index].constantBufferBindings,                                                      \
        info[index].constantBufferCounts,                                                        \
        info[index].constantBufferSpaces,                                                        \
        info[index].srvTextureNames,                                                             \
        info[index].srvTextureBindings,                                                          \
        info[index].srvTextureCounts,                                                            \
        info[index].srvTextureSpaces,                                                            \
        info[index].uavTextureNames,                                                             \
        info[index].uavTextureBindings,                                                          \
        info[index].uavTextureCounts,                                                            \
        info[index].uavTextureSpaces,                                                            \
        info[index].srvBufferNames,                                                              \
        info[index].srvBufferBindings,                                                           \
        info[index].srvBufferCounts,                                                             \
        info[index].srvBufferSpaces,                                                             \
        info[index].uavBufferNames,                                                              \
        info[index].uavBufferBindings,                                                           \
        info[index].uavBufferCounts,                                                             \
        info[index].uavBufferSpaces,                                                             \
        info[index].samplerNames,                                                                \
        info[index].samplerBindings,                                                             \
        info[index].samplerCounts,                                                               \
        info[index].samplerSpaces,                                                               \
        info[index].rtAccelerationStructureNames,                                                \
        info[index].rtAccelerationStructureBindings,                                             \
        info[index].rtAccelerationStructureCounts,                                               \
        info[index].rtAccelerationStructureSpaces,                                               \
        nullptr, nullptr, nullptr, nullptr,                                                      \
        nullptr, nullptr, nullptr, nullptr,                                                      \
        nullptr, nullptr, nullptr, nullptr                                                       \
    }

#include <ffx_opticalflow_compute_luminance_pyramid_pass_permutations.h>
#include <ffx_opticalflow_compute_luminance_pyramid_pass_wave64_permutations.h>
#include <ffx_opticalflow_compute_luminance_pyramid_pass_16bit_permutations.h>
#include <ffx_opticalflow_compute_luminance_pyramid_pass_wave64_16bit_permutations.h>

#include <ffx_opticalflow_compute_optical_flow_advanced_pass_v5_permutations.h>
#include <ffx_opticalflow_compute_optical_flow_advanced_pass_v5_wave64_permutations.h>
#include <ffx_opticalflow_compute_optical_flow_advanced_pass_v5_16bit_permutations.h>
#include <ffx_opticalflow_compute_optical_flow_advanced_pass_v5_wave64_16bit_permutations.h>

#include <ffx_opticalflow_compute_scd_divergence_pass_permutations.h>
#include <ffx_opticalflow_compute_scd_divergence_pass_wave64_permutations.h>
#include <ffx_opticalflow_compute_scd_divergence_pass_16bit_permutations.h>
#include <ffx_opticalflow_compute_scd_divergence_pass_wave64_16bit_permutations.h>

#include <ffx_opticalflow_filter_optical_flow_pass_v5_permutations.h>
#include <ffx_opticalflow_filter_optical_flow_pass_v5_wave64_permutations.h>
#include <ffx_opticalflow_filter_optical_flow_pass_v5_16bit_permutations.h>
#include <ffx_opticalflow_filter_optical_flow_pass_v5_wave64_16bit_permutations.h>

#include <ffx_opticalflow_generate_scd_histogram_pass_permutations.h>
#include <ffx_opticalflow_generate_scd_histogram_pass_wave64_permutations.h>
#include <ffx_opticalflow_generate_scd_histogram_pass_16bit_permutations.h>
#include <ffx_opticalflow_generate_scd_histogram_pass_wave64_16bit_permutations.h>

#include <ffx_opticalflow_prepare_luma_pass_permutations.h>
#include <ffx_opticalflow_prepare_luma_pass_wave64_permutations.h>
#include <ffx_opticalflow_prepare_luma_pass_16bit_permutations.h>
#include <ffx_opticalflow_prepare_luma_pass_wave64_16bit_permutations.h>

#include <ffx_opticalflow_scale_optical_flow_advanced_pass_v5_permutations.h>
#include <ffx_opticalflow_scale_optical_flow_advanced_pass_v5_wave64_permutations.h>
#include <ffx_opticalflow_scale_optical_flow_advanced_pass_v5_16bit_permutations.h>
#include <ffx_opticalflow_scale_optical_flow_advanced_pass_v5_wave64_16bit_permutations.h>


#include <string.h> // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key)   \
    key.index                                   = 0; \
    key.FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT = FFX_CONTAINS_FLAG(options, OPTICALFLOW_HDR_COLOR_INPUT);

static FfxShaderBlob opticalflowGetComputeLuminancePyramidPassPermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_compute_luminance_pyramid_pass_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_luminance_pyramid_pass_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_luminance_pyramid_pass_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_luminance_pyramid_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_luminance_pyramid_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetComputeScdDivergencePassPermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_compute_scd_divergence_pass_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_scd_divergence_pass_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_scd_divergence_pass_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_scd_divergence_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_scd_divergence_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetGenerateScdHistogramPassPermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_generate_scd_histogram_pass_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_generate_scd_histogram_pass_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_generate_scd_histogram_pass_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_generate_scd_histogram_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_generate_scd_histogram_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetPrepareLumaPassPermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_prepare_luma_pass_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_prepare_luma_pass_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_prepare_luma_pass_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_prepare_luma_pass_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_prepare_luma_pass_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetComputeOpticalFlowAdvancedPassV5PermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_compute_optical_flow_advanced_pass_v5_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_optical_flow_advanced_pass_v5_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_optical_flow_advanced_pass_v5_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_compute_optical_flow_advanced_pass_v5_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_compute_optical_flow_advanced_pass_v5_PermutationInfo, tableIndex);
    }
}
 
static FfxShaderBlob opticalflowGetFilterOpticalFlowPassV5PermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_filter_optical_flow_pass_v5_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_filter_optical_flow_pass_v5_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_filter_optical_flow_pass_v5_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_filter_optical_flow_pass_v5_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_filter_optical_flow_pass_v5_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetScaleOpticalFlowAdvancedPassV5PermutationBlobByIndex(uint32_t permutationOptions, bool isWave64, bool)
{
    ffx_opticalflow_scale_optical_flow_advanced_pass_v5_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    if (isWave64)
    {
        const int32_t tableIndex = g_ffx_opticalflow_scale_optical_flow_advanced_pass_v5_wave64_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scale_optical_flow_advanced_pass_v5_wave64_PermutationInfo, tableIndex);
    }
    else
    {
        const int32_t tableIndex = g_ffx_opticalflow_scale_optical_flow_advanced_pass_v5_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scale_optical_flow_advanced_pass_v5_PermutationInfo, tableIndex);
    }
}

FfxErrorCode opticalflowGetPermutationBlobByIndex(
    FfxOpticalflowPass passId,
    uint32_t permutationOptions,
    FfxShaderBlob* outBlob,
    FfxShaderBlob* outVertBlob) {

    if (outVertBlob != nullptr)
    {
        memset(outVertBlob, 0, sizeof(FfxShaderBlob));
    }

    bool isWave64 = FFX_CONTAINS_FLAG(permutationOptions, OPTICALFLOW_SHADER_PERMUTATION_FORCE_WAVE64);
    bool is16bit = FFX_CONTAINS_FLAG(permutationOptions,  OPTICALFLOW_SHADER_PERMUTATION_ALLOW_FP16);

    switch (passId) {

        case FFX_OPTICALFLOW_PASS_PREPARE_LUMA:
        {
            FfxShaderBlob blob = opticalflowGetPrepareLumaPassPermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }

        case FFX_OPTICALFLOW_PASS_GENERATE_OPTICAL_FLOW_INPUT_PYRAMID:
        {
            FfxShaderBlob blob = opticalflowGetComputeLuminancePyramidPassPermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }

        case FFX_OPTICALFLOW_PASS_GENERATE_SCD_HISTOGRAM:
        {
            FfxShaderBlob blob = opticalflowGetGenerateScdHistogramPassPermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }

        case FFX_OPTICALFLOW_PASS_COMPUTE_SCD_DIVERGENCE:
        {
            FfxShaderBlob blob = opticalflowGetComputeScdDivergencePassPermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }

        case FFX_OPTICALFLOW_PASS_COMPUTE_OPTICAL_FLOW_ADVANCED_V5:
        {
            FfxShaderBlob blob = opticalflowGetComputeOpticalFlowAdvancedPassV5PermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }
        
        case FFX_OPTICALFLOW_PASS_FILTER_OPTICAL_FLOW_V5:
        {
            FfxShaderBlob blob = opticalflowGetFilterOpticalFlowPassV5PermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }
        
        case FFX_OPTICALFLOW_PASS_SCALE_OPTICAL_FLOW_ADVANCED_V5:
        {
            FfxShaderBlob blob = opticalflowGetScaleOpticalFlowAdvancedPassV5PermutationBlobByIndex(permutationOptions, isWave64, is16bit);
            memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
            return FFX_OK;
        }

        default:
            FFX_ASSERT_FAIL("Should never reach here.");
            break;
    }

    // return an empty blob
    memset(outBlob, 0, sizeof(FfxShaderBlob));
    return FFX_OK;
}

FfxErrorCode opticalflowIsWave64(uint32_t permutationOptions, bool& isWave64)
{
    isWave64 = FFX_CONTAINS_FLAG(permutationOptions, OPTICALFLOW_SHADER_PERMUTATION_FORCE_WAVE64);
    return FFX_OK;
}
