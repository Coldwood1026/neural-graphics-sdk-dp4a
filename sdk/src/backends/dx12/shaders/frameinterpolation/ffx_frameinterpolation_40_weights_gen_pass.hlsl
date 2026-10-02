// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_40_weights_gen_pass.glsl.
//
// The GLSL original contains its own entry point: this pass is not a call into a
// shared algorithm header, the whole pass body lives in the entry file. It is
// transcribed here rather than reimplemented.
//
// The GLSL text is GLSL, so the transcription is mechanical: int32_t2/float2/uint32_t
// resolve through the same shim the callbacks header installs, gl_GlobalInvocationID
// becomes SV_DispatchThreadID, and the `#include "frameinterpolation/
// ffx_frameinterpolation_callbacks_glsl.h"` / common.h pair becomes the HLSL pair.

//-----------------------------------------------------------------------------
// Input: SRV bindings
//
// Declared before the data-graph split so the no-op stub below still has the
// callbacks header (for FfxUInt32x2) and the thread-group macros in scope. The GLSL
// original reaches both only on the non-data-graph path, because glslang's two-pass
// include model lets it; HLSL needs them declared before use.
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR 0

//-----------------------------------------------------------------------------
// Output: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR 1

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 2

#include "frameinterpolation/ffx_frameinterpolation_callbacks_hlsl.h"
#include "frameinterpolation/ffx_frameinterpolation_common.h"

#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS \
    [numthreads(FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH)]
#endif  // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

#if FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI
// Shouldn't access this function.
//
// The GLSL original is a bare `void main() { return; }`, which glslang accepts. In
// HLSL a compute entry point without [numthreads] is a hard error ("compute entry
// point must have a valid numthreads attribute"), so the attribute is declared above
// and applied here too. The body is still a no-op; the real work runs on the
// data-graph path.
FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void CS(FfxUInt32x2 iGlobalId : SV_DispatchThreadID)
{
    return;
}
#else

// Reverse of WriteToInputTensor()
// Input:  tensor : InputTensorElement_t
// Output: the original fields used to write it

void DecodeFromInputTensor(FFX_PARAMETER_IN InputTensorElement_t tensor,
                           FFX_PARAMETER_OUT FFX_MIN16_F3          rgb_mv_m1,
                           FFX_PARAMETER_OUT FFX_MIN16_F3          rgb_mv_p1,
                           FFX_PARAMETER_OUT FFX_MIN16_F3          rgb_mv2_m1,
                           FFX_PARAMETER_OUT FFX_MIN16_F3          rgb_of_m1,
                           FFX_PARAMETER_OUT FFX_MIN16_F3          rgb_of_p1,
                           FFX_PARAMETER_OUT FFX_MIN16_F2          dis_mask,
                           FFX_PARAMETER_OUT FFX_MIN16_F           qDepth)  // what WriteToInputTensor stored from `depth`
{
    rgb_mv_m1 = FFX_MIN16_F3(tensor.rgbM1MV_rP1MV.rgb);

    rgb_mv_p1 = FFX_MIN16_F3(tensor.rgbM1MV_rP1MV.a, tensor.gbP1MV_rgM1OF.r, tensor.gbP1MV_rgM1OF.g);

    rgb_of_m1 = FFX_MIN16_F3(tensor.gbP1MV_rgM1OF.b, tensor.gbP1MV_rgM1OF.a, tensor.bM1OF_rgbP1OF.r);

    // The GLSL original is FFX_MIN16_F3(0), a scalar broadcast. HLSL has no
    // scalar-to-vector constructor broadcast, so all three components are explicit.
    rgb_mv2_m1 = FFX_MIN16_F3(0, 0, 0);  // TODO: how to set this value

    rgb_of_p1 = FFX_MIN16_F3(tensor.bM1OF_rgbP1OF.g, tensor.bM1OF_rgbP1OF.b, tensor.bM1OF_rgbP1OF.a);

    qDepth = FFX_MIN16_F(tensor.depth_disMask.r);  // TODO: check this value

    dis_mask = FFX_MIN16_F2(tensor.depth_disMask.b, tensor.depth_disMask.a);
}

FFX_MIN16_F CalcDistance(FFX_MIN16_F3 a, FFX_MIN16_F3 b)
{
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
}

#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS \
    [numthreads(FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH)]
#endif  // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void CS(FfxUInt32x2 iGlobalId : SV_DispatchThreadID)
{
    FfxInt32x2 output_pixel = FfxInt32x2(iGlobalId);
    FfxInt32x2 renderSize   = RenderSize();
    if (any(greaterThanEqual(output_pixel, renderSize)))
    {
        return;
    }

    InputTensorElement_t tensor;
    tensor.rgbM1MV_rP1MV = LoadInTensorData0(FfxUInt32x2(output_pixel));
    tensor.gbP1MV_rgM1OF = LoadInTensorData1(FfxUInt32x2(output_pixel));
    tensor.bM1OF_rgbP1OF = LoadInTensorData2(FfxUInt32x2(output_pixel));
    tensor.depth_disMask = LoadInTensorData3(FfxUInt32x2(output_pixel));

    FFX_MIN16_F3 rgb_mv_m1;
    FFX_MIN16_F3 rgb_mv_p1;
    FFX_MIN16_F3 rgb_mv2_m1;
    FFX_MIN16_F3 rgb_of_m1;
    FFX_MIN16_F3 rgb_of_p1;
    FFX_MIN16_F2 dis_mask;
    FFX_MIN16_F  qDepth;

    // Decode back into original components
    DecodeFromInputTensor(tensor, rgb_mv_m1, rgb_mv_p1, rgb_mv2_m1, rgb_of_m1, rgb_of_p1, dis_mask, qDepth);

    // Compute distances & consistencies
    FFX_MIN16_F dPrev_MV_OF  = CalcDistance(rgb_mv_m1, rgb_of_m1);
    FFX_MIN16_F dNext_MV_OF  = CalcDistance(rgb_mv_p1, rgb_of_p1);
    FFX_MIN16_F dPrev_MV_MV2 = CalcDistance(rgb_mv_m1, rgb_mv2_m1);
    FFX_MIN16_F dCross       = CalcDistance(FFX_MIN16_F3(rgb_mv_m1 + rgb_of_m1) * FFX_MIN16_F(0.5f), FFX_MIN16_F3(rgb_mv_p1 + rgb_of_p1) * FFX_MIN16_F(0.5f));

    FFX_MIN16_F cPrev = FFX_MIN16_F(1.0f) / (FFX_MIN16_F(1e-3) + FFX_MIN16_F(0.5) * dPrev_MV_OF + FFX_MIN16_F(0.5) * dPrev_MV_MV2);
    FFX_MIN16_F cNext = FFX_MIN16_F(1.0f) / (FFX_MIN16_F(1e-3) + dNext_MV_OF);

    // Disocclusion / Invalid-Motion Penalty
    FFX_MIN16_F m         = clamp(max(dis_mask.x, dis_mask.y), FFX_MIN16_F(0.0f), FFX_MIN16_F(1.0f));
    FFX_MIN16_F trustMask = FFX_MIN16_F(1.0f) - m;

    // Depth-Based Temporal Trust
    const FFX_MIN16_F kNear = FFX_MIN16_F(0.15f);
    const FFX_MIN16_F kFar  = FFX_MIN16_F(0.85f);
    FfxFloat32        tDepth = smoothstep(kNear, kFar, clamp(qDepth, FFX_MIN16_F(0.0f), FFX_MIN16_F(1.0f)));

    // Direction Split (Prev vs Next)
    FfxFloat32 confPrev = cPrev * trustMask * tDepth;
    FfxFloat32 confNext = cNext * trustMask * tDepth;

    FfxFloat32 crossBoost = 1.0 / (1e-3 + dCross);
    confPrev *= (1.0 + 0.5 * cPrev * crossBoost);
    confNext *= (1.0 + 0.5 * cNext * crossBoost);

    FfxFloat32 dirSum = max(confPrev + confNext, 1e-8);
    FfxFloat32 Sprev  = confPrev / dirSum;
    FfxFloat32 Snext  = confNext / dirSum;

    // Method Split (MV vs OF)
    // For OF
    FFX_MIN16_F3 prevAvg = FFX_MIN16_F3(rgb_mv_m1 + rgb_of_m1) * FFX_MIN16_F(0.5f);
    FfxFloat32   ePrevMV  = 1.0 / (1e-3 + CalcDistance(rgb_mv_m1, prevAvg));
    FfxFloat32   ePrevOF  = 1.0 / (1e-3 + CalcDistance(rgb_of_m1, prevAvg));
    FfxFloat32   mv2Agree = 1.0 / (1e-3 + dPrev_MV_MV2);
    ePrevMV *= (1.0 + 0.25 * mv2Agree);
    FfxFloat32 Pmv = ePrevMV / (ePrevMV + ePrevOF + 1e-8);
    FfxFloat32 Pof = ePrevOF / (ePrevMV + ePrevOF + 1e-8);
    // For MVs
    FFX_MIN16_F3 nextAvg = FFX_MIN16_F3(rgb_mv_p1 + rgb_of_p1) * FFX_MIN16_F(0.5f);
    FfxFloat32   eNextMV  = 1.0 / (1e-3 + CalcDistance(rgb_mv_p1, nextAvg));
    FfxFloat32   eNextOF  = 1.0 / (1e-3 + CalcDistance(rgb_of_p1, nextAvg));
    FfxFloat32   eNextSum = max(eNextMV + eNextOF, 1e-8);
    FfxFloat32   Nmv      = eNextMV / eNextSum;  // "next MV" share
    FfxFloat32   Nof      = eNextOF / eNextSum;  // "next OF" share

    // Final Weights & Safety Rails
    FfxFloat32 wPrevMV = Sprev * Pmv;
    FfxFloat32 wPrevOF = Sprev * Pof;
    FfxFloat32 wNextMV = Snext * Nmv;
    FfxFloat32 wNextOF = Snext * Nof;

    // disocclusion hard fail
    if (m > 0.9)
    {
        wPrevMV = wPrevOF = wNextMV = 0.0;
        wNextOF = 1.0;
    }

    FfxFloat32 sumW = wPrevMV + wPrevOF + wNextMV + wNextOF;
    // fallback
    if (sumW < 1e-6)
    {
        wPrevMV = 0.0;
        wPrevOF = 0.5;
        wNextMV = 0.0;
        wNextOF = 0.5;
        sumW    = 1.0;
    }

    // Normalize weights
    FfxFloat32 inv = 1.0 / sumW;
    wPrevMV *= inv;
    wPrevOF *= inv;
    wNextMV *= inv;
    wNextOF *= inv;

    FFX_MIN16_F4 weights = FFX_MIN16_F4(wPrevMV, wNextMV, wPrevOF, wNextOF);

    // Store the weights
    StoreOutParamsTensor(output_pixel, weights);
}

#endif
