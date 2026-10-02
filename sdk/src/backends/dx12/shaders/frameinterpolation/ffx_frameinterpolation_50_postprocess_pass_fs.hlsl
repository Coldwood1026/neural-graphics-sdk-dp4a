// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_50_postprocess_pass_fs.glsl.

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR             0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1        1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1          2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TM1        3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE 4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE  5

//-----------------------------------------------------------------------------
// Input: RT bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT 0
// Additional render target used when the SDK (not the application) owns the
// previous-interpolation-source ping-pong buffer; the postprocess pass writes it
// here so it can be sampled as colorTm1 next frame.
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE 1
#endif

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 6

// See the note in ffx_frameinterpolation_50_postprocess_pass.hlsl: postprocess()
// calls Dequantize(), which lives in the common header, so that has to precede the
// shared per-pass header under HLSL's single-pass include model.
#include "frameinterpolation/ffx_frameinterpolation_50_postprocess_pass.h"

// entry-point
FrameInterpolationOutput_t PS(FfxFloat32x4 iFragCoord : SV_Position)
{
    postprocess(FfxInt32x2(iFragCoord.xy));
    return g_GFIOutput;
}
