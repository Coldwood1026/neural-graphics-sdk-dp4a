// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_10_warp_flow_pass_fs.glsl.

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1       0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1        1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1        2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1 3

//-----------------------------------------------------------------------------
// Output: RT bindings + UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1 4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1             5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1             6
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_DEPTH
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT   0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1 1
#else
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1 0
#endif

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 8

#include "frameinterpolation/ffx_frameinterpolation_10_warp_flow_pass.h"

// entry-point
FrameInterpolationOutput_t PS(FfxFloat32x4 iFragCoord : SV_Position)
{
    warp_flow(FfxInt32x2(iFragCoord.xy));
    return g_GFIOutput;
}
