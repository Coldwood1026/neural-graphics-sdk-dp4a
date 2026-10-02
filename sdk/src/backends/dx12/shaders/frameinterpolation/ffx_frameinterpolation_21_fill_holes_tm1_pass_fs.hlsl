// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_21_fill_holes_tm1_pass_fs.glsl.

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1 0

//-----------------------------------------------------------------------------
// Output: RT bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1 0

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 1

#include "frameinterpolation/ffx_frameinterpolation_21_fill_holes_tm1_pass.h"

// entry-point
FrameInterpolationOutput_t PS(FfxFloat32x4 iFragCoord : SV_Position)
{
    fill_holes(FfxInt32x2(iFragCoord.xy));
    return g_GFIOutput;
}
