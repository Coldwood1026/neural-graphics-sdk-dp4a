// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_01_downsample_of_colour_pass_fs.glsl.

// SRV bindings
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE 0

// Render-target binding. In HLSL the render-target variant of StoreColourP1Internal
// writes this shader's SV_Target instead of a UAV, so this define is what selects the
// render-target form in the callbacks header.
#define FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL 0

// CB binding
#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 1

#include "frameinterpolation/ffx_frameinterpolation_01_downsample_of_colour_pass.h"

FrameInterpolationOutput_t PS(FfxFloat32x4 iFragCoord : SV_Position)
{
    downsample_of_colour(FfxInt32x2(iFragCoord.xy));
    return g_GFIOutput;
}
