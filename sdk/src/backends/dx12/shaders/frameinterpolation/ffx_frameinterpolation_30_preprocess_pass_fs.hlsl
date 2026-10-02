// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_30_preprocess_pass_fs.glsl.

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1        0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1          1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE 2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE  3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1               4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1               5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1               6
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1               7

//-----------------------------------------------------------------------------
// Output: UAV bindings
//
// preprocess() writes the tensor through StoreInTensor, which in GLSL has a
// storage-buffer / ARM-tensor form rather than a render-target form, so this pixel
// shader variant keeps the UAV and has no SV_Target. It is retained because the
// Vulkan list compiles it and the D3D12 list mirrors that list.
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR 8

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION 9

#include "frameinterpolation/ffx_frameinterpolation_30_preprocess_pass.h"

// entry-point
void PS(FfxFloat32x4 iFragCoord : SV_Position)
{
    preprocess(FfxInt32x2(iFragCoord.xy));
}
