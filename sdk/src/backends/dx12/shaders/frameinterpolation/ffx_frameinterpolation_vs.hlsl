// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// D3D12 mirror of vk/shaders/frameinterpolation/ffx_frameinterpolation_vs.glsl.
//
// The GLSL original is self-contained -- no bindings, no includes, no algorithm --
// but it is written in GLSL's spelling of the portable types, so it includes the
// callbacks header exactly as the other entry points do, for the FfxFloat32*/
// FfxUInt32 names and FFX_PARAMETER_OUT. It defines no BIND_* macro, so the
// callbacks header compiles to nothing but the type names and the shim.
//
// The GLSL full-screen triangle is generated from gl_VertexIndex; the D3D12
// equivalent is SV_VertexID, which is why there is no vertex buffer binding here.
// Geometry and winding are unchanged, so this vertex shader is pair-compatible with
// the Vulkan one's triangle.

#include "frameinterpolation/ffx_frameinterpolation_callbacks_hlsl.h"

void VS(FfxUInt32 iVertexId : SV_VertexID, FFX_PARAMETER_OUT FfxFloat32x4 oPosition : SV_Position, FFX_PARAMETER_OUT FfxFloat32x2 oUv : TEXCOORD0)
{
    FfxFloat32x2 uv = FfxFloat32x2(FfxFloat32(iVertexId & 1u), FfxFloat32(iVertexId >> 1u)) * 2.0f;
    oPosition       = FfxFloat32x4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    oUv             = uv;
}
