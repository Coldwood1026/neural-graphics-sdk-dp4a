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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// -----------------------------------------------------------------------------
// Arm's neural frame interpolation -- D3D12 / HLSL half.
//
// This is the HLSL twin of ffx_frameinterpolation_callbacks_glsl.h. It was
// produced by taking AMD FidelityFX SDK 1.1.3's
// ffx_frameinterpolation_callbacks_hlsl.h as the structural template -- that file
// already encodes AMD's own GLSL->HLSL mapping for every abstraction this effect
// needs -- and applying to it the per-symbol delta that distinguishes Arm's GLSL
// callbacks from AMD's GLSL callbacks. See the accompanying report for the diff
// summary.
//
// Three things in here are not a straight transliteration, and each is called out
// where it occurs:
//
//   * The GLSL-name compatibility shim immediately below. Arm's *shared* algorithm
//     headers (ffx_frameinterpolation_common.h, fill_holes_common.h,
//     warp_flow_common.h, quant.h and the per-pass headers) are written in
//     GLSL-flavoured portable syntax and use GLSL primitive names -- int32_t2,
//     uint32_t, float2/3/4, half, half2. HLSL has no such names, so they are mapped
//     here. Without this the shared headers do not compile under dxc at all.
//
//   * The int8 tensor load/store. DXIL has no 8-bit scalar or vector type; see the
//     block comment at FFX_ARM_FRAMEINTERPOLATION_TENSOR_STRIDE.
//
//   * hf. Arm's ffx_frameinterpolation_debug_view.h writes half literals as
//     `0.4hf`, which is GLSL's f16 suffix and is not HLSL syntax.
// -----------------------------------------------------------------------------

#ifndef FFX_ARM_FRAMEINTERPOLATION_CALLBACKS_HLSL_H
#define FFX_ARM_FRAMEINTERPOLATION_CALLBACKS_HLSL_H

#include "ffx_frameinterpolation_resources.h"

// =============================================================================
// GLSL-name compatibility shim -- part 1, which must precede ffx_core.h.
//
// ffx_core.h -> ffx_common_types.h defines the GLSL spellings (int32_t2,
// uint32_t2, ...) in terms of *these* names when FFX_GLSL is set, and
// ffx_core_hlsl.h is written against the same spellings. An HLSL translation unit
// of Arm's shared headers therefore needs the names to exist before ffx_core.h is
// reached, not after; part 2 of the shim follows it.
//
// Every mapping here is the one ffx_common_types.h's FFX_GLSL block also gives, so
// the two backends agree name-for-name.
// =============================================================================

// Scalars and the uint32_t/int32_t families need no help here: dxc predefines
// int32_t, int32_t2/3/4, uint32_t and uint32_t2/3/4 as HLSL builtins (verified), and
// ffx_common_types.h's FFX_HLSL branch builds FfxInt32/FfxUInt32 on top of
// int32_t/uint32_t. Defining uint32_t here would shadow that builtin and break
// ffx_common_types.h's own typedefs, so it is deliberately absent.

// -----------------------------------------------------------------------------
// The 8-bit integer names.
//
// These must exist as distinct names because quant.h and
// ffx_frameinterpolation_common.h declare overload sets on them (Quantize /
// Dequantize) and the tensor element struct is built from int8_t4 -- but they are
// NOT aliased to a native 8-bit type, because DXIL has none: dxc 1.8 rejects
// `int8_t`, `int8_t4`, `int16_t` and `int16_t`-family types outright, and
// -enable-16bit-types (which is what float16_t needs) deliberately still excludes
// 8-bit. That was verified directly; see the report.
//
// Aliasing to the 32-bit vector type keeps the geometry the algorithm is written
// against (four lanes per int8_t4, sixteen lanes per tensor element) and keeps every
// arithmetic result value-identical, because the algorithm's int8 values all lie in
// [-128,127]. The narrowing back to 8 bits happens exactly once, at the tensor
// store, where StoreInTensor masks each lane to its low byte.
// -----------------------------------------------------------------------------
// The aliases are written against dxc's *builtin* spellings (int32_t, int32_t4),
// not against FfxInt32/FfxInt32x4. Both are the same type under FFX_HLSL_SM>=62, but
// only the builtin spelling is accepted by dxc when it arrives through macro
// expansion in a type position: `#define int8_t FfxInt32` resolves to the
// ffx_common_types.h typedef and dxc then reports "unknown type name 'int8_t'",
// whereas `#define int8_t int32_t` is accepted. Verified both ways.
#define int8_t  int32_t
#define int8_t2 int32_t2
#define int8_t3 int32_t3
#define int8_t4 int32_t4

// -----------------------------------------------------------------------------
// ffx_core_hlsl.h spells the comparison macros as bare HLSL operators
// (`#define FFX_GREATER_THAN_EQUAL(x, y) x >= y`) while Arm's shared headers use
// them the GLSL way, on vectors: ffx_frameinterpolation_common.h's IsOnScreen does
//
//     all(FFX_GREATER_THAN_EQUAL(FfxUInt32x2(pos), FfxUInt32x2(0)))
//
// In GLSL a vector comparison yields a bvec and `any`/`all` consume it. Under HLSL
// `vec >= vec` does yield a bool vector, so the operator itself is fine -- but the
// earlier fix to the scalar operand is what makes the expression well-formed, and
// being explicit here means the comparison macros cannot silently degrade into
// scalar comparisons.
//
// These are #undef'd and redefined *after* ffx_core.h below, because ffx_core_hlsl.h
// defines them itself and a definition placed before that include would simply be
// overwritten (the last definition wins).
// -----------------------------------------------------------------------------
#define int8_t  int32_t
#define int8_t2 int32_t2
#define int8_t3 int32_t3
#define int8_t4 int32_t4

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif  //__hlsl_dx_compiler
#include "ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif  //__hlsl_dx_compiler

// Override the comparison macros now that ffx_core_hlsl.h has had its say.
#undef FFX_LESS_THAN
#undef FFX_GREATER_THAN
#undef FFX_GREATER_THAN_EQUAL
#undef FFX_LESS_THAN_EQUAL
#undef FFX_EQUAL
#undef FFX_NOT_EQUAL
#define FFX_LESS_THAN(x, y)          ((x) < (y))
#define FFX_GREATER_THAN(x, y)       ((x) > (y))
#define FFX_GREATER_THAN_EQUAL(x, y) ((x) >= (y))
#define FFX_LESS_THAN_EQUAL(x, y)    ((x) <= (y))
#define FFX_EQUAL(x, y)              ((x) == (y))
#define FFX_NOT_EQUAL(x, y)          ((x) != (y))
#endif  // #if defined(FFX_GPU)

#if defined(FFX_GPU)

// =============================================================================
// GLSL-name compatibility shim -- part 2.
//
// The shared algorithm headers in this directory are shared with the Vulkan backend
// and are written against GLSL's spelling of the portable types. AMD never hit this
// because AMD's GLSL and HLSL shared headers are separate files; Arm's are not.
// This is the one place the mapping lives.
// =============================================================================

// half2/3/4. float2/3/4 and int32_t2/3/4 already exist: the former are HLSL
// natives, the latter are #defined by ffx_common_types.h's FFX_GLSL block, which is
// active in an HLSL translation unit too. `half` itself is an HLSL native.
#define half2 FFX_MIN16_F2
#define half3 FFX_MIN16_F3
#define half4 FFX_MIN16_F4

// GLSL matrix spelling. ffx_frameinterpolation_10_warp_flow_pass.h's
// CalculateCameraMotion takes a `mat4`, which HLSL does not name; the SDK's
// FfxFloat32Mat4 is exactly that (float4x4) under FFX_HLSL.
#define mat4 FfxFloat32Mat4

// GLSL intrinsics and spelling used by the shared algorithm headers.
#define uintBitsToFloat(x) asfloat(x)
#define floatBitsToUint(x) asuint(x)

// GLSL findMSB == HLSL firstbithigh. (dxc 1.8 has no `firstbit_hi`; that spelling
// errors as an undeclared identifier, while `firstbithigh` compiles.)
// quant.h calls this on a non-negative int, where the two agree exactly, except at
// zero: both return -1, and quant.h clamps to 0 immediately after.
#define findMSB(x) firstbithigh(x)

bool2 greaterThan(FfxInt32x2 a, FfxInt32x2 b) { return a > b; }
bool2 greaterThanEqual(FfxInt32x2 a, FfxInt32x2 b) { return a >= b; }
bool2 lessThan(FfxInt32x2 a, FfxInt32x2 b) { return a < b; }
bool2 lessThanEqual(FfxInt32x2 a, FfxInt32x2 b) { return a <= b; }
bool2 equal(FfxInt32x2 a, FfxInt32x2 b) { return a == b; }
bool2 notEqual(FfxInt32x2 a, FfxInt32x2 b) { return a != b; }

// GLSL `mix` is HLSL `lerp`.
#define mix(a, b, t) lerp(a, b, t)

// GLSL integer-conversion spellings. `uint8_t` is not an HLSL name; the dynamic
// mask in ffx_frameinterpolation_10_warp_flow_pass.h is a 0/1 flag whose value range
// fits a byte, so it is carried as the 32-bit equivalent for the same reason
// int8_t is (see above).
#define uint8_t uint32_t

// GLSL's isnan/isinf take a vector and yield a bool vector; so do HLSL's.
// GLSL `any`/`all` on a bool vector are identical in HLSL. GLSL `sign` is HLSL
// `sign`. All four are used by ffx_frameinterpolation_10_warp_flow_pass.h.
// Nothing to define for any of them -- listed here so the set of GLSL spellings this
// effect relies on is enumerable in one place.

// GLSL's half-precision literal suffixes. `0.4hf` / `0.HF` tokenise as the literal
// followed by an identifier, so expanding the identifier to nothing makes them parse
// as plain float literals -- the right value, at the precision the permutations
// actually use (FFX_HALF=0 widens everything to fp32 anyway).
#define hf
#define HF

// GLSL ES precision qualifiers are not HLSL syntax.
#define highp
#define mediump
#define lowp

#define COUNTER_SPD                          0
#define COUNTER_FRAME_INDEX_SINCE_LAST_RESET 1

// -----------------------------------------------------------------------------
// The inference input tensor.
//
// Arm's GLSL carries the inference input as sixteen signed 8-bit values per
// element, laid out as four int8_t4 members, and gets that from
// GL_EXT_shader_8bit_storage + GL_EXT_shader_explicit_arithmetic_types_int8. It has
// two storage flavours selected by FFX_ARM_FRAMEINTERPOLATION_OPTION_SUPPORT_TENSOR:
// a GL_ARM_tensors `tensorARM<int8_t,4>` (data-graph path) or a std430 storage
// buffer of InputTensorElement_t.
//
// DXIL has neither. What it does have, verified against dxc 1.8:
//
//   * No native 8-bit scalar or vector type at all -- `int8_t`, `int8_t4`, `int16_t`
//     are "unknown type name". `-enable-16bit-types` enables float16_t/int16_t and
//     still not 8-bit.
//   * RWByteAddressBuffer / ByteAddressBuffer compile fine and give byte-exact
//     control, but every access needs explicit masking and sign extension.
//   * Typed R8_SINT views work: Texture2D<int> / RWTexture2D<int> and
//     StructuredBuffer<int> all compile, but a structured buffer of an 8-bit vector
//     does not (int8_t4 is not a legal struct member type), so a structured-buffer
//     representation would have had to widen the host-side buffer 4x.
//
// The representation chosen is: a ByteAddressBuffer / RWByteAddressBuffer holding
// the GLSL std430 layout verbatim -- sixteen int8 per element, no padding, element
// stride FFX_ARM_FRAMEINTERPOLATION_TENSOR_STRIDE -- with each 4-int8 group loaded
// as one packed 32-bit word and sign-extended per lane on read. That is what
// InputTensorElement_t below is: four FfxInt32x4, one per packed word, where
// component N of a member holds lane N's int8 value (already sign-extended, so
// within [-128,127]).
//
// Why this and not the alternatives:
//
//   * It is the only representation that keeps the on-GPU element size at 16 bytes,
//     so the host's tensor allocation, the dp4a inference backend
//     (sdk/src/backends/shared/nfru_dp4a, which is already int8 and already
//     API-portable) and this shader all agree on the byte layout. Widening to
//     32 bits per lane would have silently changed the tensor ABI that the inference
//     graph is written against.
//   * It works without -enable-16bit-types, so it is identical in the FP32 and FP16
//     permutations.
//
// The cost is explicit unpacking, confined to TensorLoadElement / TensorStoreElement
// below, and one narrowing to 8 bits at the store.
// -----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_TENSOR_STRIDE 16u  // 16 int8 per element

struct InputTensorElement_t
{
    // Each member is one packed 32-bit word holding four int8 lanes, already
    // sign-extended on load. Component N == lane N, which is exactly the indexing
    // Arm's GLSL does on InputTensorElement_t's int8_t4 members.
    FFX_MIN16_I4 rgbM1MV_rP1MV;  // rgb_m1_warp_t_mv.rgb, rgb_p1_warp_t_mv.r
    FFX_MIN16_I4 gbP1MV_rgM1OF;  // rgb_p1_warp_t_mv.gb, rgb_m1_warp_t_flow.rg
    FFX_MIN16_I4 bM1OF_rgbP1OF;  // rgb_m1_warp_t_flow.b, rgb_p1_warp_t_flow.rgb
    FFX_MIN16_I4 depth_disMask;  // depth_m1_warp_t_mv_norm, depth_p1_warp_t_mv_norm, dis_mask_m1, dis_mask_p1
};

// Names the shared headers use for "one int8" and "one int8_t4". tensor_t is a
// scalar lane value; tensorVec_t is the four-lane group. Both are 32-bit carriers;
// see the block comment above for why the width cannot be 8 here. As with the int8_t
// aliases, the builtin spellings are what dxc accepts in a type position.
#define tensor_t    int32_t
#define tensorVec_t int32_t4

// =============================================================================
// Root signature and constant buffers.
//
// Binding macro names are Arm's (FFX_ARM_FRAMEINTERPOLATION_BIND_*), matching Arm's
// GLSL callbacks. The register numbers come straight from the caller's binding
// values, exactly as AMD's HLSL callbacks do: the D3D12 backend builds each
// pipeline's descriptor tables and CBV slots from dxc's reflection data
// (ffx_dx12.cpp: srvTextureBindings/uavTextureBindings/constantBufferBindings come
// from shaderBlob.boundSRVTextures/boundUAVTextures/boundConstantBuffers), not from
// a hand-written root signature. Nothing here needs a fixed offset as a result.
// =============================================================================

#define FFX_FRAMEINTERPOLATION_ROOTSIG_STRINGIFY(p) FFX_FRAMEINTERPOLATION_ROOTSIG_STR(p)
#define FFX_FRAMEINTERPOLATION_ROOTSIG_STR(p)       #p
#define FFX_FRAMEINTERPOLATION_ROOTSIG                                                                                                   \
    [RootSignature("DescriptorTable(UAV(u0, numDescriptors = " FFX_FRAMEINTERPOLATION_ROOTSIG_STRINGIFY(                                \
                       FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT) ")), "                                                       \
                   "DescriptorTable(SRV(t0, numDescriptors = " FFX_FRAMEINTERPOLATION_ROOTSIG_STRINGIFY(                                \
                       FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT) ")), "                                                       \
                   "CBV(b0), "                                                                                                         \
                   "StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, "                                                            \
                   "addressU = TEXTURE_ADDRESS_CLAMP, "                                                                                \
                   "addressV = TEXTURE_ADDRESS_CLAMP, "                                                                                \
                   "addressW = TEXTURE_ADDRESS_CLAMP, "                                                                                \
                   "comparisonFunc = COMPARISON_NEVER, "                                                                               \
                   "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)")]

#if defined(FFX_FRAMEINTERPOLATION_EMBED_ROOTSIG)
#define FFX_FRAMEINTERPOLATION_EMBED_ROOTSIG_CONTENT FFX_FRAMEINTERPOLATION_ROOTSIG
#else
#define FFX_FRAMEINTERPOLATION_EMBED_ROOTSIG_CONTENT
#endif  // #if FFX_FRAMEINTERPOLATION_EMBED_ROOTSIG

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
    cbuffer cbFI : FFX_DECLARE_CB(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
    {
        FfxFloat32x4x4 _MotionTM1ToTP1;
        FfxFloat32x4x4 _MotionTP1ToTM1;

        FfxFloat32x4 _QuantParamsSINT;   ///< Quant parameters between float and sint. .xy for quantize, .zw for dequantize
        FfxFloat32x4 _QuantParamsSNORM;  ///< Quant parameters between float and snorm. .xy for quantize, .zw for dequantize

        FfxFloat32x4 _DeviceToViewDepth;

        FfxInt32x2 _RenderSize;
        FfxInt32x2 _DisplaySize;
        FfxInt32x2 _OfSize;

        FfxFloat32x2 _RenderSizeRcp;
        FfxFloat32x2 _DisplaySizeRcp;
        FfxFloat32x2 _OfSizeRcp;

        FfxFloat32x2 _Jitter;
        FfxFloat32x2 _MotionVectorScale;
        FfxFloat32x2 _MvSimilarityThresholds;

        FfxInt32x2   _InputTensorSize;
        FfxFloat32x2 _InputTensorSizeRcp;
        FfxInt32     _Reset;
        FfxFloat32   _Timestep;
        FfxUInt32    _RandomSeed;
        FfxUInt32    _pad0;

        FfxInt32x2   _MvDepthLaneSize;
        FfxFloat32x2 _MvDepthLaneSizeRcp;
        FfxInt32x2   _FlowLaneSize;
        FfxFloat32x2 _FlowLaneSizeRcp;
        FfxFloat32   _OfGridSizeRcp;
    }

    FfxFloat32x4x4 GetMotionTM1ToTP1()
    {
        return _MotionTM1ToTP1;
    }

    FfxFloat32x4x4 GetMotionTP1ToTM1()
    {
        return _MotionTP1ToTM1;
    }

    FfxFloat32x2 Jitter()
    {
        return _Jitter;
    }

    FFX_MIN16_F2 MotionVectorScale()
    {
        return FFX_MIN16_F2(_MotionVectorScale);
    }

    FFX_MIN16_F MvSimilarityThreshold()
    {
        return FFX_MIN16_F(_MvSimilarityThresholds.x);
    }

    FFX_MIN16_F MvSimilarityNoiseThreshold()
    {
        return FFX_MIN16_F(_MvSimilarityThresholds.y);
    }

    FfxInt32x2 RenderSize()
    {
        return _RenderSize;
    }

    FfxFloat32x2 RenderSizeRcp()
    {
        return _RenderSizeRcp;
    }

    FfxInt32x2 DisplaySize()
    {
        return _DisplaySize;
    }

    FfxFloat32x2 DisplaySizeRcp()
    {
        return _DisplaySizeRcp;
    }

    FfxInt32x2 InputTensorSize()
    {
        return _InputTensorSize;
    }

    FfxFloat32x2 InputTensorSizeRcp()
    {
        return _InputTensorSizeRcp;
    }

    FfxBoolean Reset()
    {
        return _Reset == 1;
    }

    FfxFloat32 Timestep()
    {
        return _Timestep;
    }

    FfxUInt32 RandomSeed()
    {
        return _RandomSeed;
    }

    FfxInt32x2 MvDepthLaneSize()
    {
        return _MvDepthLaneSize;
    }

    FfxFloat32x2 MvDepthLaneSizeRcp()
    {
        return _MvDepthLaneSizeRcp;
    }

    FfxInt32x2 FlowLaneSize()
    {
        return _FlowLaneSize;
    }

    FfxFloat32x2 FlowLaneSizeRcp()
    {
        return _FlowLaneSizeRcp;
    }

    FfxFloat32x4 DeviceToViewSpaceTransformFactors()
    {
        return _DeviceToViewDepth;
    }

    FfxInt32x2 GetOpticalFlowResolution()
    {
        return _OfSize;
    }

    FfxFloat32x2 GetOpticalFlowInvSize()
    {
        return _OfSizeRcp;
    }

    FfxFloat32 OfGridSizeRcp()
    {
        return _OfGridSizeRcp;
    }

    FFX_MIN16_F4 QuantParamsSINT()
    {
        return FFX_MIN16_F4(_QuantParamsSINT);
    }

    FFX_MIN16_F4 QuantParamsSNORM()
    {
        return FFX_MIN16_F4(_QuantParamsSNORM);
    }

    FfxFloat32 ConvertFromDeviceDepthToViewSpace(FfxFloat32 fDeviceDepth)
    {
        const FfxFloat32x4 deviceToViewDepth = DeviceToViewSpaceTransformFactors();
        return deviceToViewDepth[1] / (fDeviceDepth - deviceToViewDepth[0]);
    }

    // Arm's GLSL carries a TODO here about swizzling xy as the AMD reference does.
    // The body is transcribed as-is so the two backends agree.
    FfxFloat32x2 ComputeNdc(FfxFloat32x2 fPxPos, FfxInt32x2 iSize)
    {
        return fPxPos / FfxFloat32x2(iSize) * FfxFloat32x2(2.0f, -2.0f) + FfxFloat32x2(-1.0f, 1.0f);
    }

    FfxFloat32x3 GetViewSpacePosition(FfxInt32x2 iViewportPos, FfxInt32x2 iViewportSize, FfxFloat32 fDeviceDepth)
    {
        const FfxFloat32x4 fDeviceToViewDepth = DeviceToViewSpaceTransformFactors();

        const FfxFloat32 Z = ConvertFromDeviceDepthToViewSpace(fDeviceDepth);

        const FfxFloat32x2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
        const FfxFloat32   X       = fDeviceToViewDepth[2] * fNdcPos.x * Z;
        const FfxFloat32   Y       = fDeviceToViewDepth[3] * fNdcPos.y * Z;

        return FfxFloat32x3(X, Y, Z);
    }

    // GetOpticalFlowSize/GetOpticalFlowSize2 are consumed only by
    // ffx_frameinterpolation_common.h's SampleOpticalFlowMotionVectorField. Arm's
    // nine passes declare no OPTICAL_FLOW_MOTION_VECTOR_FIELD binding, so that
    // function is never instantiated on this side either; they are provided for
    // parity with AMD's HLSL callbacks so the shared header still resolves if a
    // future permutation enables the binding.
    FfxInt32x2 GetOpticalFlowSize()
    {
        return GetOpticalFlowResolution();
    }

    FfxInt32x2 GetOpticalFlowSize2()
    {
        return GetOpticalFlowResolution() * 1;
    }
#endif  // #if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)
    cbuffer cbInpaintingPyramid : FFX_DECLARE_CB(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)
    {
        FfxUInt32   mips;
        FfxUInt32   numWorkGroups;
        FfxUInt32x2 workGroupOffset;
    }

    FfxUInt32 NumMips()
    {
        return mips;
    }
    FfxUInt32 NumWorkGroups()
    {
        return numWorkGroups;
    }
    FfxUInt32x2 WorkGroupOffset()
    {
        return workGroupOffset;
    }
#endif  // #if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)

///////////////////////////////////////////////
// declare samplers
///////////////////////////////////////////////

// One static sampler, exactly as AMD's HLSL callbacks: s0 is linear/clamp with a
// transparent-black border, declared in the root signature above. Arm's GLSL uses
// two sampler objects (s_LinearClamp and s_LinearBorder) whose difference is only
// the border colour; D3D12's static sampler covers both call sites with one object,
// which is why SamplePreviousBackbufferClampBorder and SamplePreviousBackbuffer are
// identical here.
SamplerState s_LinearClamp : register(s0);

///////////////////////////////////////////////
// declare SRVs and SRV accessors
///////////////////////////////////////////////

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUTPUT
    Texture2D<FfxFloat32x4> r_output : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUTPUT);

    FfxFloat32x4 LoadFrameInterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return r_output[iPxInput];
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER
    Texture2D<FfxFloat32x4> r_present_backbuffer : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER);

    FfxFloat32x4 LoadPresentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return r_present_backbuffer[iPxInput];
    }
    FfxFloat32x4 SamplePresentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return r_present_backbuffer.SampleLevel(s_LinearClamp, fUv, 0);
    }
#endif

// ---------------------------------------------------------------------------
// Arm NFRU SRVs
// ---------------------------------------------------------------------------

// GLSL reaches textureSize() for the uv->texel mapping. HLSL has GetDimensions;
// each accessor below uses it rather than duplicating the size in the CB, because
// the lane dimensions are not all present in cbFI (DepthTp1Size/DepthTm1Size are
// the depth surfaces' own sizes, which are not MvDepthLaneSize in every pass).
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1
    Texture2D<FfxFloat32x2> r_motion_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1);

    FfxInt32x2 GFI_MotionTp1Size()
    {
        FfxUInt32 w, h;
        r_motion_tp1.GetDimensions(w, h);
        return FfxInt32x2(w, h);
    }

    FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        FFX_MIN16_F2 motion = FFX_MIN16_F2(r_motion_tp1[iPxPos].xy) * MotionVectorScale();
#if FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS
#error ARM_TODO : Implement Jitter cancellation (Missing uniforms)!
#endif
        // Return Motion in UV space
        return motion;
    }

    FFX_MIN16_F2 LoadInputMotionTp1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
    {
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(GFI_MotionTp1Size())));
        return LoadInputMotionTp1(iPxPos);
    }

    FFX_MIN16_F2 LoadInputMotionTP1WithOffset(FFX_PARAMETER_IN FfxFloat32x2 fUvPos, FfxInt32x2 iOffset)
    {
        FfxInt32x2 iSize  = GFI_MotionTp1Size();
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(iSize)));
        // Keep neighborhood fetch in-bounds after applying iOffset.
        FfxInt32x2   iSamplePx = clamp(iPxPos + iOffset, FfxInt32x2(0, 0), iSize - FfxInt32x2(1, 1));
        FFX_MIN16_F2 motion    = FFX_MIN16_F2(r_motion_tp1[iSamplePx].xy) * MotionVectorScale();
#if FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS
#error ARM_TODO : Implement Jitter cancellation (Missing uniforms)!
#endif
        // Return Motion in UV space
        return motion;
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_TP1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1
    Texture2D<FfxFloat32> r_depth_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1);

    FfxInt32x2 GFI_DepthTp1Size()
    {
        FfxUInt32 w, h;
        r_depth_tp1.GetDimensions(w, h);
        return FfxInt32x2(w, h);
    }

    FfxFloat32 LoadInputDepthTp1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
    {
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(GFI_DepthTp1Size())));
        return r_depth_tp1[iPxPos].x;
    }

    FfxFloat32 LoadInputDepthTp1WithOffset(FFX_PARAMETER_IN FfxFloat32x2 fUvPos, FfxInt32x2 iOffset)
    {
        FfxInt32x2 iSize     = GFI_DepthTp1Size();
        FfxInt32x2 iPxPos    = FfxInt32x2(floor(fUvPos * FfxFloat32x2(iSize)));
        FfxInt32x2 iSamplePx = clamp(iPxPos + iOffset, FfxInt32x2(0, 0), iSize - FfxInt32x2(1, 1));
        return r_depth_tp1[iSamplePx].x;
    }

    FfxFloat32 LoadInputDepthTp1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_depth_tp1[iPxPos].x;
    }

    FfxFloat32x2 DepthTp1Size()
    {
        return FfxFloat32x2(GFI_DepthTp1Size());
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TP1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1
    Texture2D<FfxFloat32x2> r_flow_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1);

    FfxInt32x2 GFI_FlowTm1Size()
    {
        FfxUInt32 w, h;
        r_flow_tm1.GetDimensions(w, h);
        return FfxInt32x2(w, h);
    }

    FFX_MIN16_F2 LoadInputFlowTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
    {
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(GFI_FlowTm1Size())));
        return FFX_MIN16_F2(r_flow_tm1[iPxPos].xy) * FFX_MIN16_F2(GetOpticalFlowInvSize()) * FFX_MIN16_F(OfGridSizeRcp());
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_FLOW_TM1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1
    Texture2D<FfxFloat32> r_depth_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1);

    FfxInt32x2 GFI_DepthTm1Size()
    {
        FfxUInt32 w, h;
        r_depth_tm1.GetDimensions(w, h);
        return FfxInt32x2(w, h);
    }

    FfxFloat32 LoadInputDepthTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
    {
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(GFI_DepthTm1Size())));
        return r_depth_tm1[iPxPos].x;
    }

    FfxFloat32 LoadInputDepthTm1Pixel(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_depth_tm1[iPxPos].x;
    }

    FfxFloat32x2 DepthTm1Size()
    {
        return FfxFloat32x2(GFI_DepthTm1Size());
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH_TM1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1
    Texture2D<FfxFloat32> r_mv_holes_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1);

    FfxInt32x2 GFI_HolesTm1Size()
    {
        FfxUInt32 w, h;
        r_mv_holes_tm1.GetDimensions(w, h);
        return FfxInt32x2(w, h);
    }

    FfxFloat32 LoadInputHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_mv_holes_tm1[iPxPos].x;
    }

    FfxFloat32 LoadInputHolesTm1(FFX_PARAMETER_IN FfxFloat32x2 fUvPos)
    {
        FfxInt32x2 iPxPos = FfxInt32x2(floor(fUvPos * FfxFloat32x2(GFI_HolesTm1Size())));
        return LoadInputHolesTm1(iPxPos);
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TM1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1
    Texture2D<FfxFloat32> r_mv_holes_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1);

    FfxFloat32 LoadInputHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_mv_holes_tp1[iPxPos].x;
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_HOLES_TP1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1
    Texture2D<FfxFloat32> r_dynamic_mask_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1);

    FfxFloat32 LoadDynamicMaskTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_dynamic_mask_tm1[iPxPos].x;
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TM1

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1
    Texture2D<FfxFloat32> r_dynamic_mask_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1);

    FfxFloat32 LoadDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_dynamic_mask_tp1[iPxPos].x;
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_INPUT_DYNAMIC_MASK_TP1

// The packed QData surfaces are GLSL `utexture2D` with an r32ui format: R32_UINT on
// D3D12. R32_UINT is the one typed UAV format D3D12 permits atomics on, so the GLSL
// imageAtomicMax in UpdateWarpMotionQDataTp1 / UpdateWarpFlowQDataTm1 becomes
// InterlockedMax on the UAV below with no format change.
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1
    Texture2D<FfxUInt32> r_warped_motion_qdata_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TP1);

    FfxUInt32 LoadWarpedMotionQDataTP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_warped_motion_qdata_tp1[iPxPos].x;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1
    Texture2D<FfxUInt32> r_warped_flow_qdata_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FLOW_QDATA_TM1);

    FfxUInt32 LoadWarpedFlowQDataTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_warped_flow_qdata_tm1[iPxPos].x;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TM1
    Texture2D<FfxUInt32> r_warped_motion_qdata_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_MOTION_QDATA_TM1);

    FfxUInt32 LoadWarpedMotionQDataTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return r_warped_motion_qdata_tm1[iPxPos].x;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1
    Texture2D<FfxFloat32x2> r_warped_filled_motion_tp1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1);

    FFX_MIN16_F2 LoadWarpedFilledMotionTP1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return FFX_MIN16_F2(r_warped_filled_motion_tp1[iPxPos].xy);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1
    Texture2D<FfxFloat32x2> r_warped_filled_flow_tm1 : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1);

    FFX_MIN16_F2 LoadWarpedFilledFlowTM1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return FFX_MIN16_F2(r_warped_filled_flow_tm1[iPxPos].xy);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE
    Texture2D<FfxFloat32x4> r_previous_interpolation_source : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE);

    FFX_MIN16_F3 LoadPreviousBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return FFX_MIN16_F3(r_previous_interpolation_source[iPxPos].rgb);
    }
    FFX_MIN16_F3 SamplePreviousBackbufferClampBorder(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return FFX_MIN16_F3(r_previous_interpolation_source.SampleLevel(s_LinearClamp, fUv, 0).xyz);
    }
    FFX_MIN16_F3 SamplePreviousBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return FFX_MIN16_F3(r_previous_interpolation_source.SampleLevel(s_LinearClamp, fUv, 0).xyz);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE
    Texture2D<FfxFloat32x4> r_current_interpolation_source : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE);

    FFX_MIN16_F3 LoadCurrentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return FFX_MIN16_F3(r_current_interpolation_source[iPxPos].rgb);
    }
    FFX_MIN16_F3 SampleCurrentBackbufferClampBorder(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return FFX_MIN16_F3(r_current_interpolation_source.SampleLevel(s_LinearClamp, fUv, 0).xyz);
    }
    FFX_MIN16_F3 SampleCurrentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return FFX_MIN16_F3(r_current_interpolation_source.SampleLevel(s_LinearClamp, fUv, 0).xyz);
    }
#endif

// ---------------------------------------------------------------------------
// The inference input tensor (SRV side). See the block comment above for why this
// is byte-addressed.
//
// Arm's GLSL has two flavours here -- a GL_ARM_tensors tensorARM (data-graph path)
// and a std430 storage buffer -- but both denote the same 16 bytes per element, so
// both collapse to the one byte-addressed reader below. LoadInTensorDataN returns
// the packed word whose lane N matches the component the GLSL int8_t4 lane N would
// have held, which is what the Dequantize(..., .x) call sites in
// ffx_frameinterpolation_debug_view.h index.
// ---------------------------------------------------------------------------
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR)
    ByteAddressBuffer r_in_tensor : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR);

    // Sign-extend one int8 lane out of a packed word into [-128,127].
    FfxInt32 GFI_TensorSignExtend(FfxUInt32 packed, FfxUInt32 lane)
    {
        FfxInt32 v = FfxInt32((packed >> (lane * 8u)) & 0xFFu);
        return (v & 0x80) ? (v - 0x100) : v;
    }

    // Repack a word whose low byte holds lane 0 into a vector of four sign-extended
    // lane values, one per component.
    FfxInt32x4 GFI_TensorUnpackWord(FfxUInt32 packed)
    {
        return FfxInt32x4(GFI_TensorSignExtend(packed, 0u),
                          GFI_TensorSignExtend(packed, 1u),
                          GFI_TensorSignExtend(packed, 2u),
                          GFI_TensorSignExtend(packed, 3u));
    }

    // Read the four packed words of one element, each already sign-extended into
    // one four-component vector.
    InputTensorElement_t GFI_LoadTensorElement(FfxUInt32x2 coord)
    {
        const FfxUInt32 elementIndex = coord.y * FfxUInt32(InputTensorSize().x) + coord.x;
        const FfxUInt32 base         = elementIndex * FFX_ARM_FRAMEINTERPOLATION_TENSOR_STRIDE;
        FfxUInt32x4     raw          = r_in_tensor.Load4(base);

        InputTensorElement_t te;
        te.rgbM1MV_rP1MV = FFX_MIN16_I4(GFI_TensorUnpackWord(raw.x));
        te.gbP1MV_rgM1OF = FFX_MIN16_I4(GFI_TensorUnpackWord(raw.y));
        te.bM1OF_rgbP1OF = FFX_MIN16_I4(GFI_TensorUnpackWord(raw.z));
        te.depth_disMask = FFX_MIN16_I4(GFI_TensorUnpackWord(raw.w));
        return te;
    }

    // channelOffset names a group of four int8 lanes: 0/4/8/12 -> member 0..3.
    // Returns that member's four lanes, which is what the GLSL returns as an
    // int8_t4 and what ffx_frameinterpolation_debug_view.h dequantises.
    tensorVec_t LoadInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN FfxUInt32 channelOffset)
    {
        InputTensorElement_t te = GFI_LoadTensorElement(coord);
        const FfxUInt32      w  = channelOffset / 4u;
        return FFX_MIN16_I4(FfxInt32x4(te.rgbM1MV_rP1MV[w], te.gbP1MV_rgM1OF[w], te.bM1OF_rgbP1OF[w], te.depth_disMask[w]));
    }

    tensorVec_t LoadInTensorData0(FFX_PARAMETER_IN FfxUInt32x2 coord)
    {
        return GFI_LoadTensorElement(coord).rgbM1MV_rP1MV;
    }

    tensorVec_t LoadInTensorData1(FFX_PARAMETER_IN FfxUInt32x2 coord)
    {
        return GFI_LoadTensorElement(coord).gbP1MV_rgM1OF;
    }

    tensorVec_t LoadInTensorData2(FFX_PARAMETER_IN FfxUInt32x2 coord)
    {
        return GFI_LoadTensorElement(coord).bM1OF_rgbP1OF;
    }

    tensorVec_t LoadInTensorData3(FFX_PARAMETER_IN FfxUInt32x2 coord)
    {
        return GFI_LoadTensorElement(coord).depth_disMask;
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_IN_TENSOR

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR
    Texture2D<FfxFloat32x4> r_out_params_tensor : FFX_DECLARE_SRV(FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR);

    FfxFloat32x4 SampleOutParamsTensor(FFX_PARAMETER_IN FfxFloat32x2 UV)
    {
        return r_out_params_tensor.SampleLevel(s_LinearClamp, UV, 0);
    }
#endif

///////////////////////////////////////////////
// declare UAVs and UAV accessors
///////////////////////////////////////////////

#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT) || defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE) ||                    \
    defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL) || defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1) ||                    \
    defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1) || defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT) ||                               \
    defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1) || defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1)
#define FFX_ARM_FRAMEINTERPOLATION_HAS_RENDER_TARGET 1

// One output per render-target store. The PS entry point returns this struct; the
// store function in the compute/render-target split above writes the member.
//
// The SV_Target indices are not all zero. Arm's GLSL gives each render-target
// binding an explicit layout(location = N), and the two that can be active together
// are the warp-flow pass's depth_tm1_next (location 0) and dynamic_mask_tp1
// (location 1) under MANAGE_PREVIOUS_DEPTH; the postprocess pass's previous-source
// output is location 1 alongside output at 0. Those cases are spelled out below. A
// struct with two SV_Target0 members is rejected by dxc ("overlapping semantic
// index"), so getting this wrong is a hard compile error, not a silent one.
struct FrameInterpolationOutput_t
{
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT
    FfxFloat32x4 o_output : SV_Target0;
#endif
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE
    FfxFloat32x4 o_previous_interpolation_source : SV_Target1;
#endif
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL
    FfxFloat32x4 o_colour_p1_internal : SV_Target0;
#endif

    // dynamic_mask_tp1 is location 1 exactly when the warp-flow pass also writes
    // depth_tm1_next (which is location 0 there); on its own it is location 0.
#if defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1) && defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT)
    FfxFloat32x4 o_dynamic_mask_tp1 : SV_Target1;
#elif defined(FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1)
    FfxFloat32x4 o_dynamic_mask_tp1 : SV_Target0;
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1
    FfxFloat32x4 o_depth_tm1 : SV_Target0;
#endif
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT
    FfxFloat32x4 o_depth_tm1_next : SV_Target0;
#endif
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1
    FfxFloat32x4 o_warped_filled_motion_tp1 : SV_Target0;
#endif
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1
    FfxFloat32x4 o_warped_filled_flow_tm1 : SV_Target0;
#endif
};

static FrameInterpolationOutput_t g_GFIOutput;

// The render-target store functions below assign these; the entry point returns the
// struct. Macro aliases keep the store bodies reading the same as the GLSL.
#define o_output                        g_GFIOutput.o_output
#define o_previous_interpolation_source g_GFIOutput.o_previous_interpolation_source
#define o_colour_p1_internal            g_GFIOutput.o_colour_p1_internal
#define o_dynamic_mask_tp1              g_GFIOutput.o_dynamic_mask_tp1
#define o_depth_tm1                     g_GFIOutput.o_depth_tm1
#define o_depth_tm1_next                g_GFIOutput.o_depth_tm1_next
#define o_warped_filled_motion_tp1      g_GFIOutput.o_warped_filled_motion_tp1
#define o_warped_filled_flow_tm1        g_GFIOutput.o_warped_filled_flow_tm1

#endif  // any render-target binding defined

// StoreDepthTm1 / StoreDepthTm1Next / StoreDynamicMaskTp1 / StoreColourP1Internal /
// StoreWarpFilledMotionTp1 / StoreWarpFilledFlowTm1 / StoreFrameinterpolationOutput
// / StorePreviousInterpolationSource each have a compute (UAV) form and a fragment
// (render-target) form in GLSL. Only one is compiled per shader, selected by which
// macro the entry point defines.
//
// In HLSL the render-target form is the pixel shader's SV_Target, not a binding, so
// its "store" writes a file-scope output variable that the PS entry point returns.
// Those variables are declared at the bottom of this file and the render-target
// branches live here rather than in the entry points, because the shared algorithm
// headers call these functions and so the definitions must precede the shared
// header -- which the entry point includes after this one. iPxPos is implicit in the
// render-target form; the parameter is kept so call sites are identical.

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1
    RWTexture2D<FfxFloat32> rw_depth_tm1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1);
    void StoreDepthTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
    {
        rw_depth_tm1[iPxPos] = depth;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1
    void StoreDepthTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
    {
        o_depth_tm1 = FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1_NEXT
    RWTexture2D<FfxFloat32> rw_depth_tm1_next : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DEPTH_TM1_NEXT);
    void StoreDepthTm1Next(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
    {
        rw_depth_tm1_next[iPxPos] = depth;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DEPTH_TM1_NEXT
    void StoreDepthTm1Next(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 depth)
    {
        o_depth_tm1_next = FfxFloat32x4(depth, 0.0f, 0.0f, 0.0f);
    }
#endif

// GLSL's dynamic-mask surface is an r8 image and the GLSL store clamps to [0,1]
// before writing, so R8_UNORM RWTexture2D<FfxFloat32> is the exact D3D12 equivalent.
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DYNAMIC_MASK_TP1
    RWTexture2D<FfxFloat32> rw_dynamic_mask_tp1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_DYNAMIC_MASK_TP1);
    void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxInt32 mask)
    {
        FfxFloat32 mask_f = clamp(FfxFloat32(mask), 0.0f, 1.0f);
        rw_dynamic_mask_tp1[iPxPos] = mask_f;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_DYNAMIC_MASK_TP1
    void StoreDynamicMaskTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxInt32 mask)
    {
        FfxFloat32 mask_f = clamp(FfxFloat32(mask), 0.0f, 1.0f);
        o_dynamic_mask_tp1 = FfxFloat32x4(mask_f, 0.0f, 0.0f, 1.0f);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1
    RWTexture2D<FfxUInt32> rw_warped_motion_qdata_tp1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_MOTION_QDATA_TP1);
    void UpdateWarpMotionQDataTp1(FFX_PARAMETER_IN FFX_MIN16_I2 iPxSample, FFX_PARAMETER_IN FfxUInt32 qdata)
    {
        InterlockedMax(rw_warped_motion_qdata_tp1[FfxInt32x2(iPxSample)], qdata);
    }
    void InitWarpClearQDataTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        rw_warped_motion_qdata_tp1[iPxPos] = 0u;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FLOW_QDATA_TM1
    RWTexture2D<FfxUInt32> rw_warped_flow_qdata_tm1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FLOW_QDATA_TM1);
    void UpdateWarpFlowQDataTm1(FFX_PARAMETER_IN FFX_MIN16_I2 iPxSample, FFX_PARAMETER_IN FfxUInt32 qdata)
    {
        InterlockedMax(rw_warped_flow_qdata_tm1[FfxInt32x2(iPxSample)], qdata);
    }
    void InitWarpClearQDataTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        rw_warped_flow_qdata_tm1[iPxPos] = 0u;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1
    RWTexture2D<FfxFloat32> rw_mv_holes_tp1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TP1);
    void StoreHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 nearestMotion)
    {
        FfxInt32x2 dst = iPxPos + FfxInt32x2(floor(nearestMotion * FFX_MIN16_F2(MvDepthLaneSize()) * FFX_MIN16_F(1.0f - Timestep())));
        rw_mv_holes_tp1[dst] = 1.0f;
    }
    void InitWarpClearHolesTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        rw_mv_holes_tp1[iPxPos] = 0.0f;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1
    RWTexture2D<FfxFloat32> rw_mv_holes_tm1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_HOLES_TM1);
    void StoreHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 nearestMotion)
    {
        FfxInt32x2 dst = iPxPos + FfxInt32x2(floor(nearestMotion * FFX_MIN16_F2(MvDepthLaneSize())));
        rw_mv_holes_tm1[dst] = 1.0f;
    }
    void InitWarpClearHolesTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        rw_mv_holes_tm1[iPxPos] = 0.0f;
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_COLOUR_P1_INTERNAL
    RWTexture2D<FfxFloat32x4> rw_colour_p1_internal : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_COLOUR_P1_INTERNAL);
    void StoreColourP1Internal(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
    {
        rw_colour_p1_internal[iPxPos] = FfxFloat32x4(val, 1.0f);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_COLOUR_P1_INTERNAL
    void StoreColourP1Internal(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 val)
    {
        o_colour_p1_internal = FfxFloat32x4(val, 1.0f);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_MOTION_TP1
    RWTexture2D<FfxFloat32x2> rw_warped_filled_motion_tp1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_MOTION_TP1);
    void StoreWarpFilledMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
    {
        rw_warped_filled_motion_tp1[iPxPos] = FfxFloat32x2(val);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_MOTION_TP1
    void StoreWarpFilledMotionTp1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
    {
        o_warped_filled_motion_tp1 = FfxFloat32x4(val, 0.0f, 0.0f);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_FLOW_TM1
    RWTexture2D<FfxFloat32x2> rw_warped_filled_flow_tm1 : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_WARP_FILLED_FLOW_TM1);
    void StoreWarpFilledFlowTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
    {
        rw_warped_filled_flow_tm1[iPxPos] = FfxFloat32x2(val);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_WARP_FILLED_FLOW_TM1
    void StoreWarpFilledFlowTm1(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 val)
    {
        o_warped_filled_flow_tm1 = FfxFloat32x4(val, 0.0f, 0.0f);
    }
#endif

// ---------------------------------------------------------------------------
// The inference input tensor (UAV side).
//
// Same 16-byte-per-element layout as the reader above. The whole point of this
// function is the one narrowing the shim comment describes: the shared headers have
// already quantised to [-128,127] as 32-bit integers, and StoreInTensor masks each
// lane to its low byte and packs the four lanes of each member into one word, so
// what lands in the buffer is the int8 std430 image the host and the dp4a backend
// expect.
// ---------------------------------------------------------------------------
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR
    RWByteAddressBuffer rw_in_tensor : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR);

    // Pack four already-quantised lane values into one word, low byte = lane 0.
    FfxUInt32 GFI_TensorPackWord(FFX_MIN16_I4 v)
    {
        return (FfxUInt32(v.x) & 0xFFu) | ((FfxUInt32(v.y) & 0xFFu) << 8) | ((FfxUInt32(v.z) & 0xFFu) << 16) | ((FfxUInt32(v.w) & 0xFFu) << 24);
    }

    void StoreInTensor(FFX_PARAMETER_IN FfxUInt32x2 coord, FFX_PARAMETER_IN InputTensorElement_t te)
    {
        const FfxUInt32 elementIndex = coord.y * FfxUInt32(InputTensorSize().x) + coord.x;
        const FfxUInt32 base         = elementIndex * FFX_ARM_FRAMEINTERPOLATION_TENSOR_STRIDE;

        rw_in_tensor.Store(base + 0u,  GFI_TensorPackWord(te.rgbM1MV_rP1MV));
        rw_in_tensor.Store(base + 4u,  GFI_TensorPackWord(te.gbP1MV_rgM1OF));
        rw_in_tensor.Store(base + 8u,  GFI_TensorPackWord(te.bM1OF_rgbP1OF));
        rw_in_tensor.Store(base + 12u, GFI_TensorPackWord(te.depth_disMask));
    }
#endif  // FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_IN_TENSOR

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUTPUT
    RWTexture2D<FfxFloat32x4> rw_output : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUTPUT);
    void StoreFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
    {
        rw_output[iPxPos] = FfxFloat32x4(val);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_OUTPUT
    void StoreFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
    {
        o_output = FfxFloat32x4(val);
    }
#endif

// StorePreviousInterpolationSource: writes the current backbuffer pixel into the
// ping-pong backup buffer that becomes colorTm1 on the next frame. Compiled in
// either a compute (UAV) or fragment (render-target) variant depending on which
// binding macro the host shader defined.
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREVIOUS_INTERPOLATION_SOURCE
    RWTexture2D<FfxFloat32x4> rw_previous_interpolation_source : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREVIOUS_INTERPOLATION_SOURCE);
    void StorePreviousInterpolationSource(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
    {
        rw_previous_interpolation_source[iPxPos] = FfxFloat32x4(val);
    }
#endif

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_RENDER_TARGET_PREVIOUS_INTERPOLATION_SOURCE
    void StorePreviousInterpolationSource(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
    {
        // iPxPos is implicit in the fragment-shader variant (SV_Position drives the write).
        o_previous_interpolation_source = FfxFloat32x4(val);
    }
#endif

#if !FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR
    RWTexture2D<FfxFloat32x4> rw_out_params_tensor : FFX_DECLARE_UAV(FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUT_PARAMS_TENSOR);
    void StoreOutParamsTensor(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F4 val)
    {
        rw_out_params_tensor[iPxPos] = FfxFloat32x4(val);
    }
#endif
#endif  //!FFX_ARM_FRAMEINTERPOLATION_OPTION_ENABLE_DATA_GRAPH_FI

#endif  // #if defined(FFX_GPU)

#endif  // FFX_ARM_FRAMEINTERPOLATION_CALLBACKS_HLSL_H
