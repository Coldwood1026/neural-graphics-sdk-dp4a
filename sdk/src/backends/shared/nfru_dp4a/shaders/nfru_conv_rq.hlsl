// =============================================================================
// NFRU v1 int8 inference -- DX12 / HLSL dp4a kernel
// =============================================================================
// Direct3D 12 counterpart of `fru_conv_rq.comp`. The two must produce IDENTICAL int8
// results; only the API differs.
//
// `dot4add_i8packed` is HLSL's DP4A intrinsic: four signed 8-bit multiply-accumulates
// into a 32-bit accumulator. Like the Vulkan kernel, activations and weights are stored
// four int8 per uint32 word, so one call consumes four channels.
//
// Semantics, identical to the Vulkan path:
//   pad   : out-of-range taps read 0x80808080 (four int8 of -128 == the input zero
//           point), so their contribution z_a*sum(w) is already in the accumulator and
//           the bias carries the matching correction
//   requantise : r = clamp(((acc + bc)*M + (1<<(S-1))) >> S + out_zp), 64-bit product
//   ReLU  : implicit when out_zp == -128; the head uses out_zp = 44 and is not ReLU'd
//
// SPDX-License-Identifier: MIT
// =============================================================================

#define OCG 4

struct ConvPush {
    uint in_w;
    uint in_c4;
    uint out_h;
    uint out_w;
    uint out_c4;
    int  pad_top;
    int  pad_left;
    uint stride_h;
    uint stride_w;
    int  out_zp;
    uint kh;
    uint kw;
    int  has_lut;
    uint in_h;
};

ConstantBuffer<ConvPush> pc : register(b0);

ByteAddressBuffer   actIn  : register(t0);
ByteAddressBuffer   wPk    : register(t1);
ByteAddressBuffer   biasCr : register(t2);
ByteAddressBuffer   mult   : register(t3);
ByteAddressBuffer   shiftB : register(t4);
RWByteAddressBuffer actOut : register(u0);

// bias -> rescale -> +out_zp -> clamp, packed into one uint of four int8.
uint PackOut4(int a0, int a1, int a2, int a3, uint oc0)
{
    uint packed = 0u;
    int accs[4] = { a0, a1, a2, a3 };
    [unroll]
    for (int j = 0; j < 4; ++j) {
        const uint oc = oc0 + (uint)j;
        const int  v  = accs[j] + biasCr.Load(oc * 4);
        const int  sh = shiftB.Load(oc * 4);
        // 64-bit: the accumulator is ~25 bits and the multiplier up to 31, so the
        // product reaches 56 bits and would silently overflow in 32.
        int64_t r = ((int64_t)v * (int64_t)mult.Load(oc * 4) + ((int64_t)1 << (sh - 1))) >> sh;
        r += (int64_t)pc.out_zp;
        r  = clamp(r, (int64_t)-128, (int64_t)127);
        // No extra ReLU: out_zp == -128 already guarantees r >= 0.
        packed |= ((uint)(int)r & 0xFFu) << ((uint)j * 8u);
    }
    return packed;
}

[numthreads(8, 8, 1)]
void CSConv(uint3 gid : SV_DispatchThreadID)
{
    const uint ox      = gid.x;
    const uint oy      = gid.y;
    const uint oc4Base = gid.z * OCG;

    if (ox >= pc.out_w || oy >= pc.out_h || oc4Base >= pc.out_c4) {
        return;
    }

    uint groups = pc.out_c4 - oc4Base;
    if (groups > OCG) groups = OCG;

    const uint K4     = pc.kh * pc.kw * pc.in_c4;
    // Channel written for group g, byte j is oc = (oc4Base + g)*4 + j, and this layer's
    // weights are bound at binding 1 laid out as wpk[oc*K4 + wOff + ic4], so
    // wBase0 + g*4*K4 + j*K4 == oc*K4 and this base is exactly right.
    // The layer's offset inside the shared weight pool is applied as a DESCRIPTOR OFFSET
    // when binding, not here; omitting it would make every layer read conv1's weights.
    const uint wBase0 = oc4Base * 4u * K4;

    int a00 = 0, a01 = 0, a02 = 0, a03 = 0;
    int a10 = 0, a11 = 0, a12 = 0, a13 = 0;
    int a20 = 0, a21 = 0, a22 = 0, a23 = 0;
    int a30 = 0, a31 = 0, a32 = 0, a33 = 0;

    for (uint ky = 0u; ky < pc.kh; ++ky) {
        const int iy = (int)oy * (int)pc.stride_h - pc.pad_top + (int)ky;
        for (uint kx = 0u; kx < pc.kw; ++kx) {
            const int ix = (int)ox * (int)pc.stride_w - pc.pad_left + (int)kx;

            // Out-of-range taps use z_a (0x80 per byte). With the bias correction this
            // contributes exactly zero, matching "skip the tap".
            const bool tapOut = (iy < 0 || iy >= (int)pc.in_h ||
                                 ix < 0 || ix >= (int)pc.in_w);
            const uint aBase = (uint)iy * (pc.in_w * pc.in_c4) + (uint)ix * pc.in_c4;
            const uint wOff  = (ky * pc.kw + kx) * pc.in_c4;

            for (uint ic4 = 0u; ic4 < pc.in_c4; ++ic4) {
                const uint a = tapOut ? 0x80808080u : actIn.Load((aBase + ic4) * 4u);

                if (groups > 0u) {
                    const uint b = wOff + ic4;
                    a00 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a00);
                    a01 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a01);
                    a02 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a02);
                    a03 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a03);
                }
                if (groups > 1u) {
                    const uint b = 4u * K4 + wOff + ic4;
                    a10 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a10);
                    a11 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a11);
                    a12 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a12);
                    a13 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a13);
                }
                if (groups > 2u) {
                    const uint b = 8u * K4 + wOff + ic4;
                    a20 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a20);
                    a21 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a21);
                    a22 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a22);
                    a23 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a23);
                }
                if (groups > 3u) {
                    const uint b = 12u * K4 + wOff + ic4;
                    a30 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a30);
                    a31 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a31);
                    a32 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a32);
                    a33 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a33);
                }
            }
        }
    }

    // Tight NHWC output, one uint per four channels.
    const uint outBase = (oy * pc.out_w + ox) * pc.out_c4;

    // Debug output paths, identical to the Vulkan kernel's:
    //   has_lut == 2 -> one int32 accumulator per output channel
    //   has_lut == 3 -> a per-thread marker instead of the accumulator
    // Without these the DX12 backend silently wrote packed codes while the harness
    // compared against int32 accumulators, which looks exactly like a broken kernel.
    if (pc.has_lut >= 2) {
        int accsAll[16] = { a00, a01, a02, a03, a10, a11, a12, a13,
                            a20, a21, a22, a23, a30, a31, a32, a33 };
        const uint pixelBase = (oy * pc.out_w + ox) * pc.out_c4 * 4u;
        for (uint g = 0u; g < groups; ++g) {
            const uint oc = (oc4Base + g) * 4u;
            for (uint j = 0u; j < 4u; ++j) {
                uint v;
                if (pc.has_lut == 3) {
                    v = 0xA5A50000u | ((oy & 0xFFu) << 16) | ((ox & 0xFFu) << 8) |
                        ((oc + j) & 0xFFu);
                } else {
                    v = (uint)accsAll[g * 4u + j];
                }
                actOut.Store((pixelBase + oc + j) * 4u, v);
            }
        }
        return;
    }
    if (groups > 0u) actOut.Store((outBase + oc4Base + 0u) * 4u, PackOut4(a00, a01, a02, a03, (oc4Base + 0u) * 4u));
    if (groups > 1u) actOut.Store((outBase + oc4Base + 1u) * 4u, PackOut4(a10, a11, a12, a13, (oc4Base + 1u) * 4u));
    if (groups > 2u) actOut.Store((outBase + oc4Base + 2u) * 4u, PackOut4(a20, a21, a22, a23, (oc4Base + 2u) * 4u));
    if (groups > 3u) actOut.Store((outBase + oc4Base + 3u) * 4u, PackOut4(a30, a31, a32, a33, (oc4Base + 3u) * 4u));
}
