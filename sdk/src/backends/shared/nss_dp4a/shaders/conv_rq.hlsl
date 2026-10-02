/*
 * conv_rq.hlsl -- fused CONV2D + bias + DP4A correction + RESCALE (+ LUT).
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Port of nss_kernel/`conv_rq.comp` (GLSL) for the D3D12 path. The GLSL is the
 * reference; this is a line-by-line translation. Index arithmetic, accumulation order
 * and the sign/width details are all preserved deliberately -- see the notes below,
 * which mark the places where a "tidier" version would stop being bit-exact.
 * =============================================================================
 *
 * TOSA semantics (from ai-ml-emulation-layer-for-vulkan TOSA graph ops):
 *
 *     acc[o] = Σ (q_a - z_a)(q_w - z_w) + bias[o]          z_w = 0 (symmetric weights)
 *            = Σ q_a·q_w - z_a·Σ_all q_w                   the second term is a per-output
 *                                                          -channel constant ("corr")
 *
 * Out-of-range taps contribute 0, which the GLSL achieves by pre-filling the activation
 * buffer border with z_a (-128, 0x80). This port uses the `inBorded == 0` branch instead,
 * which substitutes 0x80808080 for out-of-range taps directly -- strictly equivalent,
 * because (q_a - z_a) = 0 for that value, so "read the border" and "skip the tap" agree
 * exactly. The border branch is kept too, since the host may supply either.
 *
 * Requantisation (RESCALE, SINGLE_ROUND):
 *
 *     r = (acc · multiplier + (1 << (shift-1))) >> shift    arithmetic shift
 *     r = clamp(r + out_zp, -128, 127)
 *
 * When out_zp == -128 the clamp already makes the result non-negative -- that IS the
 * implicit ReLU. Do NOT add a max(x, 0); doing so is a no-op at best and a
 * divergence from the reference at worst, since it would apply to layers that were
 * never meant to be rectified.
 *
 * Activations are NHWC stored as a uint32 view: the low byte of word[d] is channel 4d.
 * So "four consecutive channels" is one uint32 access, which lines up exactly with
 * dot4add_i8packed's 4x8 packing.
 *
 * Weights are packed_w[oc][K/4] with K/4 = (kh*kw*Cin)/4 and the K axis ordered (ky,kx,ic).
 *
 * PERFORMANCE NOTE (carried over from the GLSL, 2026-10-01):
 * each thread handles OCG oc4 groups (OCG*4 output channels) rather than one. Previously
 * there were out_c4 z-slices and each one reloaded the same 3x3xCin neighbourhood --
 * with out_c4 = 8 the same activations were read 8 times, and per-dispatch measurement
 * showed the kernel reaching ~1/25 of the hardware dot-product peak and ~8% of effective
 * bandwidth. OCG must match CONV_OCG on the host side, which divides the dispatch's gz
 * by it.
 */

// ---------------------------------------------------------------------------
// Bindings. These are this port's choice; the Vulkan side uses descriptor set 0 with
// bindings 0..7, and there is no reason for D3D12 to mirror a binding *number* that only
// existed because the GLSL had to assign one.
// ---------------------------------------------------------------------------
ByteAddressBuffer   actIn  : register(t0);
ByteAddressBuffer   wPk    : register(t1);
ByteAddressBuffer   biasCr : register(t2);
ByteAddressBuffer   mult   : register(t3);
ByteAddressBuffer   shiftB : register(t4);
ByteAddressBuffer   lut    : register(t5);
RWByteAddressBuffer actOut : register(u0);
RWByteAddressBuffer dbgBuf : register(u1);

struct ConvPushConstants
{
    uint in_w;        // input interior width, excluding the border
    uint in_c4;       // Cin / 4
    uint out_h;
    uint out_w;
    uint out_c4;      // Cout / 4
    int  pad_top;
    int  pad_left;
    uint stride_h;
    uint stride_w;
    int  out_zp;
    uint kh;
    uint kw;
    uint has_lut;
    uint dbg;         // non-zero: write raw int32 accumulators to dbgBuf
    uint inH;         // input interior height (used for the inBorded == 0 bounds test)
    uint inBorded;    // 1 = bordered internal buffer; 0 = host's tight buffer
    uint outBorded;   // 1 = write the bordered internal buffer; 0 = write the host's
};
ConstantBuffer<ConvPushConstants> pc : register(b0);

/*
 * Signed 4x8 packed dot product.
 *
 * The GLSL had to declare this via spirv_intrinsics because glslang 15.4 does not expose
 * GL_EXT_shader_integer_dot_product; in HLSL it is a first-class intrinsic, and requires
 * Shader Model 6.4.
 *
 * It is used FUSED -- `acc = dot4add_i8packed(a, w, acc)` -- rather than as
 * `acc += dot4add_i8packed(a, w, 0)`, which is how the GLSL reads. Two reasons, and the
 * second is not a style preference:
 *
 *   1. One instruction instead of two, and that is the form the hardware dot-product path
 *      wants.
 *   2. The `int` cast form ICEs this dxc outright -- "Internal Compiler error:" with an
 *      empty message and no output file, at the very first use. The fusion is therefore
 *      not cosmetic, it is what compiles. Both forms are int32 add of the same product,
 *      so they agree bit for bit.
 *
 * NFRU's conv kernel uses the same fused form for the same reason.
 */

/* Number of oc4 groups each thread handles. Must match CONV_OCG on the host side. */
#define OCG 4u

/*
 * -- CONV_2X2 variant (2026-10-01) ------------------------------------------
 * Each thread produces a 2x2 block of output pixels. It exists for the layers with very
 * few output channels (out_c4 == 1, i.e. cout == 4) at full resolution -- NSS HIGH's
 * last temporal layer, op32. There the base kernel emits one word per thread for 576
 * MACs, so fixed overhead (activation re-read plus launch) dominates; measured ~94 GB/s.
 *
 * Structure: the workgroup (8x8 = 64 threads) cooperatively loads the 4x4 tap window of
 * a 2x2 pixel block across all of in_c4 into shared memory (about one word per thread,
 * so global activation traffic drops to roughly 1/64), then each thread computes its own
 * 2x2 pixels. Accumulation order is still (ky,kx,ic4), so it stays bit-identical to the
 * base kernel.
 *
 * Preconditions, guaranteed by host-side dispatch selection: strideH == strideW == 1,
 * in_c4 <= 8, out_c4 == 1.
 * ---------------------------------------------------------------------------
 */
#ifdef CONV_2X2
groupshared uint sTap[4 * 4 * 8];   // 4x4 window x in_c4(<=8); op32 uses 64 words (256 B)
#endif

/*
 * Pack four output channels: bias -> rescale -> +zp -> clamp -> LUT.
 *
 * The 64-bit product is not optional: acc is about 25 bits and multiplier reaches 31, so
 * the product can be 56 bits wide. The GLSL says the same thing at the same place; this
 * is the single easiest thing to get wrong when "simplifying" the kernel.
 */
uint packOut(int accs[4], int oc0)
{
    uint packed = 0u;
    for (int j = 0; j < 4; ++j)
    {
        const int oc = oc0 + j;
        const int v  = accs[j] + asint(biasCr.Load(oc * 4));
        const int sh = asint(shiftB.Load(oc * 4));

        int64_t r = ((int64_t)v * (int64_t)asint(mult.Load(oc * 4)) + ((int64_t)1 << (sh - 1))) >> sh;
        r += (int64_t)pc.out_zp;
        r = min(max(r, (int64_t)-128), (int64_t)127);

        int q = (int)r;
        if (pc.has_lut != 0u)
        {
            q = asint(lut.Load((q + 128) * 4));
        }
        packed |= ((uint)q & 0xFFu) << ((uint)j * 8u);
    }
    return packed;
}

#ifdef CONV_2X2
/* 2x2 pixels per thread. See the file header. Preconditions: stride == 1, in_c4 <= 8,
 * out_c4 <= OCG. */
void conv2x2Main(uint3 gid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint ox0 = gid.x * 2u;
    const uint oy0 = gid.y * 2u;
    const uint oc4Base = gid.z * OCG;

    uint groups = pc.out_c4 - oc4Base;
    if (groups > OCG) groups = OCG;
    if (ox0 >= pc.out_w || oy0 >= pc.out_h || groups == 0u) return;

    const uint inStride = (pc.in_w + 2u) * pc.in_c4;
    const uint tid   = gtid.y * 8u + gtid.x;
    const uint nTapW = 16u * pc.in_c4;

    /*
     * The window's top-left corner is the first tap of pixel (oy0, ox0). Bounds are
     * tested in INTERIOR coordinates uniformly, because on a tail block the 4x4 window
     * sticks one or two rows/columns past the edge; an out-of-range tap takes
     * z_a (0x80808080), which is strictly equivalent to the border value a bordered
     * buffer would carry, so the two paths stay bit-identical.
     *
     * Interior height = out_h, because this variant requires stride == 1.
     */
    const int y0 = (int)oy0 - pc.pad_top;
    const int x0 = (int)ox0 - pc.pad_left;
    for (uint i = tid; i < nTapW; i += 64u)
    {
        const uint t   = i / pc.in_c4;
        const uint ic4 = i - t * pc.in_c4;
        const uint ty  = t >> 2u;
        const uint tx  = t & 3u;
        const int tyi  = y0 + (int)ty;
        const int txi  = x0 + (int)tx;

        uint v = 0x80808080u;
        if (tyi >= 0 && tyi < (int)pc.out_h && txi >= 0 && txi < (int)pc.in_w)
        {
            if (pc.inBorded != 0u)
            {
                v = actIn.Load(((uint)(tyi + 1) * inStride
                                + (uint)(txi + 1) * pc.in_c4 + ic4) * 4u);
            }
            else
            {
                v = actIn.Load(((uint)tyi * (pc.in_w * pc.in_c4)
                                + (uint)txi * pc.in_c4 + ic4) * 4u);
            }
        }
        sTap[i] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    /* Four pixels (py*2+px) x four output channels. Accumulation order (ky,kx,ic4) is
     * identical to the base kernel, so the result is bit-identical. The tap window
     * coordinate is (py+ky, px+kx). */
    int acc00[4], acc01[4], acc10[4], acc11[4];
    acc00[0]=0; acc00[1]=0; acc00[2]=0; acc00[3]=0;
    acc01[0]=0; acc01[1]=0; acc01[2]=0; acc01[3]=0;
    acc10[0]=0; acc10[1]=0; acc10[2]=0; acc10[3]=0;
    acc11[0]=0; acc11[1]=0; acc11[2]=0; acc11[3]=0;

    const uint K4     = pc.kh * pc.kw * pc.in_c4;
    const uint wBase0 = oc4Base * 4u * K4;

    for (uint ky = 0u; ky < pc.kh; ++ky)
    {
        for (uint kx = 0u; kx < pc.kw; ++kx)
        {
            const uint wOff = (ky * pc.kw + kx) * pc.in_c4;
            for (uint ic4 = 0u; ic4 < pc.in_c4; ++ic4)
            {
                const uint b   = wOff + ic4;
                const uint a00 = sTap[((0u + ky) * 4u + (0u + kx)) * pc.in_c4 + ic4];
                const uint a01 = sTap[((0u + ky) * 4u + (1u + kx)) * pc.in_c4 + ic4];
                const uint a10 = sTap[((1u + ky) * 4u + (0u + kx)) * pc.in_c4 + ic4];
                const uint a11 = sTap[((1u + ky) * 4u + (1u + kx)) * pc.in_c4 + ic4];
                if (groups > 0u)
                {
                    acc00[0] = dot4add_i8packed(a00, wPk.Load((wBase0 + b) * 4u), acc00[0]);
                    acc00[1] = dot4add_i8packed(a00, wPk.Load((wBase0 + b + K4) * 4u), acc00[1]);
                    acc00[2] = dot4add_i8packed(a00, wPk.Load((wBase0 + b + 2u * K4) * 4u), acc00[2]);
                    acc00[3] = dot4add_i8packed(a00, wPk.Load((wBase0 + b + 3u * K4) * 4u), acc00[3]);
                    acc01[0] = dot4add_i8packed(a01, wPk.Load((wBase0 + b) * 4u), acc01[0]);
                    acc01[1] = dot4add_i8packed(a01, wPk.Load((wBase0 + b + K4) * 4u), acc01[1]);
                    acc01[2] = dot4add_i8packed(a01, wPk.Load((wBase0 + b + 2u * K4) * 4u), acc01[2]);
                    acc01[3] = dot4add_i8packed(a01, wPk.Load((wBase0 + b + 3u * K4) * 4u), acc01[3]);
                    acc10[0] = dot4add_i8packed(a10, wPk.Load((wBase0 + b) * 4u), acc10[0]);
                    acc10[1] = dot4add_i8packed(a10, wPk.Load((wBase0 + b + K4) * 4u), acc10[1]);
                    acc10[2] = dot4add_i8packed(a10, wPk.Load((wBase0 + b + 2u * K4) * 4u), acc10[2]);
                    acc10[3] = dot4add_i8packed(a10, wPk.Load((wBase0 + b + 3u * K4) * 4u), acc10[3]);
                    acc11[0] = dot4add_i8packed(a11, wPk.Load((wBase0 + b) * 4u), acc11[0]);
                    acc11[1] = dot4add_i8packed(a11, wPk.Load((wBase0 + b + K4) * 4u), acc11[1]);
                    acc11[2] = dot4add_i8packed(a11, wPk.Load((wBase0 + b + 2u * K4) * 4u), acc11[2]);
                    acc11[3] = dot4add_i8packed(a11, wPk.Load((wBase0 + b + 3u * K4) * 4u), acc11[3]);
                }
            }
        }
    }

    /* Output: same layout and quantisation path as the base kernel; out-of-range pixels
     * on odd widths/heights are skipped rather than written. */
    for (uint py = 0u; py < 2u; ++py)
    {
        const uint oy = oy0 + py;
        if (oy >= pc.out_h) break;
        for (uint px = 0u; px < 2u; ++px)
        {
            const uint ox = ox0 + px;
            if (ox >= pc.out_w) break;

            const uint outBase = (pc.outBorded == 0u)
                    ? (oy * pc.out_w + ox) * pc.out_c4
                    : ((oy + 1u) * (pc.out_w + 2u) * pc.out_c4 + (ox + 1u) * pc.out_c4);

            for (uint g = 0u; g < groups; ++g)
            {
                const int oc0 = (int)(oc4Base + g) * 4;
                if (py == 0u && px == 0u)
                {
                    actOut.Store((outBase + oc4Base + g) * 4u, packOut(acc00, oc0));
                }
                else if (py == 0u)
                {
                    actOut.Store((outBase + oc4Base + g) * 4u, packOut(acc01, oc0));
                }
                else if (px == 0u)
                {
                    actOut.Store((outBase + oc4Base + g) * 4u, packOut(acc10, oc0));
                }
                else
                {
                    actOut.Store((outBase + oc4Base + g) * 4u, packOut(acc11, oc0));
                }
            }
        }
    }
}
#endif

[numthreads(8, 8, 1)]
void CSConv(uint3 gid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
#ifdef CONV_2X2
    conv2x2Main(gid, gtid);
    return;
#else
    const uint ox      = gid.x;
    const uint oy      = gid.y;
    const uint oc4Base = gid.z * OCG;

    const uint K4     = pc.kh * pc.kw * pc.in_c4;
    const uint wBase0 = oc4Base * 4u * K4;

    /* How many groups this thread actually computes; a tail block may have fewer than
     * OCG. The value depends only on z, so it is uniform across the workgroup. */
    uint groups = OCG;
    if (pc.out_c4 < oc4Base + OCG)
    {
        groups = pc.out_c4 - oc4Base;
    }

    if (ox >= pc.out_w || oy >= pc.out_h || oc4Base >= pc.out_c4 || groups == 0u)
    {
        return;
    }

    const uint inStride = (pc.in_w + 2u) * pc.in_c4;   // in uint32 units

    int a00 = 0, a01 = 0, a02 = 0, a03 = 0;
    int a10 = 0, a11 = 0, a12 = 0, a13 = 0;
    int a20 = 0, a21 = 0, a22 = 0, a23 = 0;
    int a30 = 0, a31 = 0, a32 = 0, a33 = 0;

    for (uint ky = 0u; ky < pc.kh; ++ky)
    {
        // The border is one pixel, so interior coordinate iy is buffer coordinate iy+1.
        const int py = (int)oy * (int)pc.stride_h - pc.pad_top + (int)ky + 1;
        for (uint kx = 0u; kx < pc.kw; ++kx)
        {
            const int px = (int)ox * (int)pc.stride_w - pc.pad_left + (int)kx + 1;

            /* inBorded == 0: read the host's tight buffer and substitute z_a for
             * out-of-range taps. z_a = -128 = 0x80 means (q_a - z_a) = 0, which is
             * strictly equivalent both to reading a border value and to skipping the
             * tap -- so the result matches the bordered path bit for bit, while saving
             * the entire input mirror copy (measured at 19%). */
            uint aBase;
            bool tapOut = false;
            if (pc.inBorded == 0u)
            {
                const int ty = py - 1;
                const int tx = px - 1;
                tapOut = (ty < 0 || ty >= (int)pc.inH || tx < 0 || tx >= (int)pc.in_w);
                aBase  = (uint)ty * (pc.in_w * pc.in_c4) + (uint)tx * pc.in_c4;
            }
            else
            {
                aBase = (uint)py * inStride + (uint)px * pc.in_c4;
            }

            const uint wOff = (ky * pc.kw + kx) * pc.in_c4;
            for (uint ic4 = 0u; ic4 < pc.in_c4; ++ic4)
            {
                // One load, reused by all OCG groups -- this is the whole point of the
                // optimisation. The GLSL comment says the same.
                const uint a = tapOut ? 0x80808080u : actIn.Load((aBase + ic4) * 4u);
                if (groups > 0u)
                {
                    const uint b = wOff + ic4;
                    a00 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a00);
                    a01 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a01);
                    a02 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a02);
                    a03 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a03);
                }
                if (groups > 1u)
                {
                    const uint b = 4u * K4 + wOff + ic4;
                    a10 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a10);
                    a11 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a11);
                    a12 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a12);
                    a13 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a13);
                }
                if (groups > 2u)
                {
                    const uint b = 8u * K4 + wOff + ic4;
                    a20 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a20);
                    a21 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a21);
                    a22 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a22);
                    a23 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a23);
                }
                if (groups > 3u)
                {
                    const uint b = 12u * K4 + wOff + ic4;
                    a30 = dot4add_i8packed(a, wPk.Load((wBase0 + b) * 4u), a30);
                    a31 = dot4add_i8packed(a, wPk.Load((wBase0 + b + K4) * 4u), a31);
                    a32 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 2u * K4) * 4u), a32);
                    a33 = dot4add_i8packed(a, wPk.Load((wBase0 + b + 3u * K4) * 4u), a33);
                }
            }
        }
    }

    /* Debug channel: raw accumulators, first group only (equivalent to the old oc4 == 0). */
    if (pc.dbg != 0u && oc4Base == 0u && groups > 0u)
    {
        const uint dbgBase = ((uint)oy * pc.out_w + (uint)ox) * 24u;
        int accs[4] = { a00, a01, a02, a03 };
        for (int j = 0; j < 4; ++j)
        {
            const int oo = j;
            const int vv = accs[j] + asint(biasCr.Load(oo * 4));
            const int sh = asint(shiftB.Load(oo * 4));
            int64_t rr = ((int64_t)vv * (int64_t)asint(mult.Load(oo * 4)) + ((int64_t)1 << (sh - 1))) >> sh;
            int64_t qq = rr + (int64_t)pc.out_zp;
            qq = min(max(qq, (int64_t)-128), (int64_t)127);
            dbgBuf.Store((dbgBase + (uint)j * 6u + 0u) * 4u, (uint)vv);
            dbgBuf.Store((dbgBase + (uint)j * 6u + 1u) * 4u, (uint)asint(mult.Load(oo * 4)));
            dbgBuf.Store((dbgBase + (uint)j * 6u + 2u) * 4u, (uint)sh);
            dbgBuf.Store((dbgBase + (uint)j * 6u + 3u) * 4u, (uint)(int)rr);
            dbgBuf.Store((dbgBase + (uint)j * 6u + 4u) * 4u, (uint)(int)qq);
            dbgBuf.Store((dbgBase + (uint)j * 6u + 5u) * 4u, (uint)pc.out_zp);
        }
    }

    // outBorded == 0: write the host's tight buffer directly (saves the output copy-back,
    // measured at 10.5%). == 1: write the bordered internal buffer, where interior pixel
    // (oy,ox) lands at buffer (oy+1, ox+1).
    const uint outBase = (pc.outBorded == 0u)
            ? ((uint)oy * pc.out_w + (uint)ox) * pc.out_c4
            : (((uint)oy + 1u) * (pc.out_w + 2u) * pc.out_c4 + ((uint)ox + 1u) * pc.out_c4);

    if (groups > 0u)
    {
        int accs[4] = { a00, a01, a02, a03 };
        actOut.Store((outBase + oc4Base + 0u) * 4u, packOut(accs, (int)(oc4Base + 0u) * 4));
    }
    if (groups > 1u)
    {
        int accs[4] = { a10, a11, a12, a13 };
        actOut.Store((outBase + oc4Base + 1u) * 4u, packOut(accs, (int)(oc4Base + 1u) * 4));
    }
    if (groups > 2u)
    {
        int accs[4] = { a20, a21, a22, a23 };
        actOut.Store((outBase + oc4Base + 2u) * 4u, packOut(accs, (int)(oc4Base + 2u) * 4));
    }
    if (groups > 3u)
    {
        int accs[4] = { a30, a31, a32, a33 };
        actOut.Store((outBase + oc4Base + 3u) * 4u, packOut(accs, (int)(oc4Base + 3u) * 4));
    }
#endif
}
