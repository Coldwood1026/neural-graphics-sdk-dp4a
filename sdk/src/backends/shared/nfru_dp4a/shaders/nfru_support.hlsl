// =============================================================================
// NFRU v1 support kernels -- DX12 / HLSL: copy / nearest x2 resize / channel concat
// =============================================================================
// Direct3D 12 counterpart of `nfru_support.comp`. All three ops work on flat uint32
// words because an int8 NHWC tensor stores four channels per word and every op here
// preserves that packing (this model's concatenation boundaries are all 4-channel
// aligned, so no sub-word shifting is ever required).
//
// One shader with a mode switch, so the push constant block and the dispatch shape match
// the Vulkan kernel exactly: 16 uints on both APIs.
//
// SPDX-License-Identifier: MIT
// =============================================================================

// Index constants, kept identical to the Vulkan side.
#define PC_OP            0u
#define PC_COUNT         1u
#define PC_OUT_OFF       2u
#define PC_DST_W         3u
#define PC_DST_H         4u
#define PC_DST_C4        5u
#define PC_SRC_OFF       6u
#define PC_SRC_W         7u
#define PC_SRC_H         8u
#define PC_SRC_C4        9u
#define PC_PLANE_ROWS   10u
#define PC_PLANE_WORDS  11u
#define PC_SRC_C4_MUL   12u
#define PC_IN_OFF       13u

// A struct rather than an array: a constant buffer must be one object, and this maps
// directly onto a D3D12 root-constant range of 16 uints.
struct SupportPush {
    uint op;
    uint count;
    uint outOff;
    uint dstW;
    uint dstH;
    uint dstC4;
    uint srcOff;
    uint srcW;
    uint srcH;
    uint srcC4;
    uint planeRows;
    uint planeWords;
    uint srcC4Mul;
    uint inOff;
    uint reserved0;
    uint reserved1;
};

ConstantBuffer<SupportPush> pc : register(b0);

ByteAddressBuffer   src : register(t0);
RWByteAddressBuffer dst : register(u0);

[numthreads(256, 1, 1)]
void CSSupport(uint3 gid : SV_DispatchThreadID)
{
    const uint id = gid.x;
    if (id >= pc.count) {
        return;
    }

    const uint op = pc.op;

    if (op == 0u) {
        // Straight word copy: used to place a tensor into the host output buffer.
        dst.Store((pc.outOff + id) * 4u, src.Load((pc.srcOff + id) * 4u));
        return;
    }

    if (op == 1u) {
        // Nearest x2 upsample: destination (x, y) samples source (x/2, y/2). Channel
        // words pass through untouched, preserving the int8 packing.
        //
        // `id` indexes WORDS and there are `c4` words per pixel, so it splits as
        // (pixel, channel word). Reading it as a pixel index is only right when c4 == 1;
        // at conv5e's 16 channels it walked a quarter of the destination as if it were
        // the whole thing. Kept identical to the Vulkan kernel.
        const uint srcW = pc.srcW;
        const uint c4   = pc.srcC4;
        const uint dstW = pc.dstW;
        const uint dstH = pc.dstH;
        const uint word = id % c4;
        const uint pix  = id / c4;
        const uint x    = pix % dstW;
        const uint y    = (pix / dstW) % dstH;
        const uint rest = pix / (dstW * dstH);
        const uint sx   = x >> 1u;
        const uint sy   = y >> 1u;
        const uint srcIdx = pc.srcOff + rest * (srcW * pc.srcH * c4)
                          + (sy * srcW + sx) * c4 + word;
        dst.Store((pc.outOff + id) * 4u, src.Load(srcIdx * 4u));
        return;
    }

    // op == 2: concatenate along channels. One dispatch covers all source planes; the
    // plane is selected from the pixel index, so no per-source dispatch is needed.
    const uint planeRows      = pc.planeRows;
    const uint planeWords     = pc.planeWords;
    const uint mul            = pc.srcC4Mul;
    const uint srcPlane       = planeRows * planeWords;
    const uint dstPixelWords  = pc.dstC4;
    const uint srcPixelsPlane = srcPlane / mul;

    const uint pixel   = id / mul;
    const uint which   = pixel / srcPixelsPlane;
    const uint inPlane = pixel % srcPixelsPlane;
    const uint word    = id % mul;
    const uint srcIdx  = pc.inOff + which * srcPlane + inPlane * mul + word;

    dst.Store((pc.outOff + inPlane * dstPixelWords + which * mul + word) * 4u,
              src.Load(srcIdx * 4u));
}
