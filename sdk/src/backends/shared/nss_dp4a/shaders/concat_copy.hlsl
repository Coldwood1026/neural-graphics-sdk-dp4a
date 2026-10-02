/*
 * concat_copy.hlsl -- channel-axis concatenation (NHWC, axis = 3).
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Port of nss_kernel/conv_rq's sibling `concat_copy.comp`, for the D3D12 path.
 * The GLSL original is the reference; this is a line-by-line translation, and the
 * index arithmetic is deliberately left in the original's shape (see the note on `sq`
 * below) rather than "tidied", because tidying it is how a bit-exact port stops being
 * one.
 * =============================================================================
 * Copies the interior of src into dst at a channel offset. Both buffers carry a 1-pixel
 * border, hence the (x + 1) / (w + 2) terms.
 *
 * Vectorisation, same as the GLSL: each thread moves 16 bytes (one uvec4 = four channel
 * groups), not 4. That quarters the thread count and therefore quarters the fixed
 * index/guard overhead, while keeping access fully coalesced (adjacent x touch adjacent
 * uvec4). It relies on src_c4, dst_c4 and dst_c4_off all being multiples of 4 -- true for
 * this model, where the concat inputs are 16/32 channels and the output is 48, so
 * c4 = 4/8/12. The tail branch covers anything that is not.
 */

// src is a plain SRV, but dst has to be READ-WRITE despite being an output: the tail
// branch does read-modify-write so that a partial group cannot clobber its neighbour.
// That is why dst cannot be declared write-only, exactly as in the GLSL.
ByteAddressBuffer   src : register(t0);
RWByteAddressBuffer dst : register(u0);

// Push constants in the GLSL; root constants here. Field order must match.
struct ConcatPushConstants
{
    uint h;
    uint w;
    uint src_c4;
    uint dst_c4;
    uint dst_c4_off;
};
ConstantBuffer<ConcatPushConstants> pc : register(b0);

[numthreads(8, 8, 1)]
void CSConcatCopy(uint3 gid : SV_DispatchThreadID)
{
    const uint x  = gid.x;
    const uint y  = gid.y;
    const uint c4 = gid.z * 4u;
    if (x >= pc.w || y >= pc.h || c4 >= pc.src_c4)
    {
        return;
    }

    // sIdx/dIdx are in 32-bit WORDS, not uvec4 units: the GLSL indexes `uvec4 d[]`, so
    // `d[sIdx >> 2]` lands at byte offset sIdx * 4. Keeping the same shift here makes the
    // correspondence to the reference obvious; the Load4/Store4 byte offsets below are
    // the word index times 4.
    const uint sIdx = (y + 1u) * (pc.w + 2u) * pc.src_c4 + (x + 1u) * pc.src_c4 + c4;
    const uint dIdx = (y + 1u) * (pc.w + 2u) * pc.dst_c4 + (x + 1u) * pc.dst_c4 + (pc.dst_c4_off + c4);

    const uint sByte = (sIdx >> 2u) * 16u;
    const uint dByte = (dIdx >> 2u) * 16u;

    if (c4 + 4u <= pc.src_c4)
    {
        dst.Store4(dByte, src.Load4(sByte));
    }
    else
    {
        const uint4 s = src.Load4(sByte);
        uint4       o = dst.Load4(dByte);
        [unroll]
        for (uint j = 0u; j < 4u; ++j)
        {
            if (c4 + j < pc.src_c4)
            {
                o[j] = s[j];
            }
        }
        dst.Store4(dByte, o);
    }
}
