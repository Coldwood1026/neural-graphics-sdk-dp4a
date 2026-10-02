/*
 * resize2x.hlsl -- 2x nearest-neighbour upsample (TOSA RESIZE, scale = [4,2,4,2]).
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Port of nss_kernel/`resize2x.comp`, for the D3D12 path. Line-by-line translation of
 * the GLSL reference; the index arithmetic is kept in the original's shape deliberately.
 * =============================================================================
 * Mapping: out(oy, ox) = in(oy/2, ox/2). Both buffers carry a 1-pixel border.
 *
 * THE HALF-PIXEL DETAIL MATTERS. TOSA RESIZE's offset operand is -1 (measured, not
 * assumed), which gives y = oy*2 - 1 -> iy = floor(y/4), then a +1 depending on
 * 2*ry >= 4. Simplified that is iy = oy/2 -- i.e. NOT (oy+1)/2. The two differ only at
 * the edges, which is exactly the kind of difference that survives a spot check and
 * fails a byte comparison.
 */

// src is a plain SRV, but dst must be READ-WRITE: the tail branch does a
// read-modify-write so a partial group cannot overwrite an adjacent pixel or channel.
// Same reason the GLSL omits `writeonly`.
ByteAddressBuffer   src : register(t0);
RWByteAddressBuffer dst : register(u0);

// Push constants in the GLSL; root constants here. Field order must match.
struct ResizePushConstants
{
    uint in_h;   // input interior height, used for the clamp
    uint in_w;   // input interior width
    uint in_c4;
    uint out_h;
    uint out_w;
};
ConstantBuffer<ResizePushConstants> pc : register(b0);

[numthreads(8, 8, 1)]
void CSResize2x(uint3 gid : SV_DispatchThreadID)
{
    const uint ox = gid.x;
    const uint oy = gid.y;
    const uint c4 = gid.z * 4u;
    if (ox >= pc.out_w || oy >= pc.out_h || c4 >= pc.in_c4)
    {
        return;
    }

    const uint sy = min(oy / 2u, pc.in_h - 1u);
    const uint sx = min(ox / 2u, pc.in_w - 1u);

    // Every term is a multiple of 4 (in_c4 % 4 == 0 and c4 % 4 == 0), which is what makes
    // the uvec4-style addressing below legal. sIdx/dIdx are 32-bit word indices; the
    // GLSL's `d[sIdx >> 2]` addresses the same 16 bytes that Load4/Store4 take here.
    const uint sIdx = (sy + 1u) * (pc.in_w + 2u) * pc.in_c4 + (sx + 1u) * pc.in_c4 + c4;
    const uint dIdx = (oy + 1u) * (pc.out_w + 2u) * pc.in_c4 + (ox + 1u) * pc.in_c4 + c4;

    const uint sByte = (sIdx >> 2u) * 16u;
    const uint dByte = (dIdx >> 2u) * 16u;

    if (c4 + 4u <= pc.in_c4)
    {
        dst.Store4(dByte, src.Load4(sByte));
    }
    else
    {
        // Tail group: the extra words read from the source land in the buffer's trailing
        // slack, and the destination is read-modify-written so the partial group cannot
        // clobber the neighbouring pixel or channel.
        const uint4 s = src.Load4(sByte);
        uint4       o = dst.Load4(dByte);
        [unroll]
        for (uint j = 0u; j < 4u; ++j)
        {
            if (c4 + j < pc.in_c4)
            {
                o[j] = s[j];
            }
        }
        dst.Store4(dByte, o);
    }
}
