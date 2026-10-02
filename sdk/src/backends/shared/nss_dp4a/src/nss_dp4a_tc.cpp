/*
 * nss_dp4a_tc.cpp —— Tensor Core 路径的层计划与权重重排实现
 *
 * 权重重排只做一次（上下文创建时），把烘焙头的 packed_w 转成 MMA 的 B 矩阵布局。
 * 两种布局的元素含义必须完全一致，否则数值会错：
 *
 *   烘焙打包：word = packed_w[oc * K4 + (ky*kw+kx)*Cin4 + ic4]
 *             字节 b（0..3）对应通道 ic = ic4*4 + b，即 K 索引
 *             K = (ky, kx, ic)，K4 = kh*kw*Cin4
 *
 *   MMA 布局：wTc[((ky*kw+kx)*cdxCount + cdx)*32*nPad + k*nPad + n]
 *             k ∈ [0,32) 对应通道 ic = cdx*32 + k
 *             n 为输出通道；n ≥ Cout 或 ic ≥ Cin 的位置补 0
 *             （补 0 的权重对累加贡献 0，与「通道补齐用 z_a 激活」配对，
 *               组合出的结果仍是 0 —— 见 conv_tc.comp 的说明）
 */
#include "nss_dp4a_tc.h"

#include <cstdlib>
#include <cstring>

namespace nss {

uint32_t convTcWarps() {
    static uint32_t cached = 0;
    if (cached == 0) {
        const char *e = std::getenv("NSS_DP4A_TC_WARPS");
        uint32_t w = e ? static_cast<uint32_t>(std::atoi(e)) : 1u;
        if (w == 0u || w > 8u) w = 1u;      /* 内核 shared 切片按 8 个 warp 预留上限 */
        cached = w;
    }
    return cached;
}

/* Dispatch 不直接存 kh/kw —— 它们由模型加载器写进 push[10]/push[11]（见
 * nss_dp4a_model.cpp 的 conv 分支）。这里取出来做别名，避免魔法下标散落。 */
static inline uint32_t dKh(const Dispatch &d) { return d.push[10]; }
static inline uint32_t dKw(const Dispatch &d) { return d.push[11]; }

bool tcShapeSupported(const Dispatch &d) {
    if (d.kind != 0) return false;                 /* 只有 conv 有 MMA 版 */
    if (d.inShape.c == 0 || d.outShape.c == 0) return false;
    if (d.inShape.c % 4u != 0) return false;       /* 激活按 4 通道打包 */
    const uint32_t nPad = ((d.outShape.c + 15u) / 16u) * 16u;
    if (nPad / 16u > 4u) return false;             /* 内核里只有 4 个累加器 */
    if (d.outShape.w == 0 || d.outShape.h == 0) return false;
    return true;
}

bool buildTcLayerPlan(const Dispatch &d, TcLayerPlan *out, std::vector<uint8_t> *wOut) {
    if (!out || !wOut || !tcShapeSupported(d) || !d.w) return false;

    const uint32_t Cin  = d.inShape.c;
    const uint32_t Cout = d.outShape.c;
    const uint32_t Cin4 = Cin / 4u;
    const uint32_t kh   = dKh(d);
    const uint32_t kw   = dKw(d);
    const uint32_t K4   = kh * kw * Cin4;
    if (K4 == 0) return false;

    out->cdxCount = (Cin + 31u) / 32u;
    out->nPad     = ((Cout + 15u) / 16u) * 16u;
    out->nBlocks  = out->nPad / 16u;
    out->gridX    = (d.outShape.w + 15u) / 16u;
    const uint32_t warps = convTcWarps();
    out->gridY    = (d.outShape.h + warps - 1u) / warps;
    out->wBytes   = static_cast<uint64_t>(kh) * static_cast<uint64_t>(kw)
                    * out->cdxCount * 32u * out->nPad;

    wOut->assign(static_cast<size_t>(out->wBytes), 0u);

    for (uint32_t ky = 0; ky < kh; ++ky) {
        for (uint32_t kx = 0; kx < kw; ++kx) {
            const uint32_t kxy     = ky * kw + kx;
            const uint32_t srcBase = kxy * Cin4;      /* 该 (ky,kx) 在 K4 里的起点 */
            for (uint32_t cdx = 0; cdx < out->cdxCount; ++cdx) {
                const uint32_t chunk = kxy * out->cdxCount + cdx;
                uint8_t *dst = wOut->data()
                             + static_cast<size_t>(chunk) * 32u * out->nPad;
                for (uint32_t k = 0; k < 32u; ++k) {
                    const uint32_t ic = cdx * 32u + k;
                    if (ic >= Cin) break;             /* 其余通道保持 0 */
                    const uint32_t ic4  = ic >> 2;
                    const uint32_t byte = ic & 3u;
                    uint8_t *row = dst + static_cast<size_t>(k) * out->nPad;
                    const uint32_t srcK4 = srcBase + ic4;
                    for (uint32_t n = 0; n < Cout; ++n) {
                        const uint32_t word = d.w[n * K4 + srcK4];
                        row[n] = static_cast<uint8_t>((word >> (byte * 8u)) & 0xFFu);
                    }
                }
            }
        }
    }
    return true;
}

}  /* namespace nss */
