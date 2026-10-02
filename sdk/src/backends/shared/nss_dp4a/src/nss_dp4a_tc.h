/*
 * nss_dp4a_tc.h —— Tensor Core（int8 cooperative matrix）推理路径的辅助层
 *
 * 与 DP4A 路径共用：激活布局（NHWC uint32）、RESCALE 参数（bc/mult/shift/lut）、
 * 缓冲拓扑、dispatch 表。只有 conv 的内核与权重布局不同：
 *
 *   DP4A：packed_w[oc][K/4]，K 方向每 4 个 int8 打成一个 uint32，
 *         内核用 OpSDot 一次算 4 个 int8 MAC。
 *   TC  ：wTc[chunk][k][n]，chunk = (ky*kw+kx)*cdxCount + cdx（每块 32 个通道），
 *         n = 输出通道（补齐到 16 的倍数），B 矩阵行跨距 = nPad。
 *         内核一次 MMA 算 16×16×32 = 8192 个 MAC，且 int32 累加精确，
 *         结果与 DP4A 逐位一致。
 */
#ifndef NSS_DP4A_TC_H
#define NSS_DP4A_TC_H

#include "nss_dp4a_model.h"

#include <cstdint>
#include <vector>

namespace nss {

/* 每个 workgroup 的 warp 数。必须与 conv_tc.comp 的 #define WARPS 一致
 * （内核用它把行号摊到多个 warp，这里用它把 dispatch 的 gridY 缩小相应倍数）。
 * NSS_DP4A_TC_WARPS 可在测量时覆盖（只影响 gridY 与内核的行分配，不改数值）。 */
uint32_t convTcWarps();
inline constexpr uint32_t kConvTcWarps = 1;   /* 缺省值（保留给不知道运行期值的地方） */

/* Tensor Core 路径每层需要的几何参数 */
struct TcLayerPlan {
    uint32_t cdxCount = 0;   /* ceil(Cin / 32)；每块 32 个通道 */
    uint32_t nPad = 0;       /* round_up(Cout, 16) */
    uint32_t nBlocks = 0;    /* nPad / 16（内核最多支持 4） */
    uint32_t gridX = 0;      /* ceil(out_w / 16) */
    uint32_t gridY = 0;      /* ceil(out_h / kConvTcWarps) */
    uint64_t wBytes = 0;     /* 重排后权重字节数 */
};

/* 该层的形状能否走 TC 内核（目前只限制「输出通道数 ≤ 64」这一条）。 */
bool tcShapeSupported(const Dispatch &d);

/* 生成层计划并把烘焙权重重排进 wOut。
 *   wOut 的大小 = kh*kw*cdxCount*32*nPad 字节，布局 [chunk][k][n]。 */
bool buildTcLayerPlan(const Dispatch &d, TcLayerPlan *out, std::vector<uint8_t> *wOut);

}  /* namespace nss */

#endif /* NSS_DP4A_TC_H */
