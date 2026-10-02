// 自动生成，请勿手改。模型: high
// 逐层几何参数（尺寸无关）。张量形状由 nss_dp4a_model.cpp 在运行时推导。
#pragma once

#include "nss_model_common.h"

namespace nss_model {

using nss_baked::LayerDesc;
using nss_baked::KernelKind;
using nss_baked::KERNEL_CONV;
using nss_baked::KERNEL_RESIZE;
using nss_baked::KERNEL_CONCAT;

extern const unsigned int op00_w[];
extern const int op00_bc[];
extern const int op00_m[];
extern const int op00_s[];
extern const int op00_lut[];
extern const unsigned int op02_w[];
extern const int op02_bc[];
extern const int op02_m[];
extern const int op02_s[];
extern const int op02_lut[];
extern const unsigned int op04_w[];
extern const int op04_bc[];
extern const int op04_m[];
extern const int op04_s[];
extern const int op04_lut[];
extern const unsigned int op06_w[];
extern const int op06_bc[];
extern const int op06_m[];
extern const int op06_s[];
extern const int op06_lut[];
extern const unsigned int op08_w[];
extern const int op08_bc[];
extern const int op08_m[];
extern const int op08_s[];
extern const int op08_lut[];
extern const unsigned int op10_w[];
extern const int op10_bc[];
extern const int op10_m[];
extern const int op10_s[];
extern const int op10_lut[];
extern const unsigned int op12_w[];
extern const int op12_bc[];
extern const int op12_m[];
extern const int op12_s[];
extern const int op12_lut[];
extern const unsigned int op15_w[];
extern const int op15_bc[];
extern const int op15_m[];
extern const int op15_s[];
extern const int op15_lut[];
extern const unsigned int op18_w[];
extern const int op18_bc[];
extern const int op18_m[];
extern const int op18_s[];
extern const int op18_lut[];
extern const unsigned int op20_w[];
extern const int op20_bc[];
extern const int op20_m[];
extern const int op20_s[];
extern const int op20_lut[];
extern const unsigned int op24_w[];
extern const int op24_bc[];
extern const int op24_m[];
extern const int op24_s[];
extern const int op24_lut[];
extern const unsigned int op27_w[];
extern const int op27_bc[];
extern const int op27_m[];
extern const int op27_s[];
extern const int op27_lut[];
extern const unsigned int op29_w[];
extern const int op29_bc[];
extern const int op29_m[];
extern const int op29_s[];
extern const int op29_lut[];
extern const unsigned int op32_w[];
extern const int op32_bc[];
extern const int op32_m[];
extern const int op32_s[];
extern const int op32_lut[];

static const int kConcatBufs_9[2] = { 8, 3 };
static const int kConcatChans_9[2] = { 16, 32 };
static const int kConcatBufs_14[2] = { 13, 1 };
static const int kConcatChans_14[2] = { 16, 32 };

static const nss_baked::LayerDesc kLayers[19] = {
    { .kind = KERNEL_CONV, .inBuf = -1, .outBuf = 0, .cin = 12, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 2, .strideW = 2, .outZp = 4294967168u, .hasLut = 0, .w = op00_w, .bc = op00_bc, .mult = op00_m, .shift = op00_s, .lut = op00_lut, .wWords = 864u, .multCount = 32u },
    { .kind = KERNEL_CONV, .inBuf = 0, .outBuf = 1, .cin = 32, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op02_w, .bc = op02_bc, .mult = op02_m, .shift = op02_s, .lut = op02_lut, .wWords = 2304u, .multCount = 32u },
    { .kind = KERNEL_CONV, .inBuf = 1, .outBuf = 2, .cin = 32, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 2, .strideW = 2, .outZp = 4294967168u, .hasLut = 0, .w = op04_w, .bc = op04_bc, .mult = op04_m, .shift = op04_s, .lut = op04_lut, .wWords = 2304u, .multCount = 32u },
    { .kind = KERNEL_CONV, .inBuf = 2, .outBuf = 3, .cin = 32, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op06_w, .bc = op06_bc, .mult = op06_m, .shift = op06_s, .lut = op06_lut, .wWords = 2304u, .multCount = 32u },
    { .kind = KERNEL_CONV, .inBuf = 3, .outBuf = 4, .cin = 32, .cout = 64, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 2, .strideW = 2, .outZp = 4294967168u, .hasLut = 0, .w = op08_w, .bc = op08_bc, .mult = op08_m, .shift = op08_s, .lut = op08_lut, .wWords = 4608u, .multCount = 64u },
    { .kind = KERNEL_CONV, .inBuf = 4, .outBuf = 5, .cin = 64, .cout = 64, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op10_w, .bc = op10_bc, .mult = op10_m, .shift = op10_s, .lut = op10_lut, .wWords = 9216u, .multCount = 64u },
    { .kind = KERNEL_CONV, .inBuf = 5, .outBuf = 6, .cin = 64, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op12_w, .bc = op12_bc, .mult = op12_m, .shift = op12_s, .lut = op12_lut, .wWords = 4608u, .multCount = 32u },
    { .kind = KERNEL_RESIZE, .inBuf = 6, .outBuf = 7, .scaleN = 4, .scaleD = 2 },
    { .kind = KERNEL_CONV, .inBuf = 7, .outBuf = 8, .cin = 32, .cout = 16, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op15_w, .bc = op15_bc, .mult = op15_m, .shift = op15_s, .lut = op15_lut, .wWords = 1152u, .multCount = 16u },
    { .kind = KERNEL_CONCAT, .inBuf = 8, .outBuf = 9, .nInputs = 2, .concatBufs = kConcatBufs_9, .concatChans = kConcatChans_9 },
    { .kind = KERNEL_CONV, .inBuf = 9, .outBuf = 10, .cin = 48, .cout = 32, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op18_w, .bc = op18_bc, .mult = op18_m, .shift = op18_s, .lut = op18_lut, .wWords = 3456u, .multCount = 32u },
    { .kind = KERNEL_CONV, .inBuf = 10, .outBuf = 11, .cin = 32, .cout = 36, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 83u, .hasLut = 1, .w = op20_w, .bc = op20_bc, .mult = op20_m, .shift = op20_s, .lut = op20_lut, .wWords = 2592u, .multCount = 36u },
    { .kind = KERNEL_RESIZE, .inBuf = 10, .outBuf = 12, .scaleN = 4, .scaleD = 2 },
    { .kind = KERNEL_CONV, .inBuf = 12, .outBuf = 13, .cin = 32, .cout = 16, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op24_w, .bc = op24_bc, .mult = op24_m, .shift = op24_s, .lut = op24_lut, .wWords = 1152u, .multCount = 16u },
    { .kind = KERNEL_CONCAT, .inBuf = 13, .outBuf = 14, .nInputs = 2, .concatBufs = kConcatBufs_14, .concatChans = kConcatChans_14 },
    { .kind = KERNEL_CONV, .inBuf = 14, .outBuf = 15, .cin = 48, .cout = 16, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op27_w, .bc = op27_bc, .mult = op27_m, .shift = op27_s, .lut = op27_lut, .wWords = 1728u, .multCount = 16u },
    { .kind = KERNEL_CONV, .inBuf = 15, .outBuf = 16, .cin = 16, .cout = 16, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 4294967168u, .hasLut = 0, .w = op29_w, .bc = op29_bc, .mult = op29_m, .shift = op29_s, .lut = op29_lut, .wWords = 576u, .multCount = 16u },
    { .kind = KERNEL_RESIZE, .inBuf = 16, .outBuf = 17, .scaleN = 4, .scaleD = 2 },
    { .kind = KERNEL_CONV, .inBuf = 17, .outBuf = 18, .cin = 16, .cout = 4, .kh = 3, .kw = 3, .padT = 1, .padL = 1, .strideH = 1, .strideW = 1, .outZp = 44u, .hasLut = 1, .w = op32_w, .bc = op32_bc, .mult = op32_m, .shift = op32_s, .lut = op32_lut, .wWords = 144u, .multCount = 4u },
};

}  // namespace nss_model
