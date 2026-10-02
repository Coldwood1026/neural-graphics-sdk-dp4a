#include "nss_dp4a_model.h"

#include <cstring>
#include <string>

/* 烘焙产物。modelId 选择用哪一份。 */
#include "nss_model_data_high.h"
#include "nss_model_plan_high.h"
#include "nss_model_data_midlow.h"
#include "nss_model_plan_midlow.h"

namespace nss {

namespace {

/* 取某一层的描述：两套模型结构一致（19 层），只有 KPN 通道数不同，
 * 所以层表可以按模型选一份，几何字段逐个拷过来。
 * 注意 LayerDesc 定义在 nss_baked 命名空间 —— 两个模型共用同一类型，
 * 否则 nss_baked::LayerDesc 与 nss_model_midlow::LayerDesc 是不同类型，无法互转。 */
struct ModelTables {
    const nss_baked::LayerDesc *layers;
    uint32_t layerCount;
    uint32_t bufferCount;
    const nss_baked::OutDesc *outputs;
    uint32_t outputCount;
};

ModelTables pickTables(int modelId) {
    if (modelId == 0) {
        return { nss_model::kLayers, nss_model::kDispatchTotal,
                 nss_model::kBufferCount,
                 nss_model::kOutputs, nss_model::kOutputCount };
    }
    return { nss_model_midlow::kLayers, nss_model_midlow::kDispatchTotal,
             nss_model_midlow::kBufferCount,
             nss_model_midlow::kOutputs, nss_model_midlow::kOutputCount };
}

/* 输出通道数：conv 层直接给；其余算子靠回溯。烘焙表里已把 concat 的输入通道存好，
 * 这里只需要按 dispatch 顺序传播即可 —— 每个 dispatch 的 cout 要么来自 conv 权重，
 * 要么等于其输入。 */
uint32_t outChannelsOf(const nss_baked::LayerDesc &L, const std::vector<Shape> &bufs) {
    switch (L.kind) {
        case nss_baked::KERNEL_CONV:   return static_cast<uint32_t>(L.cout);
        case nss_baked::KERNEL_RESIZE: return bufs[L.inBuf].c;
        case nss_baked::KERNEL_CONCAT: {
            uint32_t sum = 0;
            for (int i = 0; i < L.nInputs; ++i) {
                sum += static_cast<uint32_t>(L.concatChans[i]);
            }
            return sum;
        }
        default: return 0;
    }
}

}  /* namespace */

bool Model::load(int modelId, uint32_t width, uint32_t height, std::string *err) {
    if (width == 0 || height == 0) {
        if (err) *err = "输入尺寸为 0";
        return false;
    }
    if ((width % 8) != 0 || (height % 8) != 0) {
        if (err) *err = "输入宽高必须是 8 的倍数（NSS 约束）";
        return false;
    }
    width_ = width;
    height_ = height;

    const ModelTables T = pickTables(modelId);
    buffers_.assign(T.bufferCount, Shape{});
    dispatches_.clear();
    dispatches_.reserve(T.layerCount);

    /* 图输入占一个逻辑位置：用 inBuf == -1 表示，形状单独记 */
    const Shape inputShape{ height, width, 12 };

    for (uint32_t i = 0; i < T.layerCount; ++i) {
        const nss_baked::LayerDesc &L = T.layers[i];
        Dispatch d;
        d.kind = L.kind;
        d.inBuf = L.inBuf;
        d.outBuf = L.outBuf;
        d.nInputs = L.nInputs;
        d.concatBufs = L.concatBufs;
        d.w = L.w;
        d.bc = L.bc;
        d.mult = L.mult;
        d.shift = L.shift;
        d.lut = L.lut;
        d.wWords = L.wWords;
        d.multCount = L.multCount;

        const Shape inShape = (L.inBuf < 0) ? inputShape : buffers_[L.inBuf];
        d.inShape = inShape;

        switch (L.kind) {
            case nss_baked::KERNEL_CONV: {
                /* out_h = (in_h + padT + padB - kh) / strideH + 1
                 * 烘焙表只存了 padT/padL。对 3x3、stride 1/2 的对称 padding，
                 * padB == padT、padR == padL；但 stride=2 时 padT=1, padB=0，
                 * 此时 padB = padT - 1。判据：stride>1 且 out 尺寸需对齐。
                 * 更稳妥的做法是直接按 nss_ref.py 的同一算式复现：
                 *   pad 由 VGF 给出 [pt, pb, pl, pr]，我们已烘焙 pt/pl，
                 *   pb = (strideH - 1) - pt + 1 ... 不通用。
                 * 因此这里改成用「输出尺寸必须满足的整除关系」反推：
                 *   out_h = floor((in_h + pt + pb - kh) / sh) + 1
                 * 对 NSS 的实际配置，pt/pb 与 pl/pr 的取值只有两种：
                 *   stride=1 -> pad=(1,1,1,1)   即 pt=pb=pl=pr=1
                 *   stride=2 -> pad=(1,0,1,0)   即 pt=pl=1, pb=pr=0
                 * 故 pb = (strideH == 1) ? pt : pt - 1，同理 pr。 */
                const uint32_t pt = static_cast<uint32_t>(L.padT);
                const uint32_t pl = static_cast<uint32_t>(L.padL);
                const uint32_t pb = (L.strideH == 1) ? pt : (pt > 0 ? pt - 1 : 0);
                const uint32_t pr = (L.strideW == 1) ? pl : (pl > 0 ? pl - 1 : 0);
                const uint32_t kh = static_cast<uint32_t>(L.kh);
                const uint32_t kw = static_cast<uint32_t>(L.kw);
                const uint32_t sh = static_cast<uint32_t>(L.strideH);
                const uint32_t sw = static_cast<uint32_t>(L.strideW);

                const uint32_t oh = (inShape.h + pt + pb - kh) / sh + 1;
                const uint32_t ow = (inShape.w + pl + pr - kw) / sw + 1;
                buffers_[d.outBuf] = Shape{ oh, ow, static_cast<uint32_t>(L.cout) };
                d.outShape = buffers_[d.outBuf];

                /* push: in_w, in_c4, out_h, out_w, out_c4, padT, padL,
                 *       strideH, strideW, out_zp, kh, kw, has_lut, dbg */
                d.push[0] = inShape.w;
                d.push[1] = inShape.c / 4;
                d.push[2] = oh;
                d.push[3] = ow;
                d.push[4] = static_cast<uint32_t>(L.cout) / 4;
                d.push[5] = static_cast<uint32_t>(L.padT);
                d.push[6] = static_cast<uint32_t>(L.padL);
                d.push[7] = sh;
                d.push[8] = sw;
                d.push[9] = L.outZp;
                d.push[10] = kh;
                d.push[11] = kw;
                d.push[12] = static_cast<uint32_t>(L.hasLut);
                d.push[13] = 0;                       /* dbg */
                /* [14]=inH [15]=inBorded [16]=outBorded：由 recordFrame
                 * 按「本层是否直接绑定宿主紧致缓冲」覆盖。 */
                d.pushSize = 17;
                d.gx = ow;
                d.gy = oh;
                d.gz = static_cast<uint32_t>(L.cout) / 4;
                break;
            }

            case nss_baked::KERNEL_RESIZE: {
                /* 最近邻上采样，scale = [n, d, n, d]。NSS 为 2x（[4,2,4,2]）。
                 * TOSA 的 offset 实测为 -1，化简后索引 = oy / (n/d) 的整除。 */
                const uint32_t n = static_cast<uint32_t>(L.scaleN);
                const uint32_t dd = static_cast<uint32_t>(L.scaleD);
                const uint32_t oh = inShape.h * n / dd;
                const uint32_t ow = inShape.w * n / dd;
                buffers_[d.outBuf] = Shape{ oh, ow, inShape.c };
                d.outShape = buffers_[d.outBuf];

                /* push: in_h, in_w, in_c4, out_h, out_w */
                d.push[0] = inShape.h;
                d.push[1] = inShape.w;
                d.push[2] = inShape.c / 4;
                d.push[3] = oh;
                d.push[4] = ow;
                d.pushSize = 5;
                d.gx = ow;
                d.gy = oh;
                /* resize 内核按 uvec4 搬（每次 4 个通道组）⇒ z 是 c4/4 的上取整 */
                d.gz = (inShape.c / 4 + 3u) / 4u;
                break;
            }

            case nss_baked::KERNEL_CONCAT: {
                /* 通道轴拼接。各输入的空间尺寸必须一致。 */
                uint32_t sum = 0;
                const Shape first = (L.concatBufs[0] < 0) ? inputShape : buffers_[L.concatBufs[0]];
                for (int j = 0; j < L.nInputs; ++j) {
                    const int b = L.concatBufs[j];
                    const Shape s = (b < 0) ? inputShape : buffers_[b];
                    sum += s.c;
                }
                buffers_[d.outBuf] = Shape{ first.h, first.w, sum };
                d.outShape = buffers_[d.outBuf];

                /* concat 的每个输入各发一次 dispatch（原实现就是这样），
                 * 这里把多次 dispatch 展开。push: h, w, src_c4, dst_c4, dst_c4_off */
                uint32_t off = 0;
                for (int j = 0; j < L.nInputs; ++j) {
                    const int b = L.concatBufs[j];
                    const Shape s = (b < 0) ? inputShape : buffers_[b];
                    Dispatch dd2 = d;
                    dd2.inBuf = b;
                    dd2.inShape = s;
                    dd2.push[0] = first.h;
                    dd2.push[1] = first.w;
                    dd2.push[2] = s.c / 4;
                    dd2.push[3] = sum / 4;
                    dd2.push[4] = off / 4;
                    dd2.pushSize = 5;
                    dd2.gx = first.w;
                    dd2.gy = first.h;
                    /* concat 内核同样按 uvec4 搬 ⇒ z 是 c4/4 的上取整 */
                    dd2.gz = (s.c / 4 + 3u) / 4u;
                    dispatches_.push_back(dd2);
                    off += s.c;
                }
                continue;      /* 已展开，跳过下面的统一 push 拷贝 */
            }

            default:
                if (err) *err = "未知层类型";
                return false;
        }
        dispatches_.push_back(d);
    }

    /* 输出缓冲：直接用烘焙表给的下标 */
    outputBufs_.clear();
    for (uint32_t i = 0; i < T.outputCount; ++i) {
        outputBufs_.push_back(T.outputs[i].buf);
    }

    /* 校验：所有被引用的缓冲形状都已填好 */
    for (const auto &d2 : dispatches_) {
        if (d2.outShape.c == 0 || d2.outShape.h == 0 || d2.outShape.w == 0) {
            if (err) *err = "形状推导产生零尺寸";
            return false;
        }
        if ((d2.outShape.c % 4) != 0) {
            if (err) *err = "输出通道数不是 4 的倍数（DP4A 打包要求）";
            return false;
        }
    }
    return true;
}

uint64_t Model::scratchBytes() const {
    /* 每个缓冲需要 (h+2)*(w+2)*c 字节（四周 1 像素边框），另加尾部冗余
     * （见 nss_dp4a_model.h 的 kTensorTailSlack：Tensor Core 内核的补齐块
     *   会多读几个 word）。 */
    uint64_t total = 0;
    for (const Shape &s : buffers_) {
        if (s.h == 0) continue;
        total += static_cast<uint64_t>(s.h + 2) * (s.w + 2) * s.c + kTensorTailSlack;
    }
    return total;
}

}  /* namespace nss */
