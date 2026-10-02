// 自动生成，请勿手改。所有烘焙模型共用的类型定义。
#pragma once

namespace nss_baked {

enum KernelKind { KERNEL_CONV = 0, KERNEL_RESIZE = 1, KERNEL_CONCAT = 2 };

struct OutDesc { int buf; };

struct LayerDesc {
    int kind;              // KernelKind
    int inBuf;             // 缓冲下标，-1 = 图输入
    int outBuf;
    int cin, cout;         // 通道数（conv/resize）
    int kh, kw;            // 卷积核
    int padT, padL;
    int strideH, strideW;
    unsigned int outZp;    // RESCALE 输出零点（-128 的原始字节 = ReLU 标记）
    int hasLut;            // 是否走 sigmoid 查表
    int scaleN, scaleD;    // resize 比例
    int nInputs;           // concat 输入数
    const int *concatBufs; // concat 各输入的缓冲下标
    const int *concatChans;
    // 常量缓冲（仅 conv）
    const unsigned int *w;
    const int *bc, *mult, *shift, *lut;
    unsigned int wWords, multCount;
};

}  // namespace nss_baked
