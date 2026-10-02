/*
 * nss_dp4a_model.h —— 运行时形状推导与层表访问
 *
 * 烘焙头文件只含「尺寸无关」的几何参数；张量形状在这里按实际输入 H/W 推导。
 * 网络只有 5 个算子、通道与核全固定，唯一变量是 H/W，所以推导就是十几行算术。
 */
#ifndef NSS_DP4A_MODEL_H
#define NSS_DP4A_MODEL_H

#include <cstdint>
#include <string>
#include <vector>

namespace nss {

/* 张量形状（NHWC） */
struct Shape {
    uint32_t h = 0, w = 0, c = 0;
};

/*
 * 每个中间张量分配时在尾部额外留出的冗余字节。
 *
 * Tensor Core 内核用「权重补 0 ⇒ 越界 word × 0 = 0」的技巧省掉了通道补齐位的
 * 掩码，代价是当 Cin 不是 32 的倍数时（12/16/48），最后一个 cdx 块会**多读**
 * 最多 (8 - Cin/4 的余数) 个 word。给每个缓冲留一点尾巴，这些读就落在已分配
 * 内存里，不再是越界访问。
 */
inline constexpr uint64_t kTensorTailSlack = 64;

/* 一次 dispatch 的完整运行期描述（几何参数来自烘焙表，形状来自推导） */
struct Dispatch {
    int kind = 0;                  /* 0=conv 1=resize 2=concat */
    int inBuf = -1;                /* -1 表示图输入 */
    int outBuf = 0;
    Shape inShape, outShape;
    uint32_t gx = 0, gy = 0, gz = 0;
    uint32_t push[20] = {};
    uint32_t pushSize = 0;         /* 以 4 字节为单位 */
    /* conv 专用 */
    const uint32_t *w = nullptr;
    const int32_t *bc = nullptr;
    const int32_t *mult = nullptr;
    const int32_t *shift = nullptr;
    const int32_t *lut = nullptr;
    uint32_t wWords = 0, multCount = 0;
    /* concat 专用 */
    int nInputs = 0;
    const int *concatBufs = nullptr;
};

/* 模型句柄：持有缓冲表与 dispatch 表 */
class Model {
public:
    /* 从烘焙头文件装载。modelId: 0=high 1=mid_low */
    bool load(int modelId, uint32_t width, uint32_t height, std::string *err);

    const std::vector<Shape> &buffers() const { return buffers_; }
    const std::vector<Dispatch> &dispatches() const { return dispatches_; }
    /* 图输出所在的缓冲下标（两个：KPN、时序） */
    const std::vector<int> &outputBufs() const { return outputBufs_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t inputChannels() const { return 12; }

    /* 内部临时缓冲总字节数（所有中间张量） */
    uint64_t scratchBytes() const;

private:
    std::vector<Shape> buffers_;
    std::vector<Dispatch> dispatches_;
    std::vector<int> outputBufs_;
    uint32_t width_ = 0, height_ = 0;
};

}  /* namespace nss */

#endif /* NSS_DP4A_MODEL_H */
