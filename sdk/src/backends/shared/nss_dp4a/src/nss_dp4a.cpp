/*
 * nss_dp4a.cpp —— Arm NSS 神经超分的 DP4A 推理后端实现
 *
 * 纯推理形态：宿主拥有 VkDevice/VkQueue/VkCommandBuffer，本库只建管线、录 dispatch。
 *
 * 关键实现要点（均来自实测，见 nss-vk/README.md）：
 *   - 激活缓冲四周预填 0x80（= 激活零点 -128），与「跳过越界 tap」严格等价
 *   - 融合 conv+bias+DP4A修正+RESCALE(+LUT)，35 个 TOSA 算子 -> 19 次 dispatch
 *   - out_zp = -128 + 钳位天然实现 ReLU，内核里不再插 max(x,0)
 *   - requantize 用 int64（acc×multiplier 可达 56 位）
 */
#include "nss_dp4a.h"

#include "nss_dp4a_vk.h"
#include "nss_dp4a_model.h"
#include "nss_spirv_embed.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nss_dp4a_tc.h"

using namespace nss;

/* ------------------------------------------------------------------ 内部结构 */

namespace {

constexpr uint32_t kDescriptorBindingCount = 8;
constexpr uint32_t kPushConstantSize = 80;      /* conv 用 17 个 uint32 = 68，留余量 */

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t size = 0;
    void *mapped = nullptr;
};

struct ContextImpl {
    NssDp4aCreateInfo ci{};
    VkFuncs vk{};
    VkLoader loader{};
    DeviceCaps caps{};
    Model model;

    /* 类型化的 Vulkan 句柄（公开 API 用 uint64_t 传递，避免头文件依赖 vulkan.h） */
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    /* 外部上传命令缓冲（SRAPI 流程）：非空时权重/scratch 的初始化拷贝录进它 */
    VkCommandBuffer externalUploadCmd = VK_NULL_HANDLE;

    /* 内部资源 */
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pll = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkPipeline pipeConv = VK_NULL_HANDLE;
    /* 小输出层（out_c4==1，即 cout=4）专用：每线程 2×2 像素 + shared 协作载入
     * 4×4 tap 窗口。该类层原内核每线程只产 1 word，固定开销占比过高。 */
    VkPipeline pipeConv2x2 = VK_NULL_HANDLE;
    VkPipeline pipeResize = VK_NULL_HANDLE;
    VkPipeline pipeConcat = VK_NULL_HANDLE;

    /* ---- Tensor Core 推理路径 ----
     * useTc 为真时 conv 走 pipeConvTc（int8 cooperative matrix MMA），
     * 权重用 tcWeightBufs / tcPlans（[chunk][k][n] 布局），其余资源全部共用。
     * 设备不支持 cooperative matrix 或强制覆盖为 dp4a 时，useTc=false，
     * 完全退回原来的 DP4A 路径。 */
    bool useTc = false;
    VkPipeline pipeConvTc = VK_NULL_HANDLE;
    /* 走 MMA 的最低输出通道数（低于它的层用 DP4A）。
     * ★ 2026-10-01 实测重扫（1280x664，三次取最小）：
     *     0(全MMA)→2.977  8→2.555  **16→2.499**  32→2.818  64→3.534  999(全DP4A)→3.717
     * 阈值 16 与 8 选中同一批层（只有末层 cout=4 用 DP4A），差异是噪声。
     * 旧默认 32 是在「im2col 暂存版 TC 内核」下调出来的；改成直连 coopMatLoad 后
     * 结论反转 —— cout=16 的四个层（L8/L13/L15/L16）用 MMA 明显更快。
     * 逐层隔离 A/B 佐证（已是含 0.249 底线的绝对值）：
     *     L1  dp4a 0.966 / tc 0.544 ｜ L13(→di14) 0.616 / 0.405
     *     L15(→di17) 0.795 / 0.528 ｜ 末层 L18(→di20) 0.554 / 0.924
     * 即除末层外 MMA 全面占优，故 16 是当前最优。 */
    uint32_t tcMinCout = 16u;
    std::vector<TcLayerPlan> tcPlans;        /* 下标与 model.dispatches() 对齐 */
    std::vector<Buffer> tcWeightBufs;        /* 同上；非 conv 层为空 */

    /* 中间张量：每个 dispatch 一个输出缓冲（含 1 像素边框） */
    std::vector<Buffer> scratchBufs;         /* 下标与 model.buffers() 对齐 */
    Buffer inputMirror;                      /* 宿主没给 scratch 时，输入也要有边框 */
    bool ownsScratch = false;

    /* 权重与量化参数常驻缓冲（每个 conv 一套） */
    struct LayerConst {
        Buffer w, bc, mult, shift, lut;
    };
    std::vector<LayerConst> layerConsts;     /* 下标与 model.dispatches() 对齐 */

    uint64_t internalBytes = 0;
    bool ready = false;
    uint32_t diagFrames = 0;                 /* 前几帧打印诊断 */
    /* per-dispatch GPU timestamps (diagnostic) */
    VkQueryPool tsPool = VK_NULL_HANDLE;
    uint32_t tsCount = 0;          /* queries written for the current frame */
    double tsPeriodNs = 1.0;       /* limits.timestampPeriod */

    /* 外部上传模式下的 staging 缓冲延迟释放队列。
     *
     * uploadBuffer 若把拷贝命令录进宿主的外部命令缓冲，此时命令尚未提交；
     * 若立即 destroyBuffer(staging)，等到真正 vkQueueSubmit 时 GPU 会访问
     * 已释放的 VkBuffer / VkDeviceMemory —— 复用后表现为读到垃圾数据或
     * 直接 VK_ERROR_DEVICE_LOST(-4)，游戏卡住十几秒后整个设备失效。
     *
     * 因此外部上传模式下把 staging 挂到这里，由宿主在提交并等待完成后
     * 调用 nssDp4aFlushPendingUploads() 释放。 */
    std::vector<Buffer> pendingUploads;
};

/* ------------------------------------------------------------------ 日志 */
void logMsg(ContextImpl *ctx, int level, const char *msg) {
    if (ctx && ctx->ci.logCallback) {
        ctx->ci.logCallback(level, msg, ctx->ci.logUserData);
    }
}

/* ------------------------------------------------------------------ 缓冲 */

bool createBuffer(ContextImpl *ctx, uint64_t size, VkBufferUsageFlags usage,
                  bool hostVisible, Buffer *out, std::string *err) {
    const VkFuncs &vk = ctx->vk;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size ? size : 4;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk.vkCreateBuffer(ctx->device, &bci, nullptr, &out->buffer) != VK_SUCCESS) {
        if (err) *err = "vkCreateBuffer 失败";
        return false;
    }

    VkMemoryRequirements req{};
    vk.vkGetBufferMemoryRequirements(ctx->device, out->buffer, &req);

    const VkMemoryPropertyFlags want = hostVisible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t mt = findMemoryType(vk, ctx->physDev, req.memoryTypeBits, want);
    if (mt == UINT32_MAX) {
        if (err) *err = "找不到合适的内存类型";
        return false;
    }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    if (vk.vkAllocateMemory(ctx->device, &mai, nullptr, &out->memory) != VK_SUCCESS) {
        if (err) *err = "vkAllocateMemory 失败";
        return false;
    }
    vk.vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);
    out->size = req.size;
    if (hostVisible) {
        vk.vkMapMemory(ctx->device, out->memory, 0, req.size, 0, &out->mapped);
    }
    return true;
}

void destroyBuffer(ContextImpl *ctx, Buffer *b) {
    if (!b || !ctx) return;
    const VkFuncs &vk = ctx->vk;
    if (b->mapped) { vk.vkUnmapMemory(ctx->device, b->memory); b->mapped = nullptr; }
    if (b->buffer) { vk.vkDestroyBuffer(ctx->device, b->buffer, nullptr); b->buffer = VK_NULL_HANDLE; }
    if (b->memory) { vk.vkFreeMemory(ctx->device, b->memory, nullptr); b->memory = VK_NULL_HANDLE; }
}

/* 上传数据到设备本地缓冲（经 staging）。
 * 外部上传模式（SRAPI 流程）：录制进宿主的命令缓冲，由宿主统一提交；
 * 独立测试模式：自建一次性命令缓冲并用 queue 提交等待。 */
bool uploadBuffer(ContextImpl *ctx, Buffer *dst, const void *data, uint64_t bytes,
                  std::string *err) {
    Buffer staging{};
    if (!createBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, &staging, err)) {
        return false;
    }
    std::memcpy(staging.mapped, data, static_cast<size_t>(bytes));

    VkBufferCopy region{};
    region.size = bytes;

    bool ok = true;
    if (ctx->externalUploadCmd != VK_NULL_HANDLE) {
        ctx->vk.vkCmdCopyBuffer(ctx->externalUploadCmd, staging.buffer, dst->buffer, 1, &region);
        /* 命令只录不提交：staging 不能立即释放，否则提交时 GPU 读到已释放内存。
         * 移交给延迟队列，等宿主提交完成后再统一销毁。 */
        ctx->pendingUploads.push_back(staging);
        return true;
    } else {
        OneShotCmd osc;
        if (!osc.begin(ctx->vk, ctx->device, ctx->cmdPool, ctx->queue)) {
            destroyBuffer(ctx, &staging);
            if (err) *err = "无法开始一次性命令缓冲";
            return false;
        }
        ctx->vk.vkCmdCopyBuffer(osc.cmd(), staging.buffer, dst->buffer, 1, &region);
        ok = osc.end(ctx->vk);        /* 同步路径：内部已 submit + waitForFences */
    }
    destroyBuffer(ctx, &staging);     /* 到这里 GPU 已完成，安全释放 */
    if (!ok && err) *err = "staging 上传失败";
    return ok;
}

/* 释放外部上传模式下延迟积压的 staging 缓冲。
 * 宿主必须在命令缓冲「提交并等待完成」之后调用，否则仍会 use-after-free。 */
void flushPendingUploads(ContextImpl *ctx) {
    if (!ctx) return;
    for (Buffer &b : ctx->pendingUploads) {
        destroyBuffer(ctx, &b);
    }
    ctx->pendingUploads.clear();
}

/* 用指定颜色填充设备本地缓冲（经 staging），用于预置边框 */
bool fillBuffer(ContextImpl *ctx, Buffer *dst, uint32_t word, std::string *err) {
    std::vector<uint32_t> tmp(static_cast<size_t>(dst->size / 4), word);
    return uploadBuffer(ctx, dst, tmp.data(), dst->size, err);
}

/* ------------------------------------------------------------------ 管线 */

bool createPipelines(ContextImpl *ctx, std::string *err) {
    const VkFuncs &vk = ctx->vk;

    /* 描述符集布局：8 个 storage buffer */
    VkDescriptorSetLayoutBinding binds[kDescriptorBindingCount]{};
    for (uint32_t i = 0; i < kDescriptorBindingCount; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = kDescriptorBindingCount;
    dlci.pBindings = binds;
    if (vk.vkCreateDescriptorSetLayout(ctx->device, &dlci, nullptr, &ctx->dsl) != VK_SUCCESS) {
        if (err) *err = "vkCreateDescriptorSetLayout 失败";
        return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = kPushConstantSize;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &ctx->dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vk.vkCreatePipelineLayout(ctx->device, &plci, nullptr, &ctx->pll) != VK_SUCCESS) {
        if (err) *err = "vkCreatePipelineLayout 失败";
        return false;
    }

    struct PipeSpec {
        const uint32_t *code;
        uint32_t words;
        VkPipeline *out;
        const char *name;
    };
    const PipeSpec specs[] = {
        { nss_model::kSpirv_conv_rq,     nss_model::kSpirv_conv_rq_words,     &ctx->pipeConv,   "conv" },
        { nss_model::kSpirv_conv_rq2x2,  nss_model::kSpirv_conv_rq2x2_words,  &ctx->pipeConv2x2, "conv2x2" },
        { nss_model::kSpirv_resize2x,    nss_model::kSpirv_resize2x_words,    &ctx->pipeResize, "resize" },
        { nss_model::kSpirv_concat_copy, nss_model::kSpirv_concat_copy_words, &ctx->pipeConcat, "concat" },
        /* Tensor Core 路径的 conv 内核（int8 cooperative matrix MMA）。
         * 只在设备可用时编译进去；否则这条 SPIR-V 在部分驱动上会被拒。 */
        { ctx->useTc ? nss_model::kSpirv_conv_tc : nullptr,
          ctx->useTc ? nss_model::kSpirv_conv_tc_words : 0u,
          &ctx->pipeConvTc, "conv_tc" },
    };

    for (const PipeSpec &s : specs) {
        if (!s.code || s.words == 0u) {
            continue;   /* 该后端不启用（例如 Tensor Core 不可用） */
        }
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = static_cast<size_t>(s.words) * 4;
        smci.pCode = s.code;
        VkShaderModule mod = VK_NULL_HANDLE;
        if (vk.vkCreateShaderModule(ctx->device, &smci, nullptr, &mod) != VK_SUCCESS) {
            if (err) *err = std::string("创建 shader module 失败: ") + s.name;
            return false;
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = ctx->pll;
        VkResult r = vk.vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                                 &cpci, nullptr, s.out);
        vk.vkDestroyShaderModule(ctx->device, mod, nullptr);
        if (r != VK_SUCCESS) {
            /* requantize 内核用 int64（acc × multiplier 可达 56 位）。
             * 宿主建 VkDevice 时必须启用 VkPhysicalDeviceFeatures::shaderInt64，
             * 否则驱动会在这里拒绝整条管线，而报错只有一句 VkResult。 */
            if (err) {
                *err = std::string("创建计算管线失败: ") + s.name
                     + "（VkResult=" + std::to_string(static_cast<int>(r))
                     + "；若驱动拒绝，检查宿主是否启用了 shaderInt64）";
            }
            return false;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ 资源准备 */

bool createScratchAndConsts(ContextImpl *ctx, std::string *err) {
    const VkFuncs &vk = ctx->vk;
    const auto &bufs = ctx->model.buffers();
    const auto &disps = ctx->model.dispatches();

    /* 中间张量：每个缓冲 (h+2)*(w+2)*c 字节，预填 0x80（激活零点） */
    ctx->scratchBufs.assign(bufs.size(), Buffer{});
    for (size_t i = 0; i < bufs.size(); ++i) {
        const Shape &s = bufs[i];
        if (s.h == 0) continue;
        const uint64_t bytes = static_cast<uint64_t>(s.h + 2) * (s.w + 2) * s.c
                               + kTensorTailSlack;
        if (!createBuffer(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &ctx->scratchBufs[i], err)) {
            return false;
        }
        ctx->internalBytes += ctx->scratchBufs[i].size;
        if (!fillBuffer(ctx, &ctx->scratchBufs[i], 0x80808080u, err)) return false;
    }

    /* Tensor Core 路径的层计划与重排权重缓冲（按 dispatch 下标对齐） */
    if (ctx->useTc) {
        ctx->tcPlans.assign(disps.size(), TcLayerPlan{});
        ctx->tcWeightBufs.assign(disps.size(), Buffer{});
    }

    /* 每层常量：权重/bias+修正/multiplier/shift/LUT */
    ctx->layerConsts.assign(disps.size(), ContextImpl::LayerConst{});
    for (size_t i = 0; i < disps.size(); ++i) {
        const Dispatch &d = disps[i];
        if (d.kind != 0) continue;                     /* 仅 conv */
        ContextImpl::LayerConst &lc = ctx->layerConsts[i];

        if (!createBuffer(ctx, static_cast<uint64_t>(d.wWords) * 4,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &lc.w, err)) return false;
        if (!uploadBuffer(ctx, &lc.w, d.w, static_cast<uint64_t>(d.wWords) * 4, err)) return false;
        ctx->internalBytes += lc.w.size;

        if (!createBuffer(ctx, static_cast<uint64_t>(d.multCount) * 4,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &lc.bc, err)) return false;
        if (!uploadBuffer(ctx, &lc.bc, d.bc, static_cast<uint64_t>(d.multCount) * 4, err)) return false;
        ctx->internalBytes += lc.bc.size;

        if (!createBuffer(ctx, static_cast<uint64_t>(d.multCount) * 4,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &lc.mult, err)) return false;
        if (!uploadBuffer(ctx, &lc.mult, d.mult, static_cast<uint64_t>(d.multCount) * 4, err)) return false;
        ctx->internalBytes += lc.mult.size;

        if (!createBuffer(ctx, static_cast<uint64_t>(d.multCount) * 4,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &lc.shift, err)) return false;
        if (!uploadBuffer(ctx, &lc.shift, d.shift, static_cast<uint64_t>(d.multCount) * 4, err)) return false;
        ctx->internalBytes += lc.shift.size;

        if (!createBuffer(ctx, 256 * 4,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &lc.lut, err)) return false;
        if (!uploadBuffer(ctx, &lc.lut, d.lut, 256 * 4, err)) return false;
        ctx->internalBytes += lc.lut.size;

        /* Tensor Core 路径：把烘焙的 [oc][K/4] 重排成 MMA 的 [chunk][k][n]。
         * 只在这里做一次；DP4A 路径下这些缓冲为空、不占显存。
         * 任一层形状不被支持 ⇒ 整体回退 DP4A（管线已建但不会被用到）。
         *
         * ★ 逐层选路：MMA 的 A 矩阵暂存成本与输出通道数无关，只由
         *   (kh·kw·ceil(Cin/32)·32) 决定；DP4A 的成本却随 out_c4（=Cout/4）线性增长。
         *   于是 **输出通道少的层 MMA 反而更贵**（实测 cout=4 的末层慢 3 倍）。
         *   这里按 cout 阈值挑层，阈值可用 NSS_DP4A_TC_MIN_COUT 覆盖做 sweep。 */
        if (ctx->useTc && i < ctx->tcPlans.size()
            && static_cast<uint32_t>(d.outShape.c) >= ctx->tcMinCout) {
            std::vector<uint8_t> wTc;
            if (!buildTcLayerPlan(d, &ctx->tcPlans[i], &wTc)) {
                logMsg(ctx, 1, "有层的形状不支持 Tensor Core 路径，该层回退 DP4A");
            } else {
                Buffer &b = ctx->tcWeightBufs[i];
                if (!createBuffer(ctx, static_cast<uint64_t>(wTc.size()),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  false, &b, err)) return false;
                if (!uploadBuffer(ctx, &b, wTc.data(), static_cast<uint64_t>(wTc.size()), err)) return false;
                ctx->internalBytes += b.size;
            }
        }
    }

    /* 描述符池：每次 record 都要分配描述符集，所以要够大。
     * 一个上限：按最大可能帧数 × dispatch 数。这里按「单帧一次」预留，
     * 并在每次 record 后重置池，避免无限增长。 */
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = static_cast<uint32_t>(disps.size()) * kDescriptorBindingCount + 16;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = static_cast<uint32_t>(disps.size()) + 8;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (vk.vkCreateDescriptorPool(ctx->device, &dpci, nullptr, &ctx->descPool) != VK_SUCCESS) {
        if (err) *err = "vkCreateDescriptorPool 失败";
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ record */

/* ---- 诊断：把一次 recordFrame 的规模与潜在越界全部算出来并打印 ---- */
static void diagnoseFrame(ContextImpl *ctx, const NssDp4aDispatchInfo &info) {
    const auto &disps = ctx->model.dispatches();
    const auto &bufs = ctx->model.buffers();

    char line[512];

    /* 输入：宿主缓冲必须容得下 H*W*12 */
    const uint64_t needIn = static_cast<uint64_t>(info.height) * info.width * 12;
    const uint64_t haveIn = info.input.size ? info.input.size : UINT64_MAX;
    std::snprintf(line, sizeof(line),
                  "[NssDiag] 输入 H=%u W=%u 需要=%llu 宿主可用=%llu%s",
                  info.height, info.width,
                  static_cast<unsigned long long>(needIn),
                  static_cast<unsigned long long>(info.input.size),
                  (haveIn < needIn) ? "  <== 越界！" : "");
    logMsg(ctx, (haveIn < needIn) ? 2 : 1, line);

    /* 输出：KPN / temporal 需要多少 */
    const uint64_t kpnW = info.width / 4, kpnH = info.height / 4;
    const uint64_t needKpnHigh = kpnH * kpnW * 36;
    const uint64_t needKpnMid = kpnH * kpnW * 16;
    const uint64_t needTmp = static_cast<uint64_t>(info.height) * info.width * 4;
    std::snprintf(line, sizeof(line),
                  "[NssDiag] KPN 需要(high)=%llu (mid)=%llu 宿主可用=%llu；"
                  "temporal 需要=%llu 宿主可用=%llu",
                  static_cast<unsigned long long>(needKpnHigh),
                  static_cast<unsigned long long>(needKpnMid),
                  static_cast<unsigned long long>(info.outputKpn.size),
                  static_cast<unsigned long long>(needTmp),
                  static_cast<unsigned long long>(info.outputTemporal.size));
    logMsg(ctx, 1, line);

    /* 每个 dispatch 的 dispatch 尺寸 vs 输出缓冲实际容量 */
    for (size_t di = 0; di < disps.size(); ++di) {
        const Dispatch &d = disps[di];
        if (d.outBuf < 0 || static_cast<size_t>(d.outBuf) >= bufs.size()) {
            std::snprintf(line, sizeof(line),
                          "[NssDiag] dispatch[%zu] outBuf=%d 越界（bufs=%zu）", di, d.outBuf, bufs.size());
            logMsg(ctx, 2, line);
            continue;
        }
        const Shape &os = bufs[d.outBuf];
        const uint64_t capBytes = static_cast<uint64_t>(os.h + 2) * (os.w + 2) * os.c;
        /* 内核每个 (x,y) 线程写 gz 组 × 4 通道 = 4*gz 字节（int8），
         * 行内跨度 (w+2)*c，故实际需要 gx*(gy*(w+2)*c) 量级；这里只做粗校验：
         * 按「带边框张量」的完整容量比，内核写的是 [1..gy][1..gx] 区域。 */
        const uint64_t writeBytes = static_cast<uint64_t>(d.gx) * d.gy * 4 * d.gz;
        std::snprintf(line, sizeof(line),
                      "[NssDiag] dispatch[%zu] kind=%d 输入=%s 输出buf=%d(%ux%ux%u cap=%llu) "
                      "gx=%u gy=%u gz=%u 写=%llu",
                      di, d.kind, (d.inBuf < 0 ? "mirror" : "scratch"),
                      d.outBuf, os.w, os.h, os.c,
                      static_cast<unsigned long long>(capBytes),
                      d.gx, d.gy, d.gz,
                      static_cast<unsigned long long>(writeBytes));
        logMsg(ctx, 1, line);
    }

    /* 顺带核对每个 scratch 缓冲与所需尺寸 */
    for (size_t i = 0; i < bufs.size(); ++i) {
        if (bufs[i].h == 0) continue;
        const uint64_t need = static_cast<uint64_t>(bufs[i].h + 2) * (bufs[i].w + 2) * bufs[i].c;
        if (ctx->scratchBufs[i].size < need) {
            std::snprintf(line, sizeof(line),
                          "[NssDiag] scratch[%zu] %ux%ux%u 需要=%llu 实际=%llu  <== 不足！",
                          i, bufs[i].w, bufs[i].h, bufs[i].c,
                          static_cast<unsigned long long>(need),
                          static_cast<unsigned long long>(ctx->scratchBufs[i].size));
            logMsg(ctx, 2, line);
        }
    }
}

/* conv_rq.comp 里每线程负责的 oc4 组数（OCG），必须与着色器的 #define OCG 一致。
 * 这里用它把 conv 的 z 维（= out_c4）按 OCG 折叠：
 * 折叠后同一个 3x3xCin 邻域只装载一次，供 OCGx4 个输出通道复用。 */
static const uint32_t CONV_OCG = 4u;

/* conv_rq.comp 里每线程负责的输出像素数（TILE_X），必须与着色器的 #define TILE_X 一致。
 * 这里用它把 conv 的 x 维（= out_w）按 TILE_X 折叠。 */
static const uint32_t CONV_TILE_X = 1u;   /* 实测 TILE_X=2 因寄存器压力反而略慢，回退 */
/* 诊断：NSS_DP4A_DIAG_SKIP 位掩码（仅测量用，结果不再位精确）。
 *   1 = 跳过分派间的全局 barrier
 *   2 = 跳过输出回拷
 *   4 = 跳过 conv 分派
 *   8 = 跳过 resize/concat 分派
 * 4/8 用来把「卷积」与「重采样/拼接」的真实占比量出来 —— 注意 per-dispatch 的
 * 时间戳归属不可靠（TOP/BOTTOM 语义会互相污染），做结构决策必须用总量。 */
static bool diagSkip(uint32_t bit) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = std::getenv("NSS_DP4A_DIAG_SKIP");
        cached = e ? std::atoi(e) : 0;
    }
    return (static_cast<uint32_t>(cached) & bit) != 0u;
}

/* 诊断：NSS_DP4A_ONLY_LAYER=<n> 时，只有第 n 次 dispatch 用真实网格，
 * 其余全部退化 1x1x1 —— 得到「底线 + 该层真实耗时」，逐层跑一遍即可拿到
 * 可靠的逐层成本表（远比被串行污染的 TOP/BOTTOM 时间戳可信）。
 * 未设置或 <0 = 关闭。 */
static int diagOnlyLayer() {
    static int cached = -2;
    if (cached == -2) {
        const char *e = std::getenv("NSS_DP4A_ONLY_LAYER");
        cached = e ? std::atoi(e) : -1;
    }
    return cached;
}

bool recordFrame(ContextImpl *ctx, VkCommandBuffer cmd, const NssDp4aDispatchInfo &info,
                 std::string *err) {
    const VkFuncs &vk = ctx->vk;
    const auto &disps = ctx->model.dispatches();

    if (ctx->diagFrames < 3) {
        diagnoseFrame(ctx, info);
        ++ctx->diagFrames;
    }

    /* 输入：宿主的缓冲已经是 int8 NHWC [H][W][12]，但内核期望四周有 1 像素边框。
     * 两种做法：
     *   (a) 宿主提供一个带边框的缓冲（推荐，零拷贝）
     *   (b) 库内做一次带边框的拷贝（需要额外 dispatch）
     * 这里采用 (b) 的简化版：把宿主输入拷进库内的带边框缓冲的中部。
     * 边框已在创建时预填 0x80。 */
    VkBuffer srcInput = reinterpret_cast<VkBuffer>(info.input.buffer);

    /* 清空描述符池（上一帧的集合已随命令缓冲执行完毕） */
    vk.vkResetDescriptorPool(ctx->device, ctx->descPool, 0);

    /*
     * 输入处理的两条路（NSS_DP4A_INPUT_MIRROR=1 可切）：
     *
     *  (a) 默认：第一个 conv 直接绑定宿主紧致张量，越界 tap 由 kernel 用 z_a(0x80)
     *      填充代替（push[15] inBorded=0 分支）—— 省掉整份镜像拷贝。
     *      ★ 代价：TC 内核在 inBorded=0 时必须走 **staged 路径**（主机输入无边框，
     *        越界必须显式判定）—— 每个 chunk 要做 128 次带谓词的共享内存写 + 2 次
     *        barrier，而且读地址是散乱的（16 像素 × 8 word 槽里只有 in_c4 个有效）。
     *
     *  (b) 镜像：把宿主紧致输入拷进带边框的镜像缓冲（边框创建时已预填 0x80），
     *      第一层就走 direct 路径。拷贝用「一条命令 + 全行区域数组」，不是逐行命令。
     *
     * 逐层隔离实测（1280x664）显示第一层是最贵的单层，两条路谁赢由 A/B 决定。
     */
    static const bool mirrorEnv = (std::getenv("NSS_DP4A_INPUT_MIRROR") != nullptr);
    const bool useInputMirror = mirrorEnv && (ctx->inputMirror.buffer != VK_NULL_HANDLE);
    if (useInputMirror) {
        const uint32_t inH = ctx->model.height();
        const uint32_t inW = ctx->model.width();
        const uint32_t inC = 12u;
        const uint32_t srcStride = inW * inC;
        const uint32_t dstStride = (inW + 2u) * inC;
        std::vector<VkBufferCopy> regions(static_cast<size_t>(inH));
        for (uint32_t y = 0; y < inH; ++y) {
            regions[y].srcOffset = info.input.offset + static_cast<uint64_t>(y) * srcStride;
            regions[y].dstOffset = static_cast<uint64_t>(y + 1u) * dstStride + inC;
            regions[y].size = srcStride;
        }
        vk.vkCmdCopyBuffer(cmd, srcInput, ctx->inputMirror.buffer,
                           static_cast<uint32_t>(regions.size()), regions.data());
        VkMemoryBarrier mbIn{};
        mbIn.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mbIn.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mbIn.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                1, &mbIn, 0, nullptr, 0, nullptr);
    }

    /* Per-dispatch GPU timestamps: reset once, then TOP_OF_PIPE before and
     * BOTTOM_OF_PIPE after each dispatch. Read with nssDp4aGetDispatchTimes(). */
    /* 本帧哪些宿主输出已由 kernel 直接写入（用于跳过回拷）。 */
    bool hostOutWritten[2] = {false, false};

    /* NSS_DP4A_NO_TS=1 时不写 per-dispatch 时间戳 —— BOTTOM_OF_PIPE 语义上要等
     * 「所有后续工作」，实测会强制管线串行化，是纯诊断开销。 */
    static const bool noTs = (std::getenv("NSS_DP4A_NO_TS") != nullptr);
    const bool tsEnabled = !noTs
                           && (ctx->tsPool != VK_NULL_HANDLE && ctx->vk.vkCmdResetQueryPool &&
                               ctx->vk.vkCmdWriteTimestamp);
    const uint32_t tsCapacity = 256u;
    uint32_t tsPairs = 0;
    if (tsEnabled) {
        size_t want = disps.size() * 2u;
        uint32_t used = want > (tsCapacity - 1) ? (tsCapacity - 1) : static_cast<uint32_t>(want);
        tsPairs = used / 2u;
        if (used > 0) {
            ctx->vk.vkCmdResetQueryPool(cmd, ctx->tsPool, 0, used);
        }
        ctx->tsCount = used;
    }

    for (size_t di = 0; di < disps.size(); ++di) {
        const Dispatch &d = disps[di];

        /* 选管线（conv 有 DP4A / Tensor Core 两条实现，数值逐位一致）。
         * 逐层判定：只有真正建好了 MMA 权重的层才走 TC 内核。 */
        const bool tcLayer = ctx->useTc && di < ctx->tcPlans.size()
                             && ctx->tcPlans[di].nBlocks > 0u;
        /* 诊断：把整类分派的网格退化成 1×1×1（几乎零工作量，但分派仍发出、
         * 图结构不变）。★ 不要用 continue 直接跳过分派 —— 实测那样会让 GPU
         * 挂死（命令缓冲里少了内容，fence 永不触发）。 */
        const bool diagKillThis =
                (d.kind == 0 && diagSkip(4u)) || (d.kind != 0 && diagSkip(8u));
        /* 逐层隔离：只让第 N 次 dispatch 跑真实网格（见 diagOnlyLayer） */
        const int onlyL = diagOnlyLayer();
        const bool diagKillThis2 = (onlyL >= 0) ? (static_cast<int>(di) != onlyL)
                                                : diagKillThis;
        VkPipeline pipe = VK_NULL_HANDLE;
        /* 2×2 像素变体选路：非 TC 的小输出 conv（cout=4）且 stride=1。
         * ⚠ 2026-10-01：harness 显示该变体 op32 提速 31% 但 temporal 输出值
         * 有偏差（range [-127,118] vs 基线 [-127,126]）且整帧异常变慢 ——
         * bug 未定位，默认关闭，保留代码与管线（NSS_DP4A_CONV2X2=1 启用调试）。 */
        static const bool conv2x2Enable =
                (std::getenv("NSS_DP4A_CONV2X2") != nullptr);
        const bool conv2x2Layer = (d.kind == 0u && !tcLayer && conv2x2Enable
                                   && d.outShape.c == 4u
                                   && d.push[7] == 1u && d.push[8] == 1u
                                   && ctx->pipeConv2x2 != VK_NULL_HANDLE);
        switch (d.kind) {
            case 0: pipe = tcLayer ? ctx->pipeConvTc
                                   : (conv2x2Layer ? ctx->pipeConv2x2 : ctx->pipeConv);
                    break;
            case 1: pipe = ctx->pipeResize; break;
            case 2: pipe = ctx->pipeConcat; break;
            default: if (err) *err = "未知 dispatch 类型"; return false;
        }

        /* 分配描述符集 */
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = ctx->descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &ctx->dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (vk.vkAllocateDescriptorSets(ctx->device, &dsai, &ds) != VK_SUCCESS) {
            if (err) *err = "分配描述符集失败";
            return false;
        }

        /* 8 个绑定：0=输入 1=权重 2=biasCr 3=mult 4=shift 5=lut 6=输出 7=调试 */
        VkDescriptorBufferInfo infos[kDescriptorBindingCount]{};
        VkWriteDescriptorSet writes[kDescriptorBindingCount]{};

        auto setBinding = [&](uint32_t b, VkBuffer buf, uint64_t size) {
            infos[b].buffer = buf;
            infos[b].offset = 0;
            infos[b].range = size ? size : VK_WHOLE_SIZE;
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = ds;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        };

        /* 同上，但允许非零绑定偏移（宿主缓冲可能带 offset）。 */
        auto setBindingAt = [&](uint32_t b, VkBuffer buf, uint64_t size, uint64_t offset) {
            infos[b].buffer = buf;
            infos[b].offset = offset;
            infos[b].range = size ? size : VK_WHOLE_SIZE;
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = ds;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        };

        /* 输入缓冲：d.inBuf < 0 表示图输入。hostInput=true 时直接绑宿主紧致张量
         * （kernel 走 inBorded=0 分支）；镜像模式下改绑带边框的镜像缓冲。 */
        VkBuffer inBuf = VK_NULL_HANDLE;
        uint64_t inSize = 0;
        uint64_t inOff = 0;
        const bool hostInput = (d.inBuf < 0) && !useInputMirror;
        if (d.inBuf < 0) {
            if (useInputMirror) {
                inBuf = ctx->inputMirror.buffer;
                inSize = ctx->inputMirror.size;
                inOff = 0;
            } else {
                inBuf = srcInput;
                inSize = info.input.size;
                inOff = info.input.offset;
            }
        } else {
            inBuf = ctx->scratchBufs[d.inBuf].buffer;
            inSize = ctx->scratchBufs[d.inBuf].size;
        }
        setBindingAt(0, inBuf, inSize, inOff);

        /* 输出缓冲：若本层直接产出宿主输出（KPN / temporal），则绑宿主缓冲，省掉回拷。 */
        VkBuffer outBuf = ctx->scratchBufs[d.outBuf].buffer;
        uint64_t outSize = ctx->scratchBufs[d.outBuf].size;
        uint64_t outOff = 0;
        int hostOutIdx = -1;
        const auto &outBufsIdx = ctx->model.outputBufs();
        if (d.kind == 0) {
            for (size_t oi = 0; oi < outBufsIdx.size() && oi < 2; ++oi) {
                if (d.outBuf == static_cast<int>(outBufsIdx[oi])) {
                    hostOutIdx = static_cast<int>(oi);
                    break;
                }
            }
        }
        if (hostOutIdx == 0) {
            outBuf = reinterpret_cast<VkBuffer>(info.outputKpn.buffer);
            outSize = info.outputKpn.size;
            outOff = info.outputKpn.offset;
            hostOutWritten[0] = true;
        } else if (hostOutIdx == 1) {
            outBuf = reinterpret_cast<VkBuffer>(info.outputTemporal.buffer);
            outSize = info.outputTemporal.size;
            outOff = info.outputTemporal.offset;
            hostOutWritten[1] = true;
        }
        setBindingAt(6, outBuf, outSize, outOff);

        if (d.kind == 0) {
            const ContextImpl::LayerConst &lc = ctx->layerConsts[di];
            if (tcLayer) {
                /* TC 路径：binding 1 换成重排后的 [chunk][k][n] 权重。
                 * bc/mult/shift/lut 与 DP4A 完全共用（重量化逻辑一致）。 */
                const Buffer &wt = ctx->tcWeightBufs[di];
                setBinding(1, wt.buffer, wt.size);
            } else {
                setBinding(1, lc.w.buffer, lc.w.size);
            }
            setBinding(2, lc.bc.buffer, lc.bc.size);
            setBinding(3, lc.mult.buffer, lc.mult.size);
            setBinding(4, lc.shift.buffer, lc.shift.size);
            setBinding(5, lc.lut.buffer, lc.lut.size);
        }
        /* 其余绑定留空（null）—— 内核在非 conv 路径不会访问它们，
         * 但描述符必须有效，所以统一指向输入缓冲。 */
        for (uint32_t b = 0; b < kDescriptorBindingCount; ++b) {
            if (writes[b].dstSet == VK_NULL_HANDLE) {
                setBinding(b, inBuf, inSize);
            }
        }
        /* binding 7（调试）恒指向输出缓冲即可（内核只在 dbg!=0 时写） */
        setBinding(7, ctx->scratchBufs[d.outBuf].buffer, ctx->scratchBufs[d.outBuf].size);

        vk.vkUpdateDescriptorSets(ctx->device, kDescriptorBindingCount, writes, 0, nullptr);

        /* 绑定 + 推送常量 + dispatch */
        vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pll,
                                   0, 1, &ds, 0, nullptr);
        /* conv 按本层的实际绑定覆盖 push[14..16]：
         *   [14]=输入内部高  [15]=inBorded  [16]=outBorded */
        uint32_t pushLocal[20];
        const uint32_t *pushPtr = d.push;
        uint32_t pushSize = d.pushSize;
        if (d.kind == 0) {
            std::memcpy(pushLocal, d.push, sizeof(uint32_t) * 17);
            pushLocal[14] = hostInput
                    ? info.height
                    : ((d.inBuf < 0) ? ctx->model.height() : ctx->model.buffers()[d.inBuf].h);
            pushLocal[15] = hostInput ? 0u : 1u;
            pushLocal[16] = (hostOutIdx >= 0) ? 0u : 1u;
            pushPtr = pushLocal;
            pushSize = 17u;
            if (tcLayer) {
                /* TC 内核额外要 3 个：每块通道数、B 矩阵行跨距、累加器块数 */
                const TcLayerPlan &tp = ctx->tcPlans[di];
                pushLocal[17] = tp.cdxCount;
                pushLocal[18] = tp.nPad;
                pushLocal[19] = tp.nBlocks;
                pushSize = 20u;
            }
        }
        vk.vkCmdPushConstants(cmd, ctx->pll, VK_SHADER_STAGE_COMPUTE_BIT,
                              0, pushSize * 4, pushPtr);
        if (tsEnabled && di < tsPairs) {
            ctx->vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                        ctx->tsPool, static_cast<uint32_t>(di * 2u));
        }
        /* conv：z 维按 CONV_OCG 折叠（见上）；resize/concat 保持原样。
         * TC 路径另有一套网格：每个 workgroup = 1 个 subgroup，负责一行里 16 个像素。 */
        uint32_t dx, dy, dz;
        if (diagKillThis2) {
            dx = dy = dz = 1u;   /* 诊断：退化网格 */
        } else if (d.kind == 0u && tcLayer) {
            const TcLayerPlan &tp = ctx->tcPlans[di];
            dx = tp.gridX;
            dy = tp.gridY;
            /* 通道块默认全部由同一个 workgroup 算（dz=1，多累加器）；
             * NSS_DP4A_TC_NBSPLIT=1 时把块切到 gridZ 上（每个 workgroup 1 个块、
             * 1 个累加器、1 轮导出）—— 与 conv_tc.comp 的 PROBE=13 配套。 */
            static const bool nbSplit = (std::getenv("NSS_DP4A_TC_NBSPLIT") != nullptr);
            dz = nbSplit ? tp.nBlocks : 1u;
        } else if (d.kind == 0u && conv2x2Layer) {
            /* 2×2 像素变体：每线程负责 2×2 输出像素 */
            dx = (d.gx + 15u) / 16u;
            dy = (d.gy + 15u) / 16u;
            dz = (d.gz + CONV_OCG - 1u) / CONV_OCG;
        } else {
            const uint32_t gz = (d.kind == 0u)
                    ? ((d.gz + CONV_OCG - 1u) / CONV_OCG)
                    : d.gz;
            const uint32_t gx = (d.kind == 0u)
                    ? ((d.gx + CONV_TILE_X - 1u) / CONV_TILE_X)
                    : d.gx;
            dx = (gx + 7) / 8;
            dy = (d.gy + 7) / 8;
            dz = gz;
        }
        vk.vkCmdDispatch(cmd, dx, dy, dz);
        if (tsEnabled && di < tsPairs) {
            ctx->vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                        ctx->tsPool, static_cast<uint32_t>(di * 2u + 1u));
        }

        /* 全局 barrier：下一层要读这一层的输出 */
        if (diagSkip(1u)) {
            continue;   /* 诊断：跳过 barrier（只测时间） */
        }
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                1, &mb, 0, nullptr, 0, nullptr);
    }

    /* 把两个输出从内部缓冲拷到宿主的缓冲（含去边框） */
    const auto &outBufs = ctx->model.outputBufs();
    const VkBuffer dsts[2] = {
        reinterpret_cast<VkBuffer>(info.outputKpn.buffer),
        reinterpret_cast<VkBuffer>(info.outputTemporal.buffer),
    };
    const uint64_t dstOffsets[2] = { info.outputKpn.offset, info.outputTemporal.offset };

    for (size_t k = 0; k < outBufs.size() && k < 2; ++k) {
        if (hostOutWritten[k]) {
            continue;      /* kernel 已直接写入宿主缓冲，无需回拷 */
        }
        const Shape &s = ctx->model.buffers()[outBufs[k]];
        const uint32_t srcStride = (s.w + 2) * s.c;
        const uint32_t rowBytes = s.w * s.c;
        if (s.h == 0 || rowBytes == 0) {
            continue;
        }
        /*
         * ★ 性能修复（2026-09-30）：原来这里是「每行一条 vkCmdCopyBuffer」——
         * 1280x664 的两路输出合计约 1300 条微小拷贝命令，且源偏移带 +s.c（仅 4B 对齐），
         * GPU 前端命令处理开销达数毫秒；游戏内表现为下一个消费该缓冲的 pass 等待 ~5.7ms
         * （实测 NSS 分段：时序转换 5.71ms，而其它所有 pass 合计 < 0.5ms）。
         * 改为「一条命令 + 全部行区域数组」，命令数从 ~1300 降到 2。
         */
        std::vector<VkBufferCopy> regions(static_cast<size_t>(s.h));
        for (uint32_t y = 0; y < s.h; ++y) {
            regions[y].srcOffset = static_cast<uint64_t>(y + 1) * srcStride + s.c;
            regions[y].dstOffset = dstOffsets[k] + static_cast<uint64_t>(y) * rowBytes;
            regions[y].size = rowBytes;
        }
        vk.vkCmdCopyBuffer(cmd, ctx->scratchBufs[outBufs[k]].buffer, dsts[k],
                           static_cast<uint32_t>(regions.size()), regions.data());
    }

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            1, &mb, 0, nullptr, 0, nullptr);
    return true;
}

}  /* namespace */

/* ------------------------------------------------------------------ C API */

struct NssDp4aContext {
    ContextImpl impl;
};

extern "C" {

NSS_DP4A_API uint32_t NSS_DP4A_CALL nssDp4aGetVersion(void) {
    return NSS_DP4A_VERSION;
}

NSS_DP4A_API const char * NSS_DP4A_CALL nssDp4aGetResultString(NssDp4aResult result) {
    switch (result) {
        case NSS_DP4A_OK: return "OK";
        case NSS_DP4A_ERROR_GENERIC: return "generic error";
        case NSS_DP4A_ERROR_UNSUPPORTED: return "unsupported device";
        case NSS_DP4A_ERROR_INVALID_ARGUMENT: return "invalid argument";
        case NSS_DP4A_ERROR_OUT_OF_MEMORY: return "out of memory";
        case NSS_DP4A_ERROR_VULKAN_FAILED: return "vulkan call failed";
        case NSS_DP4A_ERROR_NOT_READY: return "context not ready";
        case NSS_DP4A_ERROR_MODEL_MISMATCH: return "model mismatch";
        default: return "unknown";
    }
}

/*
 * 解析实例级函数指针并做一次完整能力探测。
 *
 * ★ 抽成公共实现是刻意的：能力查询有**两个**导出（一个填 NssDp4aDeviceCaps，
 *   一个填 NssDp4aCooperativeMatrixCaps），两者的加载器/函数解析逻辑必须一致 ——
 *   曾经因为只给其中一个补了 vkEnumerateDeviceExtensionProperties 而出现
 *   「独立查询说没有 cooperative matrix、上下文里却说有」的自相矛盾。
 * loader 在函数返回前 unload，调用方拿到的 DeviceCaps 是纯值拷贝。
 */
static bool probeDeviceCaps(uint64_t instance, uint64_t physicalDevice, uint32_t apiVersion,
                            void *vkGetInstanceProcAddr, DeviceCaps *out, std::string *err) {
    VkLoader loader;
    PFN_vkGetInstanceProcAddr gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vkGetInstanceProcAddr);
    if (!gipa) {
        if (!loader.loadLibrary()) return false;
        gipa = loader.getInstanceProcAddr;
    }
    /* 宿主没给、自行加载又失败的场景：不能拿空指针去 call */
    if (!gipa || !instance) {
        loader.unload();
        return false;
    }

    /* 探测只需要实例级函数 */
    VkFuncs vk{};
    vk.vkGetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        gipa(reinterpret_cast<VkInstance>(instance), "vkGetPhysicalDeviceProperties"));
    vk.vkGetPhysicalDeviceProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        gipa(reinterpret_cast<VkInstance>(instance), "vkGetPhysicalDeviceProperties2"));
    vk.vkGetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        gipa(reinterpret_cast<VkInstance>(instance), "vkGetPhysicalDeviceFeatures2"));
    /* Tensor Core 探测需要这两个（枚举设备扩展 + 查 cooperative matrix 形状） */
    vk.vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(reinterpret_cast<VkInstance>(instance), "vkEnumerateDeviceExtensionProperties"));
    vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            gipa(reinterpret_cast<VkInstance>(instance),
                 "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!vk.vkGetPhysicalDeviceProperties2 || !vk.vkGetPhysicalDeviceFeatures2) {
        loader.unload();
        return false;
    }

    const bool ok = queryDeviceCaps(vk, reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                    apiVersion, out, err);
    loader.unload();
    return ok;
}

NSS_DP4A_API NssDp4aResult NSS_DP4A_CALL
nssDp4aQueryDeviceCaps(uint64_t instance, uint64_t physicalDevice, uint32_t apiVersion,
                       void *vkGetInstanceProcAddr, NssDp4aDeviceCaps *outCaps) {
    if (!outCaps || !physicalDevice) return NSS_DP4A_ERROR_INVALID_ARGUMENT;

    DeviceCaps caps{};
    std::string err;
    const bool ok = probeDeviceCaps(instance, physicalDevice, apiVersion,
                                    vkGetInstanceProcAddr, &caps, &err);

    /* ★ 只写老字段。这个结构是**公共 ABI**：宿主（SRNativeNSS）会把实例建在栈上，
     *   任何布局变更都会让旧版宿主被越界写坏。Tensor Core 的信息走下面独立的
     *   查询函数，不塞进这个结构。 */
    outCaps->integerDotProduct = caps.integerDotProduct ? 1u : 0u;
    outCaps->dotProduct4x8BitPackedSigned = caps.dotProduct4x8BitPackedSigned ? 1u : 0u;
    outCaps->shaderInt64 = caps.shaderInt64 ? 1u : 0u;
    outCaps->maxComputeWorkGroupInvocations = caps.maxComputeWorkGroupInvocations;
    outCaps->maxComputeSharedMemorySize = caps.maxComputeSharedMemorySize;
    outCaps->maxStorageBufferRange = caps.maxStorageBufferRange;
    std::memcpy(outCaps->deviceName, caps.deviceName, sizeof(outCaps->deviceName));

    return ok ? NSS_DP4A_OK : NSS_DP4A_ERROR_UNSUPPORTED;
}

/* Tensor Core 能力查询（独立结构与导出，见头文件里对 ABI 的说明）。 */
NSS_DP4A_API NssDp4aResult NSS_DP4A_CALL
nssDp4aQueryCooperativeMatrix(uint64_t instance, uint64_t physicalDevice, uint32_t apiVersion,
                              void *vkGetInstanceProcAddr, NssDp4aCooperativeMatrixCaps *outCaps) {
    if (!outCaps || !physicalDevice) return NSS_DP4A_ERROR_INVALID_ARGUMENT;
    std::memset(outCaps, 0, sizeof(*outCaps));

    DeviceCaps caps{};
    std::string err;
    probeDeviceCaps(instance, physicalDevice, apiVersion, vkGetInstanceProcAddr, &caps, &err);

    outCaps->extensionEnabled = caps.cooperativeMatrix ? 1u : 0u;
    outCaps->featureEnabled = caps.cooperativeMatrix ? 1u : 0u;
    outCaps->int8Usable = caps.cmUsable ? 1u : 0u;
    outCaps->M = caps.cmM;
    outCaps->N = caps.cmN;
    outCaps->K = caps.cmK;
    return NSS_DP4A_OK;
}

/* 查询本上下文实际选用的推理后端（0=DP4A, 1=TENSOR_CORE）。诊断用。 */
NSS_DP4A_API uint32_t NSS_DP4A_CALL
nssDp4aGetBackend(const NssDp4aContext *context) {
    if (!context) return 0u;
    return context->impl.useTc ? 1u : 0u;
}

NSS_DP4A_API NssDp4aResult NSS_DP4A_CALL
nssDp4aCreateContext(const NssDp4aCreateInfo *createInfo, NssDp4aContext **outContext) {
    if (!createInfo || !outContext) return NSS_DP4A_ERROR_INVALID_ARGUMENT;
    if (!createInfo->device || !createInfo->physicalDevice) return NSS_DP4A_ERROR_INVALID_ARGUMENT;
    *outContext = nullptr;

    NssDp4aContext *ctx = new (std::nothrow) NssDp4aContext();
    if (!ctx) return NSS_DP4A_ERROR_OUT_OF_MEMORY;
    ContextImpl &I = ctx->impl;
    I.ci = *createInfo;

    /* 公开 API 用 uint64_t 传句柄，这里转成类型化形式 */
    I.instance = reinterpret_cast<VkInstance>(createInfo->instance);
    I.physDev = reinterpret_cast<VkPhysicalDevice>(createInfo->physicalDevice);
    I.device = reinterpret_cast<VkDevice>(createInfo->device);
    I.queue = reinterpret_cast<VkQueue>(createInfo->queue);
    I.externalUploadCmd = reinterpret_cast<VkCommandBuffer>(createInfo->uploadCommandBuffer);

    std::string err;

    /* 1) 加载 Vulkan
     * 宿主给的入口必须写回 loader.hostGipa —— resolveFuncs 读的是 loader，
     * 而 loader.getInstanceProcAddr 只在自行 LoadLibrary 成功时才有值。
     * 之前的写法把宿主值赋给一个用完即弃的局部变量，导致宿主传了值也拿不到。 */
    I.loader.hostGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        createInfo->vkGetInstanceProcAddr);
    if (!I.loader.hostGipa) {
        /* 宿主没给（例如 Java 侧反射取值失败）：自行加载 vulkan-1.dll 兜底 */
        if (!I.loader.loadLibrary()) {
            logMsg(&I, 2, "宿主未提供 vkGetInstanceProcAddr，且无法加载 vulkan-1.dll");
            delete ctx;
            return NSS_DP4A_ERROR_VULKAN_FAILED;
        }
        logMsg(&I, 1, "提示: 改用自行加载的 vulkan-1.dll 提供 vkGetInstanceProcAddr");
    }
    if (!resolveFuncs(I.loader, I.instance, I.device, &I.vk, &err)) {
        /* 这条路径原来静默失败，排查时只能看到 "vulkan call failed" */
        logMsg(&I, 2, err.empty() ? "解析 Vulkan 函数指针失败" : err.c_str());
        I.loader.unload();
        delete ctx;
        return NSS_DP4A_ERROR_VULKAN_FAILED;
    }

    /* 2) 能力检查 */
    if (!queryDeviceCaps(I.vk, I.physDev,
                         createInfo->apiVersion, &I.caps, &err)) {
        /* 同理：设备缺特性时要说明缺什么，否则宿主只会看到 ERROR。 */
        logMsg(&I, 2, err.empty() ? "设备能力检查未通过" : err.c_str());
        I.loader.unload();
        delete ctx;
        return NSS_DP4A_ERROR_UNSUPPORTED;
    }
    if (!I.caps.dotProduct4x8BitPackedSigned) {
        /* 有 DP4A 特性位但无硬件加速 —— 仍可运行（驱动会展开），但提示一下 */
        logMsg(&I, 1, "警告: DP4A 无硬件加速，性能会下降");
    }

    /*
     * 推理后端选择（单 DLL 内双路径）。
     *
     * 默认策略：设备能用 int8 cooperative matrix 就走 TENSOR_CORE，否则 DP4A。
     * 两条路径共用激活布局、RESCALE 参数、缓冲拓扑与 dispatch 表，只有 conv 内核
     * 与权重布局不同，且**数值逐位一致**（int32 累加精确、求和顺序无关）。
     *
     * 环境变量 NSS_DP4A_BACKEND=dp4a|tensorcore 可强制覆盖，用于 A/B 与回归。
     * 注意：扩展必须由**宿主**在 vkCreateDevice 时启用，DLL 无法事后追加 ——
     * caps.cmUsable 已经把这一点算进去了（它读的是设备实际已启用的状态）。
     */
    {
        const char *forced = std::getenv("NSS_DP4A_BACKEND");
        bool wantTc = I.caps.cmUsable;
        if (forced && *forced) {
            if (std::strcmp(forced, "dp4a") == 0) {
                wantTc = false;
            } else if (std::strcmp(forced, "tensorcore") == 0) {
                wantTc = I.caps.cmUsable;
            }
        }
        I.useTc = wantTc;
        /* 逐层选路的输出通道阈值（见 createScratchAndConsts 的说明）。
         * 0 = 所有层都用 MMA；999 = 全部回 DP4A。 */
        if (const char *mc = std::getenv("NSS_DP4A_TC_MIN_COUT")) {
            const int v = std::atoi(mc);
            if (v >= 0 && v <= 4096) I.tcMinCout = static_cast<uint32_t>(v);
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "推理后端: %s（cooperative matrix %s，int8 形状 %ux%ux%u%s）",
                      I.useTc ? "TENSOR_CORE(MMA)" : "DP4A",
                      I.caps.cmUsable ? "可用" : "不可用",
                      I.caps.cmM, I.caps.cmN, I.caps.cmK,
                      (forced && *forced) ? "，env 强制" : "");
        logMsg(&I, 0, buf);
    }

    /* 3) 命令池 —— 只在「自提交」模式下需要。
     *
     * uploadBuffer() 有两条路：宿主给了 uploadCommandBuffer 就把拷贝录进去、由宿主统一
     * 提交（此时 queue 与 queueFamilyIndex 仅作占位，本头文件也写明「可为 0」）；否则才
     * 自建一次性命令缓冲、用 queue 提交并等待。
     *
     * 原来无条件建池是错的，而且是硬错：vkCreateCommandPool 要求 queueFamilyIndex 必须是
     * VkDevice 创建时**请求过**的队列族，而 ffx-api 的 ffxCreateBackendVKDesc 根本不带
     * queue —— FFX 的整个 Vulkan 后端都是 queue-free 的，nssDp4aRecord 也只录制不提交。
     * 猜一个 family 不是「可能不生效」，是未定义行为。所以外部上传模式下不建池、不碰 queue。
     */
    if (I.externalUploadCmd == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = createInfo->queueFamilyIndex;
        if (I.vk.vkCreateCommandPool(reinterpret_cast<VkDevice>(createInfo->device),
                                     &cpci, nullptr, &I.cmdPool) != VK_SUCCESS) {
            I.loader.unload();
            delete ctx;
            return NSS_DP4A_ERROR_VULKAN_FAILED;
        }
    }

    /* 4) 管线 */
    if (!createPipelines(&I, &err)) {
        logMsg(&I, 2, err.c_str());
        nssDp4aDestroyContext(ctx);
        return NSS_DP4A_ERROR_VULKAN_FAILED;
    }

    /* 5) 模型：按 createInfo 指定的尺寸装载 */
    if (!I.model.load(createInfo->quality == NSS_DP4A_QUALITY_HIGH ? 0 : 1,
                      createInfo->width, createInfo->height, &err)) {
        logMsg(&I, 2, err.c_str());
        nssDp4aDestroyContext(ctx);
        return NSS_DP4A_ERROR_MODEL_MISMATCH;
    }

    /* 6) 中间张量与常量 */
    if (!createScratchAndConsts(&I, &err)) {
        logMsg(&I, 2, err.c_str());
        nssDp4aDestroyContext(ctx);
        return NSS_DP4A_ERROR_OUT_OF_MEMORY;
    }

    /* 7) 输入镜像缓冲（带边框） */
    {
        const Shape inShape = { I.model.height(), I.model.width(), 12 };
        const uint64_t bytes = static_cast<uint64_t>(inShape.h + 2) * (inShape.w + 2) * inShape.c;
        if (!createBuffer(&I, bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false, &I.inputMirror, &err)) {
            nssDp4aDestroyContext(ctx);
            return NSS_DP4A_ERROR_OUT_OF_MEMORY;
        }
        I.internalBytes += I.inputMirror.size;
        if (!fillBuffer(&I, &I.inputMirror, 0x80808080u, &err)) {
            nssDp4aDestroyContext(ctx);
            return NSS_DP4A_ERROR_VULKAN_FAILED;
        }
    }

    /* Diagnostic query pool for per-dispatch GPU timestamps. */
    {
        VkPhysicalDeviceProperties props{};
        if (I.vk.vkGetPhysicalDeviceProperties) {
            I.vk.vkGetPhysicalDeviceProperties(I.physDev, &props);
            if (props.limits.timestampPeriod > 0.0f) {
                I.tsPeriodNs = static_cast<double>(props.limits.timestampPeriod);
            }
        }
        VkQueryPoolCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 256u;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (I.vk.vkCreateQueryPool &&
            I.vk.vkCreateQueryPool(I.device, &qci, nullptr, &pool) == VK_SUCCESS) {
            I.tsPool = pool;
        }
    }

    I.ready = true;
    *outContext = ctx;
    return NSS_DP4A_OK;
}

NSS_DP4A_API void NSS_DP4A_CALL nssDp4aFlushPendingUploads(NssDp4aContext *context) {
    if (!context) return;
    flushPendingUploads(&context->impl);
}

NSS_DP4A_API void NSS_DP4A_CALL nssDp4aDestroyContext(NssDp4aContext *context) {
    if (!context) return;
    ContextImpl &I = context->impl;

    if (I.device) {
        I.vk.vkDeviceWaitIdle(reinterpret_cast<VkDevice>(I.device));

        /* 先清延迟队列：外部上传模式下 staging 可能还没释放，
         * 此时已 vkDeviceWaitIdle，可以安全销毁。 */
        flushPendingUploads(&I);

        for (auto &b : I.scratchBufs) destroyBuffer(&I, &b);
        destroyBuffer(&I, &I.inputMirror);
        for (auto &lc : I.layerConsts) {
            destroyBuffer(&I, &lc.w);
            destroyBuffer(&I, &lc.bc);
            destroyBuffer(&I, &lc.mult);
            destroyBuffer(&I, &lc.shift);
            destroyBuffer(&I, &lc.lut);
        }
        for (auto &b : I.tcWeightBufs) destroyBuffer(&I, &b);
        if (I.pipeConvTc) {
            I.vk.vkDestroyPipeline(I.device, I.pipeConvTc, nullptr);
            I.pipeConvTc = VK_NULL_HANDLE;
        }

        if (I.descPool) I.vk.vkDestroyDescriptorPool(I.device, I.descPool, nullptr);
        if (I.pipeConv) I.vk.vkDestroyPipeline(I.device, I.pipeConv, nullptr);
        if (I.pipeConv2x2) I.vk.vkDestroyPipeline(I.device, I.pipeConv2x2, nullptr);
        if (I.pipeResize) I.vk.vkDestroyPipeline(I.device, I.pipeResize, nullptr);
        if (I.pipeConcat) I.vk.vkDestroyPipeline(I.device, I.pipeConcat, nullptr);
        if (I.pll) I.vk.vkDestroyPipelineLayout(I.device, I.pll, nullptr);
        if (I.dsl) I.vk.vkDestroyDescriptorSetLayout(I.device, I.dsl, nullptr);
        if (I.tsPool) I.vk.vkDestroyQueryPool(I.device, I.tsPool, nullptr);
        if (I.cmdPool) I.vk.vkDestroyCommandPool(I.device, I.cmdPool, nullptr);
    }
    I.loader.unload();
    delete context;
}

NSS_DP4A_API NssDp4aResult NSS_DP4A_CALL
nssDp4aRecord(NssDp4aContext *context, uint64_t commandBuffer,
              const NssDp4aDispatchInfo *dispatchInfo) {
    if (!context || !commandBuffer || !dispatchInfo) return NSS_DP4A_ERROR_INVALID_ARGUMENT;
    ContextImpl &I = context->impl;
    if (!I.ready) return NSS_DP4A_ERROR_NOT_READY;

    if (dispatchInfo->width != I.model.width() || dispatchInfo->height != I.model.height()) {
        return NSS_DP4A_ERROR_MODEL_MISMATCH;
    }
    if (!dispatchInfo->input.buffer || !dispatchInfo->outputKpn.buffer ||
        !dispatchInfo->outputTemporal.buffer) {
        return NSS_DP4A_ERROR_INVALID_ARGUMENT;
    }

    std::string err;
    if (!recordFrame(&I, reinterpret_cast<VkCommandBuffer>(commandBuffer),
                     *dispatchInfo, &err)) {
        logMsg(&I, 2, err.c_str());
        return NSS_DP4A_ERROR_VULKAN_FAILED;
    }
    return NSS_DP4A_OK;
}

/* Reads the per-dispatch GPU durations recorded by the last nssDp4aRecord().
 * Must be called after the work completed. Writes at most `capacity` nanosecond
 * durations into outNanos and returns how many were written (0 = unavailable). */
NSS_DP4A_API uint32_t NSS_DP4A_CALL
nssDp4aGetDispatchTimes(NssDp4aContext *context, uint64_t *outNanos, uint32_t capacity) {
    if (!context || !outNanos || capacity == 0) return 0;
    ContextImpl &I = context->impl;
    if (!I.tsPool || I.tsCount < 2 || !I.vk.vkGetQueryPoolResults) return 0;
    uint32_t pairs = I.tsCount / 2u;
    uint32_t n = pairs < capacity ? pairs : capacity;
    if (n == 0) return 0;
    uint32_t count = n * 2u;
    std::vector<uint64_t> raw(static_cast<size_t>(count), 0ull);
    VkResult r = I.vk.vkGetQueryPoolResults(
            I.device, I.tsPool, 0, count,
            static_cast<size_t>(count) * sizeof(uint64_t), raw.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) return 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t a = raw[i * 2u];
        uint64_t b = raw[i * 2u + 1u];
        outNanos[i] = (b > a) ? static_cast<uint64_t>((b - a) * I.tsPeriodNs) : 0ull;
    }
    return n;
}

NSS_DP4A_API uint64_t NSS_DP4A_CALL nssDp4aGetScratchSize(const NssDp4aContext *context) {
    if (!context) return 0;
    return context->impl.model.scratchBytes();
}

NSS_DP4A_API uint64_t NSS_DP4A_CALL nssDp4aGetInternalMemoryUsage(const NssDp4aContext *context) {
    if (!context) return 0;
    return context->impl.internalBytes;
}

}  /* extern "C" */
