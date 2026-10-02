# nss —— Arm NSS 神经超分的桌面 GPU 推理后端（nss_dp4a.dll）

在 NVIDIA / AMD / Intel 桌面显卡上执行 Arm Neural Super Sampling（NSS）int8 神经网络，
**不需要 Vulkan ML 仿真层**，输出与 **Arm 官方实现位精确一致**。

两个 compute 后端，创建上下文时**逐层自动选路**：

| 后端 | 内核 | 依赖 | 选路条件 |
|---|---|---|---|
| **Tensor Core**（默认） | `conv_tc.comp`（cooperative matrix MMA） | `VK_KHR_cooperative_matrix` | 仅 conv 层，`cout ≥ 16`（`NSS_DP4A_TC_MIN_COUT` 可覆盖） |
| DP4A / OpSDot | `conv_rq.comp` | `shaderIntegerDotProduct` | 小层回退；不支持 coop-mat 的卡整体走此路 |

选路依据：MMA 的 A 矩阵暂存成本与输出通道数无关，**通道少的层 MMA 反而亏**
（实测 cout=4 的输出头层慢 3 倍）。当前两份模型（HIGH / MID_LOW）均 13/14 层走
Tensor Core，仅 oc=4 的 temporal/输出头回退 DP4A。

## 仓库关系

```
superresolution @ NSS_NEW 分支      Minecraft mod 集成（内嵌编译好的 nss_dp4a.dll）
        │ 使用
        ▼
nss（本仓库）                        DLL 源码 + 烘焙工具链(tools/) + 黄金参考
        │ SPIR-V 内核来自
        ▼
nss_kernel                          GPU 内核源(conv_rq/conv_tc/…) + 内核级验证台
```

模型权重 `.vgf` 不随仓库分发（Arm AI Model Community License），需自行获取，
见文末「许可与模型来源」。

## 形态：纯推理 DLL

**宿主拥有 VkDevice / VkQueue / VkCommandBuffer**，本库只在宿主的设备上建管线、
往宿主的命令缓冲里录 dispatch。输入输出都是宿主已有的 `VkBuffer`，**零拷贝**。

```
宿主应用（游戏 / Python 测试宿主）        nss_dp4a.dll
  VkDevice / VkQueue          ──────►   建管线（复用宿主设备）
  VkBuffer（GPU 原地）         ◄─────►   录 dispatch 进宿主的命令缓冲
```

这样设计的原因有两条，都很硬：

1. **输入是 GPU 纹理**。前处理 shader 的输出在显存里。若 DLL 自建设备，就得先把纹理读回
   CPU 再上传——而网络本身只要几毫秒，这个往返会吃掉大部分收益。
2. **替换点在宿主的命令缓冲里**。SDK 的 `ffx_vk.cpp` 中 `vkCmdDispatchDataGraphARM`
   与前后处理 shader 在同一个 command buffer 里，共享 barrier 与同步。
   接口必须能接受宿主的 device/queue/cmd，否则替换不了。

---

## 快速开始

依赖：Python 3.12+（仅 `numpy`）、MSVC x64 + Windows SDK（编译，见「构建环境」）、
glslangValidator（仅改内核后重烘 SPIR-V 时需要）。

```bash
# 0) 约定：把 nss_kernel 仓库 clone 到与本仓库同级目录
git clone https://github.com/eastear23333/nss_kernel ../nss_kernel

# 1) 烘焙模型（VGF -> generated/ 头文件；模型获取见文末许可节）
python tools/bake_model.py <MODELS>/nss_v1_0_1_high_int8.vgf --name high \
    -o generated --shaders ../nss_kernel/shaders
python tools/bake_model.py <MODELS>/nss_v1_0_1_mid_low_int8.vgf --name midlow \
    -o generated --shaders ../nss_kernel/shaders

# 2) 编译 DLL（build.sh 顶部的 MSVC/SDK 路径按本机修改）
bash build.sh            # 或 bash build.sh clean

# 3) 测试（Python 宿主，直连 vulkan-1.dll）
python test_dll.py --h 544 --w 960 -o out.npz
python test_dll.py --quality midlow --h 272 --w 480

# 4) 与 Arm 官方对拍
python test_dll.py --input ../nss_kernel/ref_official/out_high/out_input_tensor.npy -o out.npz
```

## C API 用法

```c
#include "nss_dp4a.h"

NssDp4aContext *ctx = NULL;
NssDp4aCreateInfo ci = {0};
ci.instance        = (uint64_t)myInstance;
ci.physicalDevice  = (uint64_t)myPhysicalDevice;
ci.device          = (uint64_t)myDevice;
ci.queue           = (uint64_t)myQueue;
ci.queueFamilyIndex = myComputeQueueFamily;
ci.apiVersion      = myApiVersion;          /* Vulkan 1.3 = 0x00403000 */
ci.quality         = NSS_DP4A_QUALITY_HIGH;
ci.width           = 960;                    /* 输入分辨率，8 的倍数 */
ci.height          = 544;
nssDp4aCreateContext(&ci, &ctx);

/* 每帧：录进宿主自己的命令缓冲，与前后处理同批提交 */
NssDp4aDispatchInfo di = {0};
di.input.buffer          = (uint64_t)myInputBuffer;   /* int8 NHWC [H][W][12] */
di.outputKpn.buffer      = (uint64_t)myKpnBuffer;     /* int8 [H/4][W/4][36 或 16] */
di.outputTemporal.buffer = (uint64_t)myTemporalBuffer;/* int8 [H][W][4] */
di.width  = 960;
di.height = 544;
nssDp4aRecord(ctx, (uint64_t)myCommandBuffer, &di);

/* 提交由宿主决定 */
vkEndCommandBuffer(myCommandBuffer);
vkQueueSubmit(...);

nssDp4aDestroyContext(ctx);
```

**注意**：`nssDp4aRecord` 只录制、不提交。这样前后处理与本库可以同批提交，避免额外的同步开销。

创建上下文前可探测设备能力，用于降级决策：

| 导出 | 用途 |
|---|---|
| `nssDp4aQueryDeviceCaps` | Vulkan 特性 / DP4A 加速位查询 |
| `nssDp4aQueryCooperativeMatrix` | coop-mat 形状探测（sint8×sint8→sint32，subgroup） |
| `nssDp4aGetBackend` | 上下文实际使用的后端（TC / DP4A） |
| `nssDp4aGetDispatchTimes` | 上次 record 每次 dispatch 的 GPU 耗时（纳秒，timestamp） |
| `nssDp4aGetScratchSize` / `nssDp4aGetInternalMemoryUsage` | 内部显存占用 |

---

## 前提

| 项 | 要求 |
|---|---|
| Vulkan | 1.3（或 1.2 + `VK_KHR_shader_integer_dot_product`） |
| 设备特性 | `shaderIntegerDotProduct` + `shaderInt64` |
| Tensor Core 路径（可选） | `VK_KHR_cooperative_matrix` + `cooperativeMatrix` 特性；sint8×sint8→sint32、subgroup scope 的形状（NVIDIA int8 为 16×16×32 / 16×8×32）；**不支持时自动整体回落 DP4A** |
| 硬件加速（DP4A 路径） | `integerDotProduct4x8BitPackedSignedAccelerated`；无则驱动展开，性能下降 |
| 输入尺寸 | 宽高必须是 **8 的倍数**（结构性约束：网络含三次 2× 下采样至 1/8 再逐级恢复）。注意宿主传的**不是渲染分辨率，而是 padded 尺寸**：`alignUp(渲染宽高, 8)` |
| **不需要** | `VK_ARM_data_graph`、`VK_ARM_tensors`（这正是绕开仿真层的关键） |

**Vulkan 是运行时动态加载的**（`LoadLibrary("vulkan-1.dll")` + `vkGetInstanceProcAddr`），
DLL 没有链接期 Vulkan 依赖，也不挑 Vulkan SDK 版本。宿主的 loader 入口可通过
`createInfo.vkGetInstanceProcAddr` 传入。

---

## 文件

| 文件 | 说明 |
|---|---|
| `include/nss_dp4a.h` | **公开 C API**（照搬 ffx-api 风格） |
| `src/nss_dp4a.cpp` | DLL 主体：上下文、资源、record、逐层后端选路 |
| `src/nss_dp4a_vk.{h,cpp}` | Vulkan 动态加载层 + 设备/coop-mat 探测（不链接 vulkan-1.lib） |
| `src/nss_dp4a_tc.{h,cpp}` | Tensor Core 后端：层规划与权重重排（wTc 布局） |
| `src/nss_dp4a_model.{h,cpp}` | 运行时形状推导（尺寸无关） |
| `tools/` | 烘焙与参考工具链：`bake_model.py` / `vgf_extract.py` / `vgf_graph.py` / `spv_graph.py` / `nss_ref.py` / `nss_compare.py` / `nss_kernel_plan.py` / `vk_probe.py` + `weights_high/` 逐层参考权重 |
| `generated/` | **烘焙产物**，由 `tools/bake_model.py` 生成 |
| `test_dll.py` | Python 宿主测试（`--quality high\|midlow`、直连 vulkan-1.dll） |
| `build.sh` | MSVC 构建脚本 |

### generated/ 的内容

| 文件 | 内容 |
|---|---|
| `nss_model_common.h` | 共用类型（`LayerDesc` / `OutDesc` / `KernelKind`） |
| `nss_model_data_<name>.h` | 权重 / bias+修正 / multiplier / shift / LUT |
| `nss_model_plan_<name>.h` | 逐层几何参数（**尺寸无关**） |
| `nss_spirv_embed.h` | 四个内核的 SPIR-V 字节码（conv_rq / conv_tc / resize2x / concat_copy） |

**DLL 里不做任何 VGF 解析**——所有内容都是编译期常量。

> 模型结构事实：HIGH 与 MID_LOW 的 conv 主干同构（32→64→32），仅 KPN 输出头
> 36 通道 vs 16 通道不同；MID 与 LOW 共用 `mid_low` 权重。

---

## 关键设计：尺寸无关的烘焙

实测发现**权重与量化参数完全与输入分辨率无关**，只有张量形状依赖输入 H/W。
所以烘焙表只存几何参数，C++ 侧运行时推导形状：

```
out_h = (in_h + padT + padB - kh) / strideH + 1
```

结果是 **一个 DLL 支持任意 8 的倍数尺寸**，不需要为每个分辨率重新烘焙。
（实测：128×128 与 544×960 用同一个 DLL，输出都正确。）

网络只有 5 个算子、通道与核全固定，唯一变量是 H/W，所以推导就是十几行算术。

## 关键设计：Tensor Core 路径

- 权重在上下文创建时从烘焙的 `[oc][K/4]` 重排为 MMA 的 `[chunk][k][n]`（wTc 布局），
  DP4A 路径下这些缓冲为空、不占显存。
- conv_tc 内核当前按 NVIDIA 的 **M=16 N=16 K=32** sint8 形状编译（形状是 coop-mat
  类型的编译期常量，运行时不可变）；形状探测只接受 16×16×32 / 16×8×32。
- **跨厂商现状**：AMD WMMA（int8 16×16×16）、Intel XMX（DPAS）的形状不同，目前不会
  命中探测 → 自动走 DP4A 路径。泛化需要：内核形状宏化出多份变体 + 探测放宽 +
  wTc 重排参数化，权威接口是 `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`。
- AMD 另需注意 WMMA 要求 wave32（`VK_EXT_subgroup_size_control` 固定 subgroup size）。

---

## 构建环境

| 项 | 要求 |
|---|---|
| 编译器 | MSVC x64（14.4x 实测可用） |
| Windows SDK | 10.0.26100 实测可用 |
| Vulkan 头文件 | Vulkan SDK，或 Arm neural-graphics SDK 自带的 `vulkan-headers` |

`build.sh` 已把 `vcvars64.bat` 会设的 PATH/INCLUDE/LIB 内联写好（顶部变量按本机
路径修改）——因为在 Git Bash 里调 vcvars 的批处理不可靠，而且 **`cl.exe` 不接受
MSYS 风格的 `/e/...` 路径**。

---

## 踩过的坑

| 现象 | 根因 |
|---|---|
| `无法打开包括文件 cstdio` | INCLUDE 用了 MSYS 路径（`/f/...`），`cl.exe` 不认。必须用 `F:/...` 风格 |
| `LNK1104 无法打开 LIBCMT.lib` | 只给了 `/I` 没给 `/LIBPATH`，链接期找不到运行库 |
| `error C2664 无法将 uint64_t 转为 VkDevice` | 公开 API 用 `uint64_t` 传句柄，内部要转成类型化句柄存起来 |
| 层表初始化「初始值设定项太多」 | 手写字段顺序与结构体不匹配。**改用 C++20 指定初始化器**（`.field = value`），彻底消除这类错误 |
| `nss_model::LayerDesc` 无法转为 `nss_model_midlow::LayerDesc` | 两个命名空间下是**不同类型**。类型定义抽到共享的 `nss_baked` 命名空间 |
| `model mismatch` | 模型按 544×960 装载但传了 128×128。`CreateInfo` 加上 width/height |
| 输入尺寸写进 createInfo 但仍是旧值 | 测试脚本在创建上下文**之后**才读输入文件。读取要提前 |

---

## 尚未做的优化

| 项 | 说明 |
|---|---|
| 输入边框拷贝 | 当前每帧在库内做一次带边框的拷贝（逐行 `vkCmdCopyBuffer`）。若宿主直接提供带边框的缓冲，可完全省掉 |
| 描述符集复用 | 当前每次 record 重新分配描述符集。可预建并只更新绑定 |
| 输出拷贝 | 当前从内部缓冲拷到宿主缓冲。若宿主接受"带边框"的输出布局，可省掉 |
| CONCAT/RESIZE 融进消费者卷积 | 19 → 16 次 dispatch |
| 跨厂商 coop-mat 形状泛化 | 见「关键设计：Tensor Core 路径」——AMD 16×16×16 / Intel DPAS 需形状宏化 + 探测放宽 + wTc 重排参数化 |
| 2×2 conv 变体 | `conv_rq2x2` 变体待定位问题（环境变量 `NSS_DP4A_CONV2X2=1` 可复现），主线默认关闭 |

---

## 许可与模型来源

- **代码**（`src/`、`include/`、`tools/`、脚本）：自研实现，MIT 许可发布（© 2026 eastear23333）。
  本库是独立推理后端，**不含** Arm 任何仓库的派生代码；与
  `neural-graphics-sdk-for-game-engines` / `ai-ml-emulation-layer-for-vulkan`
  的关系是算法行为对齐与位精确验证参考。
- **模型权重**（`generated/nss_model_*.h` 烘焙数据、`tools/weights_high/` 逐层参考）：
  来源于 Arm 的 NSS v1_0_1 模型（`nss_v1_0_1_high_int8.vgf`），其再分发遵守
  **Arm AI Model Community License**。原始 `.vgf` 可从 Arm neural-graphics SDK /
  AI Model Community 渠道获取后用 `tools/bake_model.py` 重新生成。
- 上游模型与规范：Arm neural-graphics SDK（MIT）与 Arm 模型库（模型许可）。
