# 面向游戏引擎的神经图形 SDK

[English](README.md) | **简体中文**

神经图形软件开发套件（Neural Graphics SDK）是 Arm 面向多种渲染用途、跨多种游戏引擎与平台的统一图形
SDK。它派生自 [AMD FidelityFX SDK 1.1.3](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)，
提供模块化、与引擎无关的框架，让高质量的神经超采样（NSS）与神经帧率提升（NFRU）易于集成。

**本仓库是一个修改版分支。** 见[本分支改了什么](#本分支改了什么)。

---

## 本分支改了什么

在上游，NFRU 推理图通过 Arm 的 Vulkan ML 扩展（`VK_ARM_tensors`、`VK_ARM_data_graph`）执行。这些扩展
只存在于 Arm GPU 上；在其他任何硬件上，SDK 都依赖
[Arm Vulkan ML Emulation Layer](https://github.com/arm/ai-ml-emulation-layer-for-vulkan) ——
一个 Vulkan **layer**，它拦截设备创建并伪造对这些扩展的支持。

本分支**删除了那条执行路径以及那个 layer**，改为一个可移植的 int8 DP4A 推理后端，以普通 compute
shader 运行 —— D3D12 上用 `dot4add_i8packed`，Vulkan 上用 `OpSDot` —— 这是 Turing / Pascal 之后硬件的
核心能力。

| | 上游 | 本分支 |
|---|---|---|
| NFRU 执行方式 | `vkCmdDispatchDataGraphARM` + 绑定自有显存的 data-graph session | dp4a compute kernel，只向宿主的命令缓冲区录制 |
| 需要的扩展 | `VK_ARM_tensors`、`VK_ARM_data_graph` | 无 |
| 模拟层 | 非 Arm GPU 上必需 | **已删除**（`layer/`） |
| 后端 | 仅 Vulkan | **Vulkan 与 D3D12** |
| 权重 / 着色器 | 驱动侧 VGF 载荷 | 烘进后端，以 C 头文件形式入库 |
| 宿主是否需提供 queue | 需要 | **不需要** —— 后端只录制，从不提交 |

这套替换由**一个与图形 API 无关的规划器**（`BuildRecordPlan()`）加**两个薄发射层**组成，因此
*「两个后端结果一致」是代码本身的性质*，而不是巧合 —— 并且已被实测。架构、int8 数值约定与调试开关见
[`sdk/src/backends/shared/nfru_dp4a/README.md`](sdk/src/backends/shared/nfru_dp4a/README.md)。

### 导出面

从产出 DLL 中**移除**：`vkCmdDispatchDataGraphARM`、`vkCreateDataGraphPipelinesARM`、
`vkCreateDataGraphPipelineSessionARM`、`vkDestroyDataGraphPipelineSessionARM`、
`vkBindDataGraphPipelineSessionMemoryARM`、`VkDataGraphPipelineARM`。

**有意保留**：六个 `VK_ARM_tensors` 入口点，因为 NSS 自身的系数与反馈 pass 要绑定张量。那些不属于推理路径。

Vulkan 构建的导出表为 **62 个符号 —— 与上游一致**。

### 不宣称做到的事

见[已知缺口](#已知缺口)。简言之：两个后端的推理都**没有在 SDK 内部**做过数值验证，Tensor Core
路径仅限 Vulkan，且本分支的 Vulkan 光流未恢复。

---

## 功能

| 组件 | 说明 | 推理后端 |
|---|---|---|
| **NFRU** —— 神经帧率提升 | 在每两帧渲染帧之间生成一帧插值帧，提升感知帧率且延迟可预测。 | 可移植 dp4a 后端，为本分支所写。Vulkan **与** D3D12。 |
| **NSS** —— 神经超采样 | 从渲染分辨率到时显示分辨率的时序超采样。 | [`nss_dp4a`](sdk/src/backends/shared/nss_dp4a/README.md) —— 独立的 int8 后端，Vulkan 部分来自上游，D3D12 宿主由本分支补上。 |

两者都以普通 compute shader 运行（Vulkan 上 `OpSDot`，D3D12 上 `dot4add_i8packed`），都不需要
厂商扩展或模拟层。它们是**两个互相独立的模块**：不同的网络、不同的内核、不同的工具链，除了同被 SDK
后端驱动之外不共享任何代码。

> NSS 的 Vulkan 部分**不是**本分支的成果 —— 见[第三方组件](#第三方组件)。它的 D3D12 部分是。

---

## 环境要求

| 类别 | 说明 |
|---|---|
| CMake | 3.23 或更高 |
| Visual Studio 2022 | MSVC 工具集与 Windows SDK |
| Python 3 | 仅用于重新生成烘死的产物；**构建不需要** |
| Vulkan 头文件 | 已内置于 `sdk/include/vulkan-headers`。**不需要** LunarG SDK。 |
| Vulkan 运行时 | 运行时需要 `vulkan-1.dll`。后端动态解析所有入口点，从不链接 `vulkan-1.lib`。 |

**不需要** Arm Vulkan ML Emulation Layer，也不需要任何厂商 data-graph 扩展。

### 关于 D3D12 后端

D3D12 后端自 AMD FidelityFX SDK 1.1.3 恢复而来，并且刻意**只支持 compute 与 data graph**：

* 图形管线创建会显式返回错误，而不是塞回一个 compute PSO。它原本服务的那些 pass 依赖本分支未附带的 HLSL。
* 推理图走可移植 dp4a 后端，其自带预制 DXIL，从不向 SDK 索取着色器。
* 它不构建 SPIRV-Tools，也不跑着色器排列步骤，因此 D3D12 构建比 Vulkan 构建轻得多。

---

## 快速开始

### 1. 克隆

```bash
git clone <本仓库>
cd neural-graphics-sdk-for-game-engines
```

### 2. 构建

> **请使用纯 ASCII 路径，或先映射一个。**
>
> 着色器编译器（`sdk/tools/binary_store/FidelityFX_SC.exe`）无法处理非 ASCII 路径：它会把 UTF-8 路径
> 按本地 ANSI 代码页误读，报出一个乱码目录名的 `create_directory` 错误。**目录联接（junction）绕不过
> 这个问题** —— CMake 会把联接解析回真实路径。**subst 虚拟盘可以**，因为 CMake 记录到的是纯 `X:/...`
> 路径。
>
> ```powershell
> subst N: "D:\path\to\neural-graphics-sdk-for-game-engines"
> ```
>
> `build.py` 也按自身位置解析路径，因此会遇到同样的问题。在非 ASCII 路径下，请直接按下述方式用 `cmake` 配置。

```powershell
# Vulkan
cmake -A x64 -S N:/ -B N:/build/vk `
      -DFFX_API_BACKEND=vk_windows_x64 -DFFX_FSR3_AS_LIBRARY=OFF -DFFX_BUILD_AS_DLL=ON
cmake --build N:/build/vk --config Release --parallel 8

# D3D12
cmake -A x64 -S N:/ -B N:/build/dx12 `
      -DFFX_API_BACKEND=dx12_windows_x64 -DFFX_FSR3_AS_LIBRARY=OFF -DFFX_BUILD_AS_DLL=ON
cmake --build N:/build/dx12 --config Release --parallel 8
```

> **两个后端产出同名 DLL。** `SDK_LIB_NAME` 是 `ngsdk_${FFX_PLATFORM_NAME}`，而 `vk` 与 `dx12` 的
> `FFX_PLATFORM_NAME` 都是 `windows_x64`，因此两者都产出 `ngsdk_windows_x64.dll`。请按上面的做法把它们
> 配置到**各自独立的构建目录**中；它们无法共存于同一棵构建树。

### 3. 验证

```powershell
powershell -File sdk/src/backends/shared/nfru_dp4a/tools/regress.ps1
```

会跑整图、18 级 stage 阶梯、全部 16 个卷积，以及跨后端逐字节比对。最近一次在 RTX 2060 上的运行
（Turing；`shaderInt64` 有，`shaderInt8` **没有** —— 内核不需要它）：

```
PASS vulkan|dx12 whole graph        0 / 518400
PASS stage ladder                   18 / 18
PASS convolutions                   16 / 16
PASS cross vulkan == dx12 == cpu    0 differing bytes
ALL CHECKS PASSED
```

另有一个夹具走 SDK **自身的后端接口** —— 也就是走集成路径而非绕开它。用
`-DFFX_BUILD_NFRU_TEST=ON` 打开。见[已知缺口](#已知缺口)。

### 4. 集成

把 SDK 作为 CMake 子项目加入，或链接构建出的库，然后创建并派发 NSS 或 NFRU 上下文。见
[用户指南](docs/user_guide.md)。

---

## 仓库结构

```
sdk/src/backends/
    shared/nfru_dp4a/          NFRU 的可移植 int8 推理后端
        README.md              架构、数值约定、如何重新生成产物
        nfru_plan.cpp          与 API 无关的规划器 —— 先读这个
        nfru_vk.cpp            Vulkan 发射层
        nfru_dx12.cpp          D3D12 发射层
        shaders/               GLSL 与 HLSL 内核、预编译 SPIR-V/DXIL、build_shaders.ps1
        tools/                 产物生成、CPU 参考、硬件回归
        nfru_model_baked.h     生成的 int8 权重（见「许可」）
        nfru_shaders_{spv,dxil}.h   生成的内嵌着色器
    shared/nss_dp4a/           NSS 的 int8 推理后端（独立模块）
        README.md              架构、来源、D3D12 部分补了什么
        UPSTREAM-README.md     原作者自己的文档
        nss_dp4a.cpp           Vulkan 上下文（上游）
        nss_dp4a_dx12.{h,cpp}  D3D12 宿主，为本分支所写
        nss_dp4a_model.{h,cpp} 与 API 无关：形状推导 + 烘焙层表
        shaders/{glsl,dxil}/   上游 GLSL、其 HLSL 移植、以及编译出的 DXIL
        generated/             烘焙权重、逐层计划、两套内嵌着色器
    vk/                        Vulkan 后端，含 data_graphs/ 模型描述符
    dx12/                      D3D12 后端，自 FSR3 1.1.3 恢复
sdk/test/nfru_datagraph/       SDK 级端到端验证夹具
sdk/include/vulkan-headers/    已内置；不需要 LunarG SDK
```

两个推理模块是**刻意分开**的。除了同被相同的后端驱动之外不共享任何代码：不同的网络、不同的内核、不同
的工具链，而且 `nfru_dp4a` 有那个与 API 无关的规划器，`nss_dp4a` 没有。

---

## 已知缺口

在此明确列出，而不是留给读者去发现。

1. **NSS 的 D3D12 宿主没有 Tensor Core 路径。** D3D12 侧对应 `VK_KHR_cooperative_matrix` 的是
   SM 6.9 的 WaveMatrix —— 不同的硬件、不同的特性，需要自己的内核和选路逻辑。那边一切走 DP4A。模块在
   cooperative matrix 不可用时本来就把所有层回退到 DP4A，所以这是一个受支持的配置而不是坏掉的配置；只是
   在硬件本可以更快的地方没有更快。
2. **NSS 的 `CONV_2X2` 变体编了但从未被派发。** D3D12 宿主所有卷积都走基础内核。两者按构造逐位一致，
   所以这是错过了一个优化而非正确性缺口 —— 代价最大的是 `op32` 层，也就是该变体存在的那个 `cout = 4`
   全分辨率情形。
3. **两个后端的推理都还没有在 SDK 内部做过数值验证。** 这是最该直说的一条。已确立的是：
   * NFRU 模块曾在真机上对 CPU 参考验证逐位精确，Vulkan 与 D3D12 两侧、且 Vulkan ≡ D3D12 ≡ CPU 零差异
     字节 —— 但那是通过模块**独立的**夹具（`tools/regress.ps1`），不是通过 SDK。
   * NSS 的 Vulkan 上下文是上游实现，原样搬入，其作者报告与 Arm 官方输出位精确一致。
   * NSS 的 D3D12 内核能由入库的 HLSL **逐字节**复现出所附 DXIL，且移植保留了参考实现的索引算术、
     累加顺序与 64 位/符号行为。
   
   **尚未**确立的是：一张图经由 SDK 自身的后端接口被驱动后，产出的就是参考字节。「可源码复现」与「在
   真机上产出参考输出」是两个不同的命题，目前手里只有前者。
4. **SDK 级验证夹具尚未通过。** 它能编译、能运行，并且走到了 `ffxGetScratchMemorySizeVK`，在那里撞上一堵
   与推理无关的墙：Vulkan 后端通过一张由 `InitVulkanWrapper()` 填充的进程级函数表来解析入口点，而该符号
   **未被导出**。唯一导出的触发途径是带 VK 后端 desc 调用 `ffxCreateContext`，而这需要比当前夹具更完整
   的 desc 链。正是这一条卡住了两个后端的第 3 项。
5. **Vulkan 光流未恢复。** 本分支原本把光流实现为 `VK_ARM_data_graph_optical_flow` 管线。data-graph
   路径删除后，它如实报告不支持，而不是静默降级。AMD FidelityFX SDK 1.1.3 含一套完整的 compute
   shader 光流，可以移植回来。
6. **NFRU 模块内嵌的 SPIR-V 无法由本树中的源码复现。** `dxc` 编译入库的 HLSL 可以**逐字节**复现出所附的
   **DXIL**；但重新编译 `.comp` 或 `.hlsl` 都会产出*不同且更大*的 SPIR-V blob。所附的 Vulkan blob 是
   经过验证的那一份，因此按入库二进制对待。细节与数字见模块 README。NSS 模块没有这个问题。
7. **上游关于 NSS 的 float32 注意事项仍然适用。** 极少数情况下，某些放大倍率会在 NSS 的动态偏移 LUT
   生成路径中触发 float32 精度问题，表现为放大输出中出现可见黑线伪影。可通过调整放大倍率规避受影响的配置。

---

## 文档

- [用户指南](docs/user_guide.md) —— 构建、集成、API 参考与示例
- [NFRU dp4a 后端](sdk/src/backends/shared/nfru_dp4a/README.md) —— 架构、数值、产物再生成
- [后端工具](sdk/src/backends/shared/nfru_dp4a/tools/README.md) —— 产物生成与回归
- [NSS dp4a 后端](sdk/src/backends/shared/nss_dp4a/README.md) —— 架构、来源、D3D12 宿主
- [发行说明](RELEASE-NOTES.md)

---

## 第三方组件

`sdk/src/backends/shared/nss_dp4a` 中含有**不属于**本分支、也**不属于** Arm 的代码。除下表标注者外，
它来自 **eastear23333 的 `nss` 项目**，MIT 许可 © 2026 eastear23333：

| 路径 | 来源 |
|---|---|
| `src/nss_dp4a.cpp`、`nss_dp4a_vk.*`、`nss_dp4a_tc.*`、`nss_dp4a_model.*`、`include/nss_dp4a.h` | 上游，原样搬入 |
| `tools/`、`generated/nss_spirv_embed.h`、`shaders/glsl/` | 上游 |
| `UPSTREAM-README.md` | 原作者自己的文档，逐字保留 |
| `src/nss_dp4a_dx12.{h,cpp}`、`shaders/*.hlsl`、`shaders/dxil/`、`generated/nss_shaders_dxil.h`、`tools/embed_dxil.py` | **为本分支所写** |

对上游代码做了两处改动，都在 `src/nss_dp4a.cpp`，且都很小：公开头里加了 `NSS_DP4A_INTERNAL` 链接模式
（使模块的 C API 不进入 SDK 的导出表），以及一个守卫使命令池只在自提交路径上创建 —— 没有它，该模块就
无法被一个不持有 queue 的宿主使用，而 SDK 的 Vulkan 后端正是如此。**这个组件的 MIT 声明必须随它一起分发。**

---

## 许可

本仓库中的 Arm 神经图形 SDK 软件采用 [MIT 许可证](LICENSES/MIT.txt)。上文所述的第三方 NSS 后端带有
它自己的 MIT 声明，© 2026 eastear23333。

**例外 —— 模型派生产物。** 有两组入库文件派生自 Arm 的 int8 模型，而 Arm 以 `license: other` 发布这些
模型，具体为 **Arm AI Model Community License v1.0**。这不是 OSI 认可的开源许可证，**不会**自动跟随上文
的 MIT 许可：

| 产物 | 派生自 |
|---|---|
| `shared/nfru_dp4a/nfru_model_baked.h`，以及 `tools/regress.ps1` 使用的 golden 输入与参考输出 | NFRU v1 int8（`Arm/neural-frame-rate-upscaling`） |
| `shared/nss_dp4a/generated/nss_model_data_*.h`、`nss_model_plan_*.h`、`nss_spirv_embed.h` | NSS v1_0_1 int8 |

如果这些产物无法在本仓库的许可下再分发，就必须由使用者从源模型自行生成而非随仓库分发 ——
`nfru_dp4a` 的 `tools/bake_c_header.py` 与 `nss_dp4a` 的 `tools/bake_model.py` 的存在正是为了让这件事
可行。**公开发布前需要对此做出明确决定。**

[Arm 神经图形 SDK 开发者指南](docs/user_guide.md) 不在 MIT 许可之下，而是单独采用
Creative Commons Attribution-NoDerivatives 4.0 International License（CC BY-ND 4.0）：
https://creativecommons.org/licenses/by-nd/4.0/ —— 见 [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)

Copyright © 2025–2026 Arm Limited.

除该许可明确授予的权利外，Arm 保留开发者指南的一切权利。该许可不授予任何专利或商标权利。适用于 Arm
神经图形 SDK 软件的 MIT 许可不适用于开发者指南。

---

## 商标与版权

AMD 是 Advanced Micro Devices, Inc. 的商标。

AMD FidelityFX™ 是 Advanced Micro Devices, Inc. 的商标。

Arm® 是 Arm Limited（或其子公司）在美国和/或其他地区的注册商标。

Vulkan 是 Khronos Group Inc. 的注册商标，Vulkan SC 标志是 Khronos Group Inc. 的商标。

Visual Studio、Windows 是 Microsoft Corporation 在美国和其他司法管辖区的注册商标或商标。
