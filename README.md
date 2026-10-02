# Neural Graphics SDK for Game Engines

**English** | [简体中文](README.zh-CN.md)

The Neural Graphics Software Development Kit (SDK) is Arm's unified graphics SDK for
multiple rendering use cases across diverse game engines and platforms. Derived from
[AMD FidelityFX SDK 1.1.3](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), it
provides a modular, engine-agnostic framework that makes high-quality Neural Super
Sampling (NSS) and Neural Frame Rate Upscaling (NFRU) straightforward to integrate.

**This tree is a modified fork.** See [What this fork changes](#what-this-fork-changes).

---

## What this fork changes

Upstream, the NFRU inference graph executed through Arm's Vulkan ML extensions
(`VK_ARM_tensors`, `VK_ARM_data_graph`). Those exist only on Arm GPUs; on anything else the
SDK required the [Arm Vulkan ML Emulation Layer](https://github.com/arm/ai-ml-emulation-layer-for-vulkan),
a Vulkan layer that interposes on device creation and fabricates extension support.

This fork **deletes that execution path and that layer**, and replaces it with a portable
int8 DP4A inference backend that runs as ordinary compute shaders — `dot4add_i8packed` on
D3D12, `OpSDot` on Vulkan — which is core functionality on hardware from Turing / Pascal
onward.

| | Upstream | This fork |
|---|---|---|
| NFRU execution | `vkCmdDispatchDataGraphARM` + a data-graph session bound to its own device memory | dp4a compute kernels, recorded into the host's command buffer |
| Extensions required | `VK_ARM_tensors`, `VK_ARM_data_graph` | none |
| Emulation layer | required on non-Arm GPUs | **deleted** (`layer/`) |
| Backends | Vulkan only | **Vulkan and D3D12** |
| Weights / shaders | driver-side VGF payload | baked into the backend, checked in as C headers |
| Host must supply a queue | yes | **no** — the backend only records, never submits |

The replacement is a single API-agnostic planner (`BuildRecordPlan()`) with two thin
emitters, so *"the two backends agree"* is a property of the code rather than a
coincidence — and it has been measured. See
[`sdk/src/backends/shared/nfru_dp4a/README.md`](sdk/src/backends/shared/nfru_dp4a/README.md)
for the architecture, the int8 numerics contract, and the debug switches.

### Export surface

Removed from the built DLL: `vkCmdDispatchDataGraphARM`,
`vkCreateDataGraphPipelinesARM`, `vkCreateDataGraphPipelineSessionARM`,
`vkDestroyDataGraphPipelineSessionARM`, `vkBindDataGraphPipelineSessionMemoryARM`,
`VkDataGraphPipelineARM`.

Kept deliberately: the six `VK_ARM_tensors` entry points, because NSS's own coefficient and
feedback passes bind tensors. Those are not the inference path.

The Vulkan build's export table is **62 symbols — unchanged from upstream**.

### What is *not* claimed

See [Known gaps](#known-gaps). In short: neither backend's inference has been numerically
verified *inside the SDK*, the Tensor Core path is Vulkan-only, and the fork's Vulkan
optical flow is not restored.

---

## Features

| Component | Description | Inference backend |
|---|---|---|
| **NFRU** — Neural Frame Rate Upscaling | Frame interpolation generating one extra frame between every two rendered frames. | Portable dp4a backend, written for this fork. Vulkan **and** D3D12. |
| **NSS** — Neural Super Sampling | Temporal upscaling from render resolution to display resolution. | [`nss_dp4a`](sdk/src/backends/shared/nss_dp4a/README.md) — a separate int8 backend, upstream on Vulkan, with a D3D12 host added here. |

Both run as ordinary compute shaders (`OpSDot` on Vulkan, `dot4add_i8packed` on D3D12) and
neither needs a vendor extension or an emulation layer. They are **independent modules**:
different networks, different kernels, different toolchains, and they share no code beyond
being driven by the same SDK backends.

> NSS's Vulkan half is **not** this fork's work — see
> [Third-party components](#third-party-components). Its D3D12 half is.

---

## Requirements

| Tool category | Notes |
|---|---|
| CMake | 3.23 or newer |
| Visual Studio 2022 | MSVC toolset and the Windows SDK |
| Python 3 | Only for regenerating the baked artefacts; **not** needed to build |
| Vulkan headers | Vendored under `sdk/include/vulkan-headers`. The LunarG SDK is **not** required. |
| Vulkan runtime | `vulkan-1.dll` at run time. The backend resolves every entry point dynamically and never links `vulkan-1.lib`. |

There is **no** requirement for the Arm Vulkan ML Emulation Layer, and none for a vendor
data-graph extension.

### A note on the D3D12 backend

The D3D12 backend is restored from AMD FidelityFX SDK 1.1.3 and is deliberately
**compute-only and data-graph-only**:

* Graphics pipeline creation returns an explicit error rather than a compute PSO. The
  passes it used to serve depend on HLSL that this fork does not ship.
* The inference graph runs through the portable dp4a backend, which carries its own premade
  DXIL and never asks the SDK for a shader.
* It builds without SPIRV-Tools and without the shader-permutation step, so a D3D12 build is
  substantially lighter than a Vulkan one.

---

## Quick Start

### 1. Clone

```bash
git clone <this repository>
cd neural-graphics-sdk-for-game-engines
```

### 2. Build

> **Use a path containing only ASCII characters, or map one first.**
>
> The shader compiler (`sdk/tools/binary_store/FidelityFX_SC.exe`) cannot handle non-ASCII
> paths: it misreads a UTF-8 path as the local ANSI code page and fails with a
> `create_directory` error naming a mangled directory. A **directory junction does not work
> around this** — CMake resolves a junction back to its real path. A **subst drive does**,
> because CMake then records pure `X:/...` paths.
>
> ```powershell
> subst N: "D:\path\to\neural-graphics-sdk-for-game-engines"
> ```
>
> `build.py` also resolves paths from its own location, so it hits the same problem. On a
> non-ASCII path, configure with `cmake` directly as below.

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

> **The two backends produce the same DLL name.** `SDK_LIB_NAME` is
> `ngsdk_${FFX_PLATFORM_NAME}`, and both `vk` and `dx12` have `FFX_PLATFORM_NAME=windows_x64`,
> so both emit `ngsdk_windows_x64.dll`. Configure them into **separate build directories**,
> as above; they cannot coexist in one build tree.

### 3. Verify

```powershell
powershell -File sdk/src/backends/shared/nfru_dp4a/tools/regress.ps1
```

Runs the whole graph, an 18-stage ladder, all 16 convolutions, and a cross-backend byte
comparison. Last run on an RTX 2060 (Turing; `shaderInt64` yes, `shaderInt8` **not**
available — the kernels do not need it):

```
PASS vulkan|dx12 whole graph        0 / 518400
PASS stage ladder                   18 / 18
PASS convolutions                   16 / 16
PASS cross vulkan == dx12 == cpu    0 differing bytes
ALL CHECKS PASSED
```

There is also a harness that drives the same graph through the SDK's **own backend
interface** — that is, through the integration rather than around it. Enable it with
`-DFFX_BUILD_NFRU_TEST=ON`. See [Known gaps](#known-gaps).

### 4. Integrate

Add the SDK as a CMake subproject or link the built library, then create and dispatch an
NSS or NFRU context. See the [User Guide](docs/user_guide.md).

---

## Repository layout

```
sdk/src/backends/
    shared/nfru_dp4a/          NFRU's portable int8 inference backend
        README.md              architecture, numerics contract, how to regenerate
        nfru_plan.cpp          the API-agnostic planner -- read this first
        nfru_vk.cpp            Vulkan emitter
        nfru_dx12.cpp          D3D12 emitter
        shaders/               GLSL + HLSL kernels, prebuilt SPIR-V/DXIL, build_shaders.ps1
        tools/                 artefact generation, CPU reference, hardware regression
        nfru_model_baked.h     generated int8 weights (see Licensing)
        nfru_shaders_{spv,dxil}.h   generated embedded shaders
    shared/nss_dp4a/           NSS's int8 inference backend (independent module)
        README.md              architecture, provenance, what the D3D12 half adds
        UPSTREAM-README.md     the original author's own documentation
        nss_dp4a.cpp           Vulkan context (upstream)
        nss_dp4a_dx12.{h,cpp}  D3D12 host, written for this fork
        nss_dp4a_model.{h,cpp} API-agnostic: shape derivation + baked layer tables
        shaders/{glsl,dxil}/   upstream GLSL, the HLSL ports, and their DXIL
        generated/             baked weights, plans, and both embedded shader sets
    vk/                        Vulkan backend, incl. data_graphs/ model descriptors
    dx12/                      D3D12 backend, restored from FSR3 1.1.3
sdk/test/nfru_datagraph/       SDK-level end-to-end verification harness
sdk/include/vulkan-headers/    vendored; no LunarG SDK needed
```

The two inference modules are deliberately separate. They share no code beyond being
driven by the same backends: different networks, different kernels, different toolchains,
and `nfru_dp4a` has the API-agnostic planner that `nss_dp4a` does not.

---

## Known gaps

Stated here rather than left to be discovered.

1. **NSS's D3D12 host has no Tensor Core path.** D3D12's counterpart to
   `VK_KHR_cooperative_matrix` is SM 6.9 WaveMatrix — different hardware, different
   feature, its own kernel and its own selection logic. Everything runs on DP4A there. The
   module already routes every layer to DP4A when cooperative matrix is unavailable, so
   this is a supported configuration rather than a broken one; it is just not the fast one
   where the hardware could do better.
2. **NSS's `CONV_2X2` variant is built but never dispatched.** The D3D12 host runs every
   convolution through the base kernel. The two are bit-identical by construction, so this
   is a missed optimisation, not a correctness gap — it costs most on the `op32` layer,
   which is the `cout = 4` full-resolution case the variant exists for.
3. **Neither backend's inference has been numerically verified *inside the SDK*.**
   This is the honest headline. What has been established:
   * The NFRU module was verified bit-exact against a CPU reference on hardware, for both
     Vulkan and D3D12, with Vulkan ≡ D3D12 ≡ CPU at zero differing bytes — but through the
     module's **standalone** harness (`tools/regress.ps1`), not through the SDK.
   * The NSS Vulkan context is upstream's implementation, carried over unmodified, and its
     author reports it bit-exact with Arm's official output.
   * The NSS D3D12 kernels reproduce their checked-in DXIL byte for byte from the
     checked-in HLSL, and their ports preserve the reference's index arithmetic,
     accumulation order and 64-bit/sign behaviour.
   
   What has **not** been established is that a graph driven through the SDK's own backend
   interface produces the reference bytes. "Reproducible from source" and "produces the
   reference output on hardware" are different claims and only the first one is in hand.
4. **The SDK-level verification harness does not yet pass.** It builds and runs, and reaches
   `ffxGetScratchMemorySizeVK`, where it hits a wall unrelated to inference: the Vulkan
   backend resolves entry points through a process-wide function table populated by
   `InitVulkanWrapper()`, which is **not exported**. The only exported trigger is
   `ffxCreateContext` with a VK backend desc, which needs a fuller desc chain than the
   harness currently supplies. This is what blocks item 3 for both backends.
5. **Vulkan optical flow is not restored.** The fork had implemented optical flow as a
   `VK_ARM_data_graph_optical_flow` pipeline. With the data-graph path deleted it reports
   unsupported rather than silently degrading. AMD FidelityFX SDK 1.1.3 contains a complete
   compute-shader optical flow that could be ported back.
6. **The NFRU module's embedded SPIR-V is not reproducible from the sources in this tree.**
   `dxc` on the checked-in HLSL reproduces the shipped **DXIL byte for byte**; recompiling
   either the `.comp` or the `.hlsl` produces a *different, larger* SPIR-V blob. The shipped
   Vulkan blobs are the validated ones and are treated as checked-in binaries. Details and
   numbers in the module README. The NSS module does not have this problem.
7. **Upstream's NSS float32 caveat still applies.** In rare cases certain upscale ratios can
   trigger float32 precision issues in the dynamic offset LUT generation path, which may
   manifest as visible black line artifacts. Adjust the upscale ratio to avoid affected
   configurations.

---

## Documentation

- [User Guide](docs/user_guide.md) — build, integration, API reference, and samples
- [NFRU dp4a backend](sdk/src/backends/shared/nfru_dp4a/README.md) — architecture, numerics, regeneration
- [Backend tools](sdk/src/backends/shared/nfru_dp4a/tools/README.md) — artefact generation and the regression
- [NSS dp4a backend](sdk/src/backends/shared/nss_dp4a/README.md) — architecture, provenance, the D3D12 host
- [Release Notes](RELEASE-NOTES.md)

---

## Third-party components

`sdk/src/backends/shared/nss_dp4a` contains code that is **not** part of this fork and
**not** Arm's. With the exceptions noted below it comes from **eastear23333's `nss`
project**, MIT licensed © 2026 eastear23333:

| Path | Origin |
|---|---|
| `src/nss_dp4a.{cpp}`, `nss_dp4a_vk.*`, `nss_dp4a_tc.*`, `nss_dp4a_model.*`, `include/nss_dp4a.h` | upstream, carried over unmodified |
| `tools/`, `generated/nss_spirv_embed.h`, `shaders/glsl/` | upstream |
| `UPSTREAM-README.md` | the original author's documentation, preserved verbatim |
| `src/nss_dp4a_dx12.{h,cpp}`, `shaders/*.hlsl`, `shaders/dxil/`, `generated/nss_shaders_dxil.h`, `tools/embed_dxil.py` | **written for this fork** |

Two changes were made to upstream code, both in `src/nss_dp4a.cpp` and both minimal: an
`NSS_DP4A_INTERNAL` linkage mode in the public header (so the module's C API does not reach
the SDK's export table), and a guard so the command pool is only created on the self-submit
path — without which the module cannot be used by a host that owns no queue, which the SDK's
Vulkan backend is. **The MIT notice for this component must travel with it.**

---

## License

The Arm Neural Graphics SDK software in this repository is licensed under the
[MIT License](LICENSES/MIT.txt). The third-party NSS backend described above carries its own
MIT notice, © 2026 eastear23333.

**Exception — model-derived artefacts.** Two sets of checked-in files are derived from
Arm's int8 models, and Arm publishes those models under `license: other` — specifically the
**Arm AI Model Community License v1.0**. That is not an OSI-approved open-source licence and
it does **not** automatically follow the MIT licence above:

| Artefacts | Derived from |
|---|---|
| `shared/nfru_dp4a/nfru_model_baked.h`, and the golden input / reference output used by `tools/regress.ps1` | NFRU v1 int8 (`Arm/neural-frame-rate-upscaling`) |
| `shared/nss_dp4a/generated/nss_model_data_*.h`, `nss_model_plan_*.h`, `nss_spirv_embed.h` | NSS v1_0_1 int8 |

If those artefacts cannot be redistributed under this repository's licence, they must be
generated by the user from the source models instead of shipped — `nfru_dp4a`'s
`tools/bake_c_header.py` and `nss_dp4a`'s `tools/bake_model.py` exist so that is possible.
**This needs a deliberate decision before publication.**

The [Arm Neural Graphics SDK Developer Guide](docs/user_guide.md) is not licensed under the
MIT License. It is licensed separately under the Creative Commons
Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0):
https://creativecommons.org/licenses/by-nd/4.0/ — see [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)

Copyright © 2025–2026 Arm Limited.

Except for the rights expressly granted under that license, Arm reserves all rights in the
Developer Guide. No patent or trademark rights are granted by that license. The MIT License
applying to the Arm Neural Graphics SDK software does not apply to the Developer Guide.

---

## Trademarks and Copyrights

AMD is a trademark of Advanced Micro Devices, Inc.

AMD FidelityFX™ is a trademark of Advanced Micro Devices, Inc.

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.

Visual Studio, Windows are registered trademarks or trademarks of Microsoft Corporation in the US and other jurisdictions.
