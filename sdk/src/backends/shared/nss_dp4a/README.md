# `nss_dp4a` — NSS int8 inference backend

A portable implementation of Arm's **NSS** (Neural Super Sampling) int8 graph, running as
ordinary compute shaders. Like its NFRU sibling, it needs no `VK_ARM_data_graph`, no
`VK_ARM_tensors`, and no Arm Vulkan ML emulation layer.

**Provenance.** The Vulkan half is not my work. `src/nss_dp4a.cpp`, `nss_dp4a_vk.*`,
`nss_dp4a_tc.*`, `nss_dp4a_model.*`, `include/nss_dp4a.h`, `tools/` and `generated/` come
from **eastear23333's `nss` project** (MIT © 2026 eastear23333), carried over verbatim
apart from the two changes listed under *Changes from upstream*. The author's own
documentation is preserved as [`UPSTREAM-README.md`](UPSTREAM-README.md). The **D3D12
half is written for this fork**, and is described below.

---

## Two backends, selected per layer

The upstream Vulkan context picks its kernel per layer at creation time:

| Backend | Kernel | Requires | Selected when |
|---|---|---|---|
| **Tensor Core** (default) | `conv_tc.comp` — cooperative-matrix MMA | `VK_KHR_cooperative_matrix` | convolutions with `cout >= 16` |
| **DP4A / OpSDot** | `conv_rq.comp` | `shaderIntegerDotProduct` | small layers, and any GPU without cooperative matrix |

The split is not arbitrary: an MMA's A-matrix staging cost is independent of the output
channel count, so **layers with few channels lose money on MMA** — the `cout = 4` output
head measures about 3× slower that way. Both shipped models therefore run 13 of their 14
convolutions on Tensor Core, with only the `oc = 4` temporal/output head falling back.

The D3D12 host has **no Tensor Core path** and runs everything on DP4A. D3D12's
counterpart to `VK_KHR_cooperative_matrix` is SM 6.9 WaveMatrix — different hardware,
different feature, and it would need its own kernel and its own selection logic. Because
the module already routes every layer to DP4A when cooperative matrix is unavailable,
a DP4A-only D3D12 host is a supported configuration, not a degraded one. It is simply
not the fast one where the hardware could do better.

---

## Layout

| Path | Role |
|---|---|
| `include/nss_dp4a.h` | Public C API (Vulkan-facing), upstream |
| `src/nss_dp4a.cpp` | Vulkan context: resources, pipelines, descriptor binding, the per-layer record loop |
| `src/nss_dp4a_vk.{h,cpp}` | Vulkan dynamic loading + device / cooperative-matrix probing |
| `src/nss_dp4a_tc.{h,cpp}` | Tensor Core layer planning and weight re-layout |
| `src/nss_dp4a_model.{h,cpp}` | **API-agnostic.** Runtime shape derivation plus the baked layer tables |
| `src/nss_dp4a_dx12.{h,cpp}` | **This fork.** D3D12 host, an emitter over `nss::Model::dispatches()` |
| `generated/` | Baked artefacts — see below |
| `shaders/glsl/` | Reference kernels, upstream |
| `shaders/*.hlsl` | D3D12 ports of those kernels |
| `tools/` | Upstream bake/reference toolchain, plus `embed_dxil.py` |

### Why the D3D12 host is not a translation of the Vulkan one

`nss_dp4a.cpp` **fuses planning and emission**. Inside a single per-layer loop body it
writes descriptors, binds the pipeline, computes the push constants, computes the
dispatch grid and issues `vkCmdDispatch`, with roughly 40 distinct Vulkan entry points
woven through context creation, resource management and the record path. There is no plan
object that could be emitted twice — which is exactly why the NFRU backend could be
ported with a shared planner and this one could not.

What it does have is `nss::Model`, and `Model::dispatches()` returns a
`vector<Dispatch>` that is already API-agnostic: kernel kind, input/output buffer
indices, input/output shapes, dispatch grid, a 20-word push-constant block, and the
weight / bias / multiplier / shift / LUT pointers for the layer. The geometry comes from
`generated/nss_model_plan_<quality>.h`, which is pure data.

So `nss_dp4a_dx12.cpp` is an **emitter over `Model::dispatches()`**. It shares `Model`
— and therefore the shape derivation and the baked weights — with the Vulkan host rather
than duplicating either. What it does duplicate is the ~200 lines of binding and grid
arithmetic that `nss_dp4a.cpp` keeps inline; those are marked in `RecordDispatches()` with
comments saying why they had to be copied rather than shared.

### What D3D12 makes easier here

**Weight upload needs a queue.** On Vulkan that was a real obstacle: the SDK's Vulkan
backend holds no `VkQueue`, and `vkCreateCommandPool` demands a queue family the device
was created with, so the module had to be taught to record its copies into a
caller-supplied command buffer and skip creating a command pool entirely. D3D12 places no
such constraint — an application may create additional queues at will — so the D3D12
context makes its own compute queue for the one-shot upload and never touches the host's.

**Binding is by heap offset.** One shader-visible `CBV_SRV_UAV` heap holds eight
descriptors per dispatch (six SRVs, two UAVs), built once at init and selected per
dispatch by offset, so recording a frame performs no descriptor writes at all.

---

## The numerics contract

The kernels are int8 throughout. The three things that are easy to get wrong, and which
the HLSL ports preserve deliberately rather than "tidying":

**64-bit requantisation.** `acc` is about 25 bits and `multiplier` reaches 31, so the
product can be 56 bits wide:

```
r = (acc · multiplier + (1 << (shift-1))) >> shift      // int64, arithmetic shift
r = clamp(r + out_zp, -128, 127)
```

**The implicit ReLU.** When `out_zp == -128` the clamp alone makes the result
non-negative. Do **not** add `max(x, 0)`: it would be applied to layers that were never
meant to be rectified.

**Out-of-range taps contribute nothing.** The border is `z_a = -128 = 0x80808080`, so
`(q_a − z_a) = 0`. Reading a border value, substituting `0x80808080`, and skipping the tap
are all strictly equivalent — which is what lets the graph input be read from the host's
tight buffer with no mirror copy, and the graph output be written straight back to the
host's buffer.

---

## Baked artefacts

Nothing in `generated/` is needed to *build* — it is all compiled-in data — but all of it
is needed to build:

| File | Contents |
|---|---|
| `nss_model_common.h` | Shared types (`LayerDesc`, `OutDesc`, `KernelKind`) |
| `nss_model_data_<quality>.h` | Weights, bias+correction, multiplier, shift, LUT |
| `nss_model_plan_<quality>.h` | Per-layer geometry, size-independent |
| `nss_spirv_embed.h` | SPIR-V for the four Vulkan kernels |
| `nss_shaders_dxil.h` | DXIL for the three D3D12 kernels (this fork) |

The DLL does no VGF parsing at all: every value is a compile-time constant.

### Size-independent baking

Measured fact: **the weights and quantisation parameters do not depend on the input
resolution**; only the tensor shapes do. So the baked tables carry geometry only, and the
C++ side derives shapes at runtime:

```
out_h = (in_h + padT + padB - kh) / strideH + 1
```

One DLL therefore covers any multiple of 8, with no per-resolution re-bake. The network
has five operators and fixed channels and kernels, so the derivation is a dozen lines of
arithmetic.

### Regenerating

* **SPIR-V (Vulkan)** — upstream: `tools/bake_model.py <vgf> --name <high|midlow> -o generated --shaders <nss_kernel>/shaders`. Requires the separate `nss_kernel` repository for the GLSL sources.
* **DXIL (D3D12)** — this fork: `shaders/build_shaders.ps1`. Unlike the NFRU module's SPIR-V, **every input here reproduces its checked-in output byte for byte.**

---

## Properties the host must respect

| Property | Requirement |
|---|---|
| Input size | width and height must be **multiples of 8** (the network downsamples three times to 1/8 and recovers). Note the host passes the **padded** size, `alignUp(renderSize, 8)`, not the render size. |
| Device features (Vulkan) | `shaderIntegerDotProduct` + `shaderInt64` |
| Tensor Core path (optional) | `VK_KHR_cooperative_matrix` + `cooperativeMatrix`, sint8×sint8→sint32, subgroup scope. Unavailable → the whole context falls back to DP4A. |
| Hardware acceleration (DP4A) | `integerDotProduct4x8BitPackedSignedAccelerated`; without it the driver expands the dot products and performance drops |
| **Not required** | `VK_ARM_data_graph`, `VK_ARM_tensors` — that is the entire point |

The input tensor is int8 NHWC `[H][W][12]`; the outputs are KPN coefficients
(`[H/4][W/4][36]` for HIGH, `16` for MID_LOW) and a temporal feedback tensor
(`[H][W][4]`). The host owns the device, queue and command buffer; the backend only
records, never submits.

---

## Known gaps

**1. The D3D12 host does not use the `CONV_2X2` variant.**
`conv_rq_2x2.dxil` is built and embedded, but `RecordDispatches()` dispatches every
convolution through the base kernel. The two are bit-identical by construction, so this is
a missed optimisation, not a correctness gap. It matters most on the `op32` layer, which
is `cout = 4` at full resolution — the case the variant exists for.

**2. The D3D12 host ignores the Tensor Core question entirely.** See above.

**3. NSS inference has not been numerically verified inside the SDK.** The upstream
Vulkan implementation is reported by its author to be bit-exact with Arm's official
output, and the Vulkan context here is that implementation, carried over unmodified. The
D3D12 host is new code and has been verified only as far as *it compiles and the module
links*. The kernel ports preserve the reference's index arithmetic, accumulation order and
width/sign behaviour, and `build_shaders.ps1` regenerates their DXIL byte for byte — but
"reproducible from source" and "produces the same bytes as the reference on hardware" are
different claims, and only the first has been established.

**4. Licensing.** The code is MIT (© 2026 eastear23333) and the notice must travel with
it. Everything in `generated/` and `tools/weights_*/` is derived from Arm's NSS v1_0_1
int8 model, whose own licence is the **Arm AI Model Community License** — not an
OSI-approved open-source licence, and not automatically covered by the MIT grant above.
The original `.vgf` is obtainable through Arm's channels and `tools/bake_model.py`
regenerates the tables from it. See the repository README for the full statement.
