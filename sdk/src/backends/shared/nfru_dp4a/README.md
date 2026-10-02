# `nfru_dp4a` — portable int8 DP4A inference backend

A self-contained implementation of the **NFRU v1** neural frame-rate-upscaling graph that
runs as ordinary compute shaders on **Vulkan** and **D3D12**. It replaces the
`VK_ARM_data_graph` / `VK_ARM_tensors` execution path the upstream SDK used.

* No vendor extension.
* No Arm Vulkan ML emulation layer.
* No request to the host for a queue, a fence, or a submission.
* No dependence on the SDK's shader pipeline: the shaders and the int8 weights are
  checked in as C headers.

---

## Why it exists

Upstream, the NFRU graph executed through Arm's Vulkan ML extensions:

```
vkCreateDataGraphPipelinesARM           create the graph
vkCreateDataGraphPipelineSessionARM     bind it to its own device memory
vkCmdBindDataGraphPipelineSessionMemoryARM
vkCmdDispatchDataGraphARM               run it
```

Those extensions exist only on Arm GPUs. On anything else the SDK required the
[Arm Vulkan ML Emulation Layer](https://github.com/arm/ai-ml-emulation-layer-for-vulkan)
— a Vulkan *layer* that interposes on device creation and fabricates support for
`VK_ARM_tensors` and `VK_ARM_data_graph`, translating them to something the driver
understands.

This backend removes that entire arrangement. The graph is a fixed, known int8
convolutional network, so it does not need a driver-side graph API at all — it needs
`dot4add_i8packed` (HLSL) / `OpSDot` (GLSL), which is core functionality on anything from
Turing / Pascal-era hardware onward.

---

## Architecture

The design rule is: **one API-agnostic planner, two thin API emitters.** Everything that
decides *what* to dispatch lives in shared code; the backends only decide *how* to say it.

| File | Role |
|---|---|
| `nfru_dp4a.h` / `.cpp` | Public C API. Context lifecycle, scratch sizing, `nfruDp4aRecord()`. |
| `nfru_device.h` | The `Device` interface the plan is recorded against. |
| `nfru_graph.cpp` | The NFRU v1 graph: op order, shapes, formats, pass structure. |
| `nfru_plan.cpp` | **`BuildRecordPlan()`** — the single planner. Push constants, dispatch extents, buffer offsets, barriers, pass order. |
| `nfru_vk.cpp` | `VkBackend` — turns a plan into `vkCmdDispatch` + descriptors. |
| `nfru_dx12.cpp` | `Dx12Device` — turns the *same* plan into `Dispatch` + root descriptors. |
| `nfru_model_baked.h` | Generated. int8 weights, per-channel multiplier/shift, biases. |
| `nfru_shaders_spv.h` | Generated. Embedded SPIR-V. |
| `nfru_shaders_dxil.h` | Generated. Embedded DXIL. |

Because the plan is built once and only emitted twice, **"Vulkan and D3D12 are
bit-identical" is a property of the code rather than a coincidence to be re-verified** —
and it has been measured (see *Verification*).

### The graph

19 dispatches: 16 convolutions, 1 resize, 2 concats.

```
conv1 → conv2 → conv3 → skip1_conv → conv5 (stride 2)
      → conv5a | conv5b | conv5c → conv5c_1 | conv5d → conv5d_1 → conv5d_2
      → concat → conv5e (1×1) → nearest ×2 → conv6
      → concat (skip1) → conv7 → output_conv_mv
```

104,004 parameters.

---

## The numerics contract

Getting these exactly right is the whole problem, so they are stated precisely.

**Layout.** Activations are int8 NHWC, four channels packed per `uint32`. Weights are
packed `wpk[oc * K4 + (ky * kw + kx) * in_c4 + ic4]` where `K4 = kh * kw * in_c4`.

**Kernel.**

```
acc = Σ q_a · q_w                        (OpSDot / dot4add_i8packed)
bc  = bias / acc_scale − z_a · Σ w
r   = ((int64(acc + bc) * M + (1 << (S-1))) >> S) + out_zp     clamp to int8
```

The **`int64` is not optional**: `acc + bc` can exceed 32 bits before the shift.

**Out-of-range taps** read `0x80808080`, i.e. four int8 `-128`, which is the input zero
point. This is what makes borders and stride-2 padding come out right without a separate
clamp path.

**ReLU is implicit** whenever `out_zp == −128`. The output head deliberately uses
`out_zp = 44` so that it is *not* ReLU'd.

**Quantisation parameters** come from the QAT observers in `fru_v1_int8.pt`
(`network.auto_encoder.activation_post_process_<i>.scale` / `.zero_point`). The weight
scales match the published VGF exactly, which is how the mapping was confirmed.

---

## How the SDK drives it

Both backends reach the module through the same four calls, and both hand it the int8
storage directly — no tensor abstraction survives the boundary:

| SDK call | Vulkan (`ffx_vk.cpp`) | D3D12 (`ffx_dx12.cpp`) |
|---|---|---|
| `fpCreateDataGraphPipeline` | `CreateDataGraphPipelineVK` → `nfruDp4aCreateContext` | `CreateDataGraphPipelineDX12` → `nfruDp4aCreateContext` |
| `fpScheduleGpuJob` | stored in `pGpuJobs` | stored in `pGpuJobs` |
| `fpExecuteGpuJobs` | `executeGpuJobDataGraph` → `nfruDp4aRecord` | `executeGpuJobDataGraphDX12` → `nfruDp4aRecord` |
| tensor → storage | `Resource::aliasedTensorBufferResource` (a `VkBuffer`) | `Resource::resourcePtr` (an `ID3D12Resource*`) |

The pipeline's identity is carried in `FfxPipelineState::rootSignature`: Vulkan stores a
`PipelineLayout*`, D3D12 stores a `DataGraphPipelineDX12*`. Both are recovered at dispatch
time and both lead to the `NfruDp4aContext`.

**No queue is ever needed.** The module only *records* into the host's command buffer — it
never submits and never waits on a fence. `NfruDp4aCreateInfo::queue` is vestigial, and
`Init` never reads it. This matters because neither SDK backend holds a queue to give it.

---

## Regenerating the checked-in artefacts

Nothing here is needed for a normal build. It is needed when the kernels or the model
change.

### Shaders

```powershell
powershell -File sdk/src/backends/shared/nfru_dp4a/shaders/build_shaders.ps1
```

* **DXIL is reproducible.** `dxc` on the checked-in `.hlsl` with the flags the script uses
  reproduces the checked-in `.dxil` **byte for byte** (verified for both kernels).
* **SPIR-V is not** — see *Known gaps*.

### Weights

```bash
python tools/bake_c_header.py --packed <packed-model-dir> --out nfru_model_baked.h
```

The packed model directory is produced from the QAT checkpoint by
`tools/quantize_qat.py`, which also emits the int8 CPU reference the hardware regression
is judged against.

> The tools were written to run from the porting workspace, so most of them take the
> packed-model directory as an explicit argument rather than assuming a location. See
> `tools/README.md`.

---

## Verification

```powershell
powershell -File sdk/src/backends/shared/nfru_dp4a/tools/regress.ps1
```

Runs every check that has actually caught a bug in this port:

1. the whole 19-dispatch graph against the CPU reference, byte for byte;
2. each of the 18 recorded stages against the CPU stage ladder — this is what localises a
   fault to a single layer;
3. each of the 16 convolutions alone against its raw int32 accumulator reference;
4. Vulkan's output vs D3D12's, byte for byte, so "the two backends agree" is **measured**
   rather than asserted.

Last full run on an RTX 2060 (Turing, `shaderInt64` yes, `shaderInt8` **not** available —
the kernels use `OpSDot` with packed int8 and do not need it):

```
PASS vulkan|dx12 whole graph        0 / 518400
PASS stage ladder                   18 / 18
PASS convolutions                   16 / 16
PASS cross vulkan == dx12 == cpu    0 differing bytes
ALL CHECKS PASSED
```

Separately, `sdk/test/nfru_datagraph/` contains a harness that drives the same graph
**through the SDK's own backend interface** (`fpCreateBackendContext`,
`fpCreateResource` with `FFX_RESOURCE_TYPE_TENSOR`, `fpCreateDataGraphPipeline`,
`fpScheduleGpuJob`, `fpExecuteGpuJobs`) and compares against the same reference. Enable it
with `-DFFX_BUILD_NFRU_TEST=ON`. It builds and runs; see *Known gaps*.

---

## Known gaps

These are open, and are stated here rather than left for a reader to discover.

**1. The embedded SPIR-V is not reproducible from the sources in this tree.**
The checked-in `.spv` blobs are the ones validated bit-exact on hardware, but recompiling
either the `.comp` or the `.hlsl` sources produces a *different, larger* blob:

| source | result |
|---|---|
| checked-in `nfru_conv_rq.spv` | 18,080 B |
| `nfru_conv_rq.comp` via glslang (vulkan1.0/1.1) | 18,140 B |
| `nfru_conv_rq.comp` via glslang (vulkan1.3) | 18,152 B |
| checked-in `nfru_support.spv` | 5,104 B |
| `nfru_support.comp` via glslang (vulkan1.3) | 5,128 B |

The likely cause is a different glslang version and/or a since-edited source. Consequence:
**the DX12 path is reproducible from source and the Vulkan path currently is not.** Fixing
it means regenerating the Vulkan blobs and re-running the full regression; until that is
done, treat the SPIR-V as a checked-in binary, not as a build product.

**2. NSS is not implemented here.** NSS's graphs are a *different* network — they consume a
preprocessed tensor and emit KPN coefficients plus temporal feedback. Running NFRU's
weights against them would silently produce plausible garbage, so an NSS model is refused
at pipeline creation *and* again at dispatch, rather than approximated.

**3. Only NFRU v1 is implemented.** The model is switched on
`FfxDataGraphBlob::graphEntryPoint`; `"nfru_v1_int8"` is the only value accepted.

**4. The SDK-level harness does not yet pass.** It builds and runs and reaches
`ffxGetScratchMemorySizeVK`, where it hits an unrelated wall: the Vulkan backend resolves
every entry point through a process-wide function table populated by `InitVulkanWrapper()`,
and that symbol is **not exported**. The only exported way to trigger it is
`ffxCreateContext` with a VK backend desc, which needs a fuller desc chain than the
harness currently supplies. The module's own regression (above) is unaffected and passes.

**5. The weights are model-derived and are not MIT.** See *Licensing*.

---

## Licensing

The code in this directory is part of the SDK and is under the SDK's MIT licence, with the
exception of the model-derived artefacts.

`nfru_model_baked.h` is a **derived form of the NFRU v1 int8 model**, which Arm publishes
on Hugging Face under `license: other` — specifically the **Arm AI Model Community License
v1.0**. That is not an OSI-approved open-source licence. Redistributing the baked weights
in a repository whose overall licence is MIT needs a deliberate decision; if the answer is
no, this file must be produced by the reader from the checkpoint rather than shipped, and
`tools/bake_c_header.py` exists precisely so that it can be.

The golden input and reference output used by the regression are derived from the same
model and carry the same question.
