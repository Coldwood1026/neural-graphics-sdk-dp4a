# Porting AMD's compute-shader optical flow back into this fork

**Status: plan only. Nothing in this document has been implemented.**

This exists because an earlier attempt at this failed, and failed in a way worth recording:
the reference implementation was copied in wholesale before checking whether the two trees
were compatible. They are not, and the incompatibility is structural rather than a matter
of missing files. The attempt was reverted; the tree builds with the fork's optical flow
intact. What follows is the analysis that should have come first.

Source of truth for the reference: **AMD FidelityFX SDK 1.1.3**, at
`<ref>/sdk/...` (the fork is derived from it).

---

## 1. Why this is a rewrite and not a restore

The two optical flow components are not the same code with pieces missing. They are two
different implementations that happen to share a filename.

### Function inventories

| Reference (`ffx_opticalflow.cpp`, 67,613 B, 987 lines) | Fork (48,288 B, 888 lines) |
|---|---|
| `getPipelinePermutationFlags(uint32_t, FfxPass, bool fp16, bool force64, bool)` | `getPipelinePermutationOptions(uint32_t contextFlags)` |
| `GetOpticalFlowTextureSize` | **deleted** |
| `GetOpticalFlowHistogramSize` | **deleted** |
| `GetGlobalMotionSearchDispatchSize` | **deleted** |
| `GetSCDHistogramTextureWidth` | **deleted** |
| `opticalflowCreate` / `opticalflowRelease` | `opticalFlowVkCreate` / `opticalFlowVkRelease` |
| `scheduleDispatch(context, pipeline, name, dx, dy, dz)` — **once per pass, seven per frame** | `scheduleDataGraph(context, pipeline, reset)` — **once per frame** |
| `dispatch` | `ArmOpticalFlowDispatch` |
| — | `createResourceFromDescription` |
| — | `scheduleMVHintsAndCopyResources`, `scheduleFragmentMVHints` |

The four deleted `Get*` functions are the geometry derivation for the pyramid, the SCD
histogram and the global motion search. They went away because a data-graph pipeline
derives that geometry itself, from the graph. Bringing the compute path back means
bringing all four back.

### The pass enum

```c
/* Reference — seven compute passes */
FFX_OPTICALFLOW_PASS_PREPARE_LUMA
FFX_OPTICALFLOW_PASS_GENERATE_OPTICAL_FLOW_INPUT_PYRAMID
FFX_OPTICALFLOW_PASS_GENERATE_SCD_HISTOGRAM
FFX_OPTICALFLOW_PASS_COMPUTE_SCD_DIVERGENCE
FFX_OPTICALFLOW_PASS_COMPUTE_OPTICAL_FLOW
FFX_OPTICALFLOW_PASS_FILTER_OPTICAL_FLOW
FFX_OPTICALFLOW_PASS_SCALE_OPTICAL_FLOW

/* Fork — one */
FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH = 0,
FFX_OPTICALFLOW_PASS_COUNT
```

### The public initialisation flags — an API-level conflict

```c
/* Fork                                            Reference */
FFX_OPTICALFLOW_ENABLE_DEPTH_INVERTED    = 1<<0    FFX_OPTICALFLOW_ENABLE_TEXTURE1D_USAGE = 1<<0
FFX_OPTICALFLOW_ENABLE_MV_HINTS_FRAGMENT = 1<<1
```

Same bit, different meaning. This is the reason the change cannot be confined to the
backend: any host that compiles against this header sees a different ABI.

**Mitigating fact:** the MV-hints path is fork-only and has **no consumer anywhere in the
tree** — no sample, no effect, and frame interpolation does not use it. It exists for
external hosts that pass `FFX_OPTICALFLOW_ENABLE_MV_HINTS_FRAGMENT`. So preserving it is
desirable but not required for anything in-repo to keep working.

---

## 2. The resource identity scheme differs

| | Fork | Reference |
|---|---|---|
| Identifiers defined in | `ffx_opticalflow_private.h` (host-side), 11 of them | `sdk/include/FidelityFX/gpu/opticalflow/ffx_opticalflow_resources.h` (**GPU-side, shared with the shaders**) |
| Prefix | `FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_*` | `FFX_OF_RESOURCE_IDENTIFIER_*` |
| Structure | `srvResources[]` / `uavResources[]` | a single `resources[]` |

The reference's arrangement is better engineering — the shader and the host agree on
resource identity by construction, from one header. The fork's identifiers are shaped
around its data-graph bindings.

Consequence: the reference's compute passes need the intermediate resources the fork no
longer declares — the luminance pyramid levels, the SCD histogram, the SCD divergence
buffer, the previous-flow buffers. **These must come back.** This is the single largest
piece of the work and it is why a backend-only fix is not possible.

---

## 3. The shader permutation scheme — the actual blocker

This is what made the first attempt fail one compiler error at a time.

The shared driver `sdk/include/FidelityFX/gpu/CMakeCompileShaders.txt`:

| | Fork | Reference |
|---|---|---|
| Size | 241 lines | 141 lines |
| Verdict | **rewritten** | original |
| Variants per pass | **two** — `_permutations.h`, `_16bit_permutations.h` | **four** — plus `_wave64_permutations.h`, `_wave64_16bit_permutations.h` |
| Why | it sets `SUPPORT_WAVE64 OFF` for GLSL, and this fork converted every shader to GLSL | it emitted all four unconditionally |

So on the Vulkan side of this fork **no wave64 permutation exists at all**, and the fork's
own accessors are written against that two-variant scheme:

```cpp
/* ffx_nss_shaderblobs.cpp — the local convention */
static FfxShaderBlob nssGetPreprocessPassPermutationBlobByIndex(
        uint32_t permutationOptions, bool is16bit)
{
    if (is16bit) { /* 16bit table */ }
    /* fp32 table */
}
```

The reference's `ffx_opticalflow_shaderblobs.cpp` instead includes all four headers per
pass and selects between them with an `isWave64` argument. Given the fork's driver, three
of those four headers are never generated.

**The obvious-looking fix is the wrong one.** Restoring wave64 permutation generation in
the shared driver would change the permutation set of *every* effect in the tree (NSS and
frame interpolation included), for the benefit of one component. Do not do that.

**The right fix** is to rewrite the optical flow accessor to the two-variant convention,
exactly as the fork's NSS and frame-interpolation accessors already are.

---

## 4. Complete interface-divergence list

Everything below was confirmed by compiling the reference component against the fork's
headers. The list is complete as of that build.

| Divergence | Where | Fix |
|---|---|---|
| `fpCreatePipeline` does not exist | `FfxInterface` | → `fpCreateComputePipeline` (the seven OF passes are all compute) |
| `wchar_t` name fields | `FfxPipelineState::name`, binding names | → `char`; `L"..."` → `"..."` (101 literals in the component) |
| `wcscmp` / `wcscpy_s` | 3 + 36 sites | → `strcmp` / `strncpy` + explicit NUL |
| `jobLabel` not a member | `FfxGpuJobDescription`, behind `#ifdef FFX_DEBUG` in the fork | wrap all 32 assignments in `#ifdef FFX_DEBUG` |
| `FFX_RESOURCE_STATE_UNORDERED_ACCESS` | removed by the fork | → `FFX_RESOURCE_STATE_COMPUTE_UAV` (same value, `1 << 1`) |
| `std::wstring` debug names | 3 sites | → `std::string` |
| `FFX_OF_BINDING_IDENTIFIER_*` absent | fork's `ffx_opticalflow_private.h` | take the reference's private header (4,468 B vs 2,732 B) |
| `FfxOpticalflowPass` has one value | `ffx_opticalflow.h` | take the reference's enum; optionally keep the fork's MV-hints value as an addition |
| Init flag bit meaning differs | `ffx_opticalflow.h` | take the reference's flags; keep the fork's two as additions if MV-hints is to survive |
| Accessor signature | `ffx_shader_blobs.cpp` calls it with **four** arguments (`…, outBlob, outVertBlob`) | the fork widened the callback; the reference's accessor takes three |

### The trap that produced a false green build

`Copy-Item` preserves `LastWriteTime`. Every file taken from the reference tree is dated
2025-03, which is **older** than the objects already in `build/`. MSBuild therefore
decided nothing needed recompiling and reported success — while the reference component
had never been compiled at all. The giveaway was a DLL whose size had not changed by a
single byte.

**Every file copied from another tree must be `touch`ed, or the build result means
nothing.**

---

## 5. Blast radius — and it is smaller than it looks

Checked, not assumed:

- **Nothing in the tree creates an optical flow context except the optical flow component
  itself.** No sample, no effect. Frame interpolation does **not** instantiate one.
- Frame interpolation in this fork takes the flow result as a *host-supplied* resource
  (`params->opticalFlowVector`, `params->opticalFlowSceneChangeDetection`), registered in
  `ffx_frameinterpolation.cpp`. The reference instead had an internal pass,
  `FFX_FRAMEINTERPOLATION_PASS_OPTICAL_FLOW_VECTOR_FIELD`, to convert the flow into its own
  vector field — the fork deleted that shader, and it must stay deleted.
- The backend change is small: with the component restored, nothing calls
  `fpCreateOpticalFlowPipeline`, so the stub `CreateOpticalFlowPipelineVK` and its
  assignment are deleted rather than fixed, along with the now-unused
  `ffxGetVkOpticalFlow*` helpers left over from the ARM path.

---

## 6. Recommended strategy

**Adapt the reference component to the fork's interface, and rewrite the accessor to the
two-variant scheme. Do not touch the shared permutation driver.**

Files touched:

| File | Action |
|---|---|
| `sdk/src/components/opticalflow/ffx_opticalflow.cpp` | take the reference's, apply the divergence table in §4 |
| `sdk/src/components/opticalflow/ffx_opticalflow_private.h` | take the reference's |
| `sdk/include/FidelityFX/host/ffx_opticalflow.h` | take the reference's enum + flags, keep the fork's two flags as additions |
| `sdk/include/FidelityFX/gpu/opticalflow/*.h` | restore the reference's nine headers (resource identity, common, per-pass) |
| `sdk/include/FidelityFX/gpu/opticalflow/ffx_opticalflow_callbacks_glsl.h` | **merge**, do not replace: the fork's 341-byte stub carries its MV-hints defines |
| `sdk/src/backends/vk/shaders/opticalflow/*.glsl` | restore the reference's seven compute passes; keep the fork's three `mv_hints` shaders |
| `sdk/include/FidelityFX/gpu/opticalflow/CMakeCompileOpticalflowShaders.txt` | **merge** the two lists: seven compute passes + three mv_hints |
| `sdk/src/backends/shared/blob_accessors/ffx_opticalflow_shaderblobs.{cpp,h}` | take the reference's, rewrite the seven cases to `is16bit ? … : …`, add the fork's MV-hints cases, add `outVertBlob` |
| `sdk/src/backends/vk/CMakeShadersOpticalflow.txt` | restore `OPTICALFLOW_API_BASE_ARGS` (the fork stripped the `set(...)` but left the `include(...)`) |
| `sdk/src/backends/vk/ffx_vk.cpp` | delete the `CreateOpticalFlowPipelineVK` stub, its assignment, and the dead `ffxGetVkOpticalFlow*` helpers |

### Dependencies to restore with it

`ffx_opticalflow_compute_luminance_pyramid.h` includes `spd/ffx_spd.h`, so **SPD comes
back too**: five GPU headers and `ffx_spd_downsample_pass.glsl`. Note this contradicts the
fork's frame interpolation, which reduces colour with its own
`01_downsample_of_colour_pass` rather than SPD — that stays as it is; SPD is needed only
by the OF luminance pyramid.

### Deliberately *not* part of this strategy

- **Wave64 permutations.** See §3. Wrong fix, broad blast radius.
- **`FFX_FRAMEINTERPOLATION_PASS_OPTICAL_FLOW_VECTOR_FIELD`.** The fork's frame
  interpolation consumes flow as a host resource; adding the reference's conversion pass
  would be a second, independent change.
- **DX12 optical flow.** Not in scope; the fork has no D3D12 optical flow and the
  reference's variant would need the same treatment starting from its HLSL.

### Options rejected, and why

- *Implement the compute passes behind `CreateOpticalFlowPipelineVK`, leaving the
  component untouched.* Attractive at first, since then only the backend changes. It fails
  because the fork's component no longer declares or allocates the pyramid, SCD and
  previous-flow resources the passes read (§2), and a data-graph job carries textures
  only — there is nowhere to put them. It would also mean reimplementing, inside a
  backend, geometry derivation that belongs to the effect.
- *Restore wave64 generation in the shared driver so the reference accessor works
  unmodified.* One line, and it invalidates the permutation set of NSS and frame
  interpolation. Rejected on blast radius.

---

## 7. Verification, and what "done" means

Nothing about this can be declared correct without running it. In order of strength:

1. **Build**, with every copied file `touch`ed first (§4).
2. **The effect-level contract**: create an optical flow context, dispatch it, and confirm
   the flow output is non-degenerate. This requires a host, and none exists in the tree.
3. **Against the reference**: run the reference's own optical flow on the same input and
   compare. This is the only check that establishes equivalence, and it needs the reference
   to be built somewhere comparable.

There is **no existing optical flow test in this tree** — unlike NFRU, which has
`tools/regress.ps1` and an 18-stage reference ladder. That is a gap worth closing as part
of the work rather than after it: without it, "restored" means "compiles", which is exactly
the claim that was wrongly made once already in this port.

---

## 8. Sizing

| Piece | Estimate |
|---|---|
| Divergence table in §4 (mechanical, proven — it was done once and reduced ~120 errors to 4) | 1 pass |
| Accessor rewrite to two variants | 1 pass |
| Header/shaders/CMake merge | 1 pass |
| Backend cleanup | trivial |
| Build convergence | 1–2 passes |
| **Verification harness (new work, no precedent in tree)** | **the larger half** |

The mechanical port is a known quantity; it has already been carried to the point where
only the accessor remained. The unknown is verification, which is why §7 says what it
says.
