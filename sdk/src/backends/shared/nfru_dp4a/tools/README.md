# NFRU dp4a — build and verification tools

These are the tools that produce the checked-in artefacts and that produce the reference
the hardware regression is judged against. Nothing here runs during a normal SDK build.

> **They were written during the port and run against a packed-model directory passed as
> an explicit argument.** They do not assume a fixed checkout layout, but they also do not
> guess: if a tool needs `--packed`, it will say so. Most were originally invoked from the
> workspace root as `python tools/<name>.py ...`.

---

## Artefact generation

The three checked-in headers, and the tools that make them:

| Tool | Input | Output |
|---|---|---|
| `embed_spv.py` | `shaders/nfru_conv_rq.spv`, `shaders/nfru_support.spv` | `nfru_shaders_spv.h` |
| `embed_dxil.py` | `shaders/dxil/*.dxil` | `nfru_shaders_dxil.h` |
| `bake_c_header.py` | packed-model directory | `nfru_model_baked.h` |

```bash
python tools/embed_spv.py  --conv shaders/nfru_conv_rq.spv --support shaders/nfru_support.spv --out nfru_shaders_spv.h
python tools/embed_dxil.py --conv shaders/dxil/nfru_conv_rq.dxil --support shaders/dxil/nfru_support.dxil --out nfru_shaders_dxil.h
python tools/bake_c_header.py --packed <packed-model-dir> --out nfru_model_baked.h
```

To rebuild the shader binaries themselves, use
`shaders/build_shaders.ps1` — and read the warning at the top of it first.

`validate_baked.py` checks a generated `nfru_model_baked.h` against the packed model, so a
baking mistake is caught before it reaches the GPU.

---

## From the QAT checkpoint to the packed model

| Tool | Role |
|---|---|
| `quantize_qat.py` | The checkpoint → int8 packed weights **and the int8 CPU reference**. This is the tool the hardware regression is ultimately judged against. |
| `quantize_from_ckpt.py` | Earlier variant of the same conversion. |
| `export_int8_weights.py` | Tensor-level export, for inspecting what actually got quantised. |
| `bake_nfru.py` | Bakes a model into the packed layout. |
| `gen_manifest.py` | Emits `nfru_packed.json`, the manifest describing the packed model. |

The quantisation parameters come from the QAT observers in the checkpoint
(`network.auto_encoder.activation_post_process_<i>.scale` / `.zero_point`). The weight
scales match Arm's published VGF exactly, which is how the mapping was confirmed — and is
worth re-checking with `export_int8_weights.py` if the checkpoint ever changes.

---

## The graph, and the references

| Tool | Role |
|---|---|
| `nfru_graph.py` | The NFRU v1 graph: op order, shapes, pass structure. Mirrors `../nfru_graph.cpp`. |
| `graph_topo.py` | Topological helpers over that graph. |
| `nfru_int8_ref.py` | The CPU int8 reference implementation. |
| `int8_sim.py` | Bit-exact integer simulation, used to pin down a miscompile. |
| `nfru_reference.py` | Float reference, for sanity rather than for byte comparison. |
| `gen_acc_ref.py` | Per-convolution **raw int32 accumulator** reference. This is what localises a fault to one convolution. |
| `gen_stage_ladder.py` | Per-stage reference data. This is what localises a fault to one layer. |
| `chain.py` | Chains the stages together. |

The two "reference" generators are worth understanding before debugging anything: when the
whole-graph check fails, the stage ladder says *which stage* and the accumulator reference
says *which convolution*, which turns a 19-dispatch graph into a single kernel to look at.

---

## `regress.ps1` — the hardware regression

```powershell
powershell -File tools/regress.ps1              # both backends
powershell -File tools/regress.ps1 -Backend dx12
```

Requires the standalone harnesses (`nfru_test.exe`, `nfru_test_dx12.exe`) and a
packed-model directory containing `golden_input_codes.bin` and
`reference_output_codes.bin`.

It clears every debug environment variable first. That is not tidiness: a stale
`NFRU_DP4A_STOP_AFTER` silently changes what is being measured and produces a green run
that means nothing.

---

## Debug switches the module honours

Useful when the regression fails and the references have narrowed it down:

| Variable | Effect |
|---|---|
| `NFRU_DP4A_BACKEND` | `vulkan` or `dx12` — force one backend. |
| `NFRU_DP4A_TRACE`, `_TRACE_PC`, `_TRACE_GEOM`, `_TRACE_BIND`, `_TRACE_SUPPORT` | Per-dispatch tracing. |
| `NFRU_DP4A_STOP_AFTER=<n>` | Stop after dispatch *n* (drives the stage ladder). |
| `NFRU_DP4A_STOP_TO_SCRATCH` | Write the stopped stage to scratch so it can be read back. |
| `NFRU_DP4A_LAYER_ONLY` | Run one convolution in isolation. |
| `NFRU_DP4A_DUMP_ACC` | Dump raw int32 accumulators instead of quantised output. |
| `NFRU_DP4A_DUMP_MARK` | Thread-marker mode: proves every thread ran, independently of the data. |
| `NFRU_DP4A_DUMP`, `NFRU_DP4A_INPUT_FOR_OP` | Dump the graph's intermediate tensors. |
| `NFRU_DP4A_RUN_TWICE` | Run the graph twice — catches state that leaks between runs. |
| `NFRU_DP4A_SUPPORT_SELFTEST` | Self-test the resize/concat kernel. |
| `NFRU_DP4A_DX12_DEBUG` | Enable the D3D12 debug layer and drain `ID3D12InfoQueue` to stdout. |

`NFRU_DP4A_DUMP_MARK` and `NFRU_DP4A_DX12_DEBUG` together are what turned two of the
longest debugging detours in this port into short ones. The D3D12 debug layer writes to
`OutputDebugString`, which is invisible from a shell — it has to be drained explicitly.
