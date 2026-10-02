#!/usr/bin/env python3
"""Bake the NFRU v1 int8 model into a complete dp4a-ready manifest.

This adapts the approach proven in the NSS desktop backend (`nss-main/tools/
bake_model.py`), which solves exactly the problem this port needs: the TOSA
`multiplier`/`shift` requantisation parameters ARE recoverable from the VGF, but
only with the correct SPIR-V operand offsets.

The operand offsets that matter (they are positional, and getting them wrong
silently substitutes tensor handles for scalars):

    OpExtInst <resultType> <result> <set> <instNum> <operands...>
    so the n-th named operand of a TOSA op lives at ops[n].
    In the NSS code's terms, in_operand(i) == ops[i + 2] because it indexes from
    the start of the raw operand list including resultType and result.

    CONV2D  : 2=pad 3=stride 4=dilation 7=input 8=weight 9=bias
              10=input_zero_point 11=weight_zero_point
    RESCALE : 2=scale32 3=rounding 4=per_channel 5=input_unsigned
              6=output_unsigned 7=input 8=multiplier 9=shift
              10=input_zero_point 11=output_zero_point

Two further details taken from the NSS backend that a naive port misses:
  * the input zero point must be folded out of the accumulator as a bias
    correction, `corr = -izp * sum(weights)`, otherwise every output is offset;
  * `output_zero_point == 128` (i.e. -128 as int8) marks a ReLU layer.

Usage:
    python bake_nfru.py ../../fru/nfru_v1_int8.vgf --json nfru_baked.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgftools"))
from vgf_graph import analyze                              # noqa: E402
from nss_ref import load_constants, const_array, as_int8    # noqa: E402

# Execution order of the convolutions, derived from the DAG in the IR.
LAYER_ORDER = [
    "conv1", "conv2", "conv3", "conv5", "skip1_conv",
    "conv5a", "conv5b", "conv5c", "conv5c_1",
    "conv5d", "conv5d_1", "conv5d_2",
    "conv5e", "conv6", "conv7", "output_conv_mv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("--json", required=True)
    ap.add_argument("--header", help="optionally emit a C++ header too")
    args = ap.parse_args()

    ir = analyze(args.vgf)
    consts = load_constants(args.vgf, ir)
    ops = ir["ops"]
    print(f"graph '{ir['graph_name']}': {len(ops)} ops, "
          f"{ir['constant_count']} constants, inputs={[i['tensor_id'] for i in ir['inputs']]}")

    by_index = {r["index"]: r for r in ops}

    # ---- pair each CONV2D with the RESCALE that consumes it --------------
    fused = {}
    for r in ops:
        if r["kind"] != "CONV2D":
            continue
        j = next((s for s in ops
                  if s["kind"] == "RESCALE" and s["input"]["kind"] == "node"
                  and s["input"]["index"] == r["index"]), None)
        fused[r["index"]] = j

    convs = [r for r in ops if r["kind"] == "CONV2D"]
    print(f"CONV2D={len(convs)}  paired rescales={sum(1 for v in fused.values() if v)}")

    layers = []
    for n, r in enumerate(convs):
        name = LAYER_ORDER[n] if n < len(LAYER_ORDER) else f"conv{n}"
        rs = fused[r["index"]]
        if rs is None:
            raise SystemExit(f"{name}: no RESCALE found")

        wgt = np.asarray(const_array(r["weight"], consts))
        bias = np.asarray(const_array(r["bias"], consts))
        izp_raw = r["input_zero_point"]
        izp = as_int8(izp_raw[0] if isinstance(izp_raw, list) else izp_raw)
        corr = (-np.int64(izp) * wgt.astype(np.int64).sum(axis=(1, 2, 3)))
        b64 = np.asarray(bias, dtype=np.int64).reshape(-1)
        if b64.size == 1:
            b64 = np.repeat(b64, wgt.shape[0])
        bc = (b64 + corr).astype(np.int32)

        mult = np.asarray(const_array(rs["multiplier"], consts), dtype=np.int64).reshape(-1)
        shv = np.asarray(const_array(rs["shift"], consts), dtype=np.int64).reshape(-1)
        ozp_raw = rs["output_zero_point"]["value"]
        ozp = as_int8(ozp_raw[0] if isinstance(ozp_raw, list) else ozp_raw)

        pad = r["pad"]
        stride = r["stride"]
        # `pad` is a 4-list for spatial convs but a bare int (0) for the 1x1 layers.
        if isinstance(pad, int):
            pt = pb = pl = pr = int(pad)
        else:
            pt, pb, pl, pr = (int(v) for v in pad)
        layers.append({
            "name": name, "opIndex": r["index"], "rescaleIndex": rs["index"],
            "weightShape": list(wgt.shape),
            "kernel": int(wgt.shape[1]), "inChannels": int(wgt.shape[3]),
            "outChannels": int(wgt.shape[0]),
            "padT": pt, "padB": pb, "padL": pl, "padR": pr,
            "strideH": int(stride[0]), "strideW": int(stride[1]),
            "dilation": r["dilation"],
            "inputZeroPoint": int(izp), "outputZeroPoint": int(ozp),
            "relu": bool(ozp == -128),
            "multiplier": [int(v) for v in mult], "shift": [int(v) for v in shv],
            "multiplierCount": int(mult.size),
            "rounding": rs["rounding"],
            # `bias` is the raw convolution bias; `biasCorrected` additionally folds
            # in -input_zero_point * sum(weights). The two are mutually exclusive
            # with the padding convention:
            #   pad with 0   -> use biasCorrected
            #   pad with izp -> use raw bias   (padding already supplies the term)
            # Mixing them double-counts izp*sum(w) and shifts every output.
            "bias": [int(v) for v in b64],
            "biasCorrected": [int(v) for v in bc],
        })
        print(f"  {name:<15} op{r['index']:<3} rs{rs['index']:<3} "
              f"w{list(wgt.shape)!s:<18} pad=[{pt},{pb},{pl},{pr}] "
              f"stride=[{stride[0]},{stride[1]}] izp={izp:<5} ozp={ozp:<5} "
              f"M={int(mult[0]):<12} S={int(shv[0]):<3} relu={ozp == -128}")

    # ---- non-conv ops -----------------------------------------------------
    others = []
    for r in ops:
        if r["kind"] in ("RESCALE", "CONV2D"):
            continue
        if r["kind"] == "RESIZE":
            others.append({"kind": "resize", "opIndex": r["index"], "mode": r["mode"],
                           "scale": r["scale"], "input": r["input"]})
        elif r["kind"] == "CONCAT":
            others.append({"kind": "concat", "opIndex": r["index"], "axis": r["axis"],
                           "inputs": r["inputs"]})
    print(f"\nnon-conv ops: {[o['kind'] + str(o['opIndex']) for o in others]}")

    manifest = {
        "schema": "nfru-v1-int8-baked/1",
        "source": {"vgf": os.path.basename(args.vgf), "graph": ir["graph_name"]},
        "io": {"inputTensorId": ir["inputs"][0]["tensor_id"],
               "inputShape": ir["inputs"][0]["shape"],
               "outputTensorId": ir["outputs"][0]["tensor_id"],
               "outputShape": ir["outputs"][0]["shape"]},
        "requantisation": {
            "formula": "y = clamp( ((acc + bc) * M) >> S ) + outZp, then clamp to int8",
            "note": "M and S are exact integers from the VGF. bc folds the weight bias "
                    "and the input-zero-point correction (-izp * sum(w)).",
            "reluMarker": "outputZeroPoint == -128 means the layer applies ReLU",
        },
        "layers": layers,
        "otherOps": others,
    }
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {args.json}")

    if args.header:
        with open(args.header, "w", encoding="utf-8") as fh:
            fh.write("// generated by bake_nfru.py - do not edit\n#pragma once\n\n")
            fh.write("namespace nfru_baked {\n\n")
            fh.write(f"static const int kLayerCount = {len(layers)};\n\n")
            fh.write("struct LayerDesc {\n"
                     "    const char* name;\n"
                     "    int kh, kw, cin, cout;\n"
                     "    int padT, padB, padL, padR;\n"
                     "    int strideH, strideW;\n"
                     "    int inZp, outZp;\n"
                     "    int relu;\n"
                     "    int multCount;\n"
                     "    const int* bc;\n"
                     "    const int* mult;\n"
                     "    const int* shift;\n"
                     "};\n\n")
            for l in layers:
                nm = l["name"]
                bc = ",".join(str(v) for v in l["biasCorrected"])
                fh.write(f"static const int {nm}_bc[{len(l['biasCorrected'])}] = {{{bc}}};\n")
                mm = ",".join(str(v) for v in l["multiplier"])
                fh.write(f"static const int {nm}_m[{len(l['multiplier'])}] = {{{mm}}};\n")
                ss = ",".join(str(v) for v in l["shift"])
                fh.write(f"static const int {nm}_s[{len(l['shift'])}] = {{{ss}}};\n\n")
            fh.write("}  // namespace nfru_baked\n")
        print(f"wrote {args.header}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
