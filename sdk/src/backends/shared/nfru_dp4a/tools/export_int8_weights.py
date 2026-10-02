#!/usr/bin/env python3
"""Export the NFRU v1 int8 network for a dp4a port, with explicit quant algebra.

PROVENANCE - what is authoritative and what is a choice
-------------------------------------------------------
Authoritative (read straight out of the released QAT checkpoint):
  * the 16-layer topology and every kernel geometry
  * activation scale / zero_point for all 36 quantization points
  * the trained weights themselves (stored dequantized as float32)

NOT in the release (so it is an explicit choice here):
  * the per-layer *weight* scale. The released checkpoint stores weights already
    dequantized (`_param_constantN` are float32 with no grid structure), and the
    published `config.json` is empty, so the int8 weight grid the VGF uses is not
    recoverable from the release. We pick the standard symmetric scheme
    `scale_w = max|w| / 127`.

Consequence for the port: because both sides of every dot product are int8 with
zero_point 0, the GPU kernel does
      acc = sum(i8(act) * i8(w))            // exact int32 via dp4a
      y   = acc * (scale_a * scale_w / scale_out)
so each layer needs the single combined multiplier
      m = scale_a * scale_w / scale_out
plus the output zero point. Those are emitted in the manifest.

Usage:
    python export_int8_weights.py --checkpoint ../../fru/nfru_v1_int8.pt --out <dir>
"""
import argparse
import json
import os
import re
import sys

import torch

# Execution order, geometry from ng_model_gym nfru_v1_nn.py + constants.py.
# (name, kernel, stride, padding, cin, cout, activation_qparam_index)
LAYER_SPEC = [
    ("conv1",          3, 1, 1, 16, 32),
    ("conv2",          5, 1, 2, 32, 16),
    ("conv3",          3, 1, 1, 16, 16),
    ("skip1_conv",     3, 1, 1, 16, 16),
    ("conv5",          5, 2, 2, 16, 16),
    ("conv5a",         1, 1, 0, 16, 16),
    ("conv5b",         3, 1, 1, 16, 16),
    ("conv5c",         7, 1, 3, 16, 16),
    ("conv5c_1",       7, 1, 3, 16, 16),
    ("conv5d",         7, 1, 3, 16, 16),
    ("conv5d_1",       7, 1, 3, 16, 16),
    ("conv5d_2",       7, 1, 3, 16, 16),
    ("conv5e",         1, 1, 0, 64, 16),
    ("conv6",          3, 1, 1, 16, 16),
    ("conv7",          3, 1, 1, 32, 16),
    ("output_conv_mv", 5, 1, 2, 16, 4),
]
CONV_SHAPES = {n: (co, ci, k, k) for n, k, _, _, ci, co in LAYER_SPEC}
GEOM = {n: {"kernel": k, "stride": s, "padding": p, "inChannels": ci, "outChannels": co}
        for n, k, s, p, ci, co in LAYER_SPEC}


def load_qparams(msd):
    qp = {}
    for k in msd:
        m = re.match(r"network\.auto_encoder\.activation_post_process_(\d+)\.scale$", k)
        if m:
            idx = int(m.group(1))
            zp_key = k.replace(".scale", ".zero_point")
            qp[idx] = {"scale": float(msd[k].item()),
                       "zero_point": int(msd[zp_key].item())}
    return qp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    msd = sd["model_state_dict"]

    # _param_constantN with 4 dims are conv weights, 1 dim are biases.
    convs, biases = {}, {}
    for k in msd:
        m = re.match(r"network\.auto_encoder\._param_constant(\d+)$", k)
        if not m:
            continue
        idx, t = int(m.group(1)), msd[k]
        (convs if t.dim() == 4 else biases)[idx] = t

    qparams = load_qparams(msd)

    # Pair each conv with the bias that follows it in the enumeration.
    ordered = []
    for w_idx in sorted(convs):
        later = [b for b in biases if b > w_idx]
        ordered.append((w_idx, min(later) if later else None, convs[w_idx]))

    # Map to layer names by shape.
    remaining = dict(CONV_SHAPES)
    named = []
    for w_idx, b_idx, w in ordered:
        hit = next((n for n, s in remaining.items() if s == tuple(w.shape)), None)
        if hit:
            del remaining[hit]
        named.append((hit, w_idx, b_idx, w))
    if remaining:
        print(f"WARNING unmatched specs: {list(remaining)}")

    os.makedirs(args.out, exist_ok=True)
    layers = []

    print("== int8 weight export (symmetric, scale_w = max|w|/127) ==")
    for name, w_idx, b_idx, w in named:
        if name is None:
            continue
        amax = w.abs().max().item()
        scale_w = amax / 127.0 if amax > 0 else 1.0
        q = torch.clamp(torch.round(w / scale_w), -127, 127).to(torch.int8)

        # [CO,CI,KH,KW] -> [KH,KW,CI,CO] so the IC axis is contiguous for dp4a.
        q_kkio = q.permute(2, 3, 1, 0).contiguous()
        wfile = f"{name}.int8"
        q_kkio.numpy().tofile(os.path.join(args.out, wfile))

        bias = biases[b_idx] if b_idx is not None else torch.zeros(w.shape[0])
        bfile = f"{name}.bias.f32"
        bias.float().numpy().tofile(os.path.join(args.out, bfile))

        # Dequantized int8 reconstruction error: how much quality the chosen
        # weight grid costs relative to the released float weights.
        deq = q.float() * scale_w
        rel_err = ((deq - w).abs().max().item() / amax) if amax else 0.0

        layers.append({**GEOM[name], "name": name,
                       "tracedWeightIndex": w_idx, "tracedBiasIndex": b_idx,
                       "weightFile": wfile, "biasFile": bfile,
                       "weightScale": scale_w, "weightZeroPoint": 0,
                       "weightBytes": int(q_kkio.numel()),
                       "maxRelDequantError": rel_err})

        print(f"  {name:<15} [{w_idx:>2}] {str(tuple(w.shape)):<18} -> "
              f"{str(tuple(q_kkio.shape)):<18} scale_w={scale_w:.8g} relErr={rel_err:.3e}")

    # Graph-level IO qparams: index 0 is the input, 51 the output.
    q_in, q_out = qparams.get(0), qparams.get(51)

    manifest = {
        "source": {"checkpoint": os.path.basename(args.checkpoint),
                   "quantization": "TOSA / ExecuTorch Arm QAT, int8 symmetric"},
        "provenance": {
            "authoritative": ["layer topology and kernel geometry",
                              "activation scale/zero_point for all quantization points",
                              "trained weights (stored dequantized as float32)"],
            "chosen": ["per-layer weight scale = max|w|/127 (not present in the "
                       "release; the published config.json is empty)"],
        },
        "algebra": {
            "accumulate": "acc = sum(i8(activation) * i8(weight)), computed with dp4a",
            "requantize": "y = acc * multiplier, then clamp to int8 output range",
            "multiplier": "scale_activation * scale_weight / scale_output_of_this_layer",
            "note": "both operands have zero_point 0, so there is no cross term",
        },
        "io": {"input": q_in, "output": q_out,
               "inputShapeNHWC": [1, -1, -1, 16], "outputShapeNHWC": [1, -1, -1, 4]},
        "graph": {
            "convs": ["conv1", "conv2", "conv3", "skip1_conv", "conv5"],
            "bottleneckBranches": [["conv5a"], ["conv5b"],
                                   ["conv5c", "conv5c_1"],
                                   ["conv5d", "conv5d_1", "conv5d_2"]],
            "concat1Channels": 64, "concat2Channels": 32,
            "upsample": "nearest x2 after conv5e",
            "activation": "ReLU (zero_point -128) or none (zero_point 0)",
        },
        "activationQparams": {str(k): v for k, v in sorted(qparams.items())},
        "layers": layers,
        "totalWeightBytes": sum(l["weightBytes"] for l in layers),
    }
    with open(os.path.join(args.out, "int8_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\nlayers exported      : {len(layers)}")
    print(f"total int8 weight B  : {manifest['totalWeightBytes']}")
    print(f"input  qparam        : scale={q_in['scale']:.8g} zp={q_in['zero_point']}")
    print(f"output qparam        : scale={q_out['scale']:.8g} zp={q_out['zero_point']}")
    print(f"wrote {os.path.join(args.out, 'int8_manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
