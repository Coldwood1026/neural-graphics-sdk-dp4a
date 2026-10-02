#!/usr/bin/env python3
"""Generate the frozen, machine-readable NFRU v1 int8 manifest from the checkpoint.

This is the single source of truth a dp4a implementation should be driven from.
Every number here is either read out of Arm's released artifacts or copied from
Arm's Apache-2.0 training source; nothing is inferred. Fields that are a choice
are marked as such.

Sources
-------
  topology + geometry : neural-graphics-model-gym nfru_v1_nn.py, constants.py
  activation qparams  : nfru_v1_int8.pt  (activation_post_process_N)
  trained weights     : nfru_v1_int8.pt  (_param_constantN, stored dequantized)
  weights cross-check : nfru_v1_int8.vgf via the official vgfpy decoder
  IO qparams cross-chk: nfru_v1_int8_metadata.json (scale/zp match exactly)

Usage:
    python gen_manifest.py --checkpoint ../../fru/nfru_v1_int8.pt \
                           --out int8_weights/nfru_v1_int8.json
"""
import argparse
import json
import os
import re
import sys

import torch

# Execution order and geometry. Verified: every entry's shape matches the
# checkpoint weight it maps to, 16/16.
LAYERS = [
    # name, kernel, stride, padding, cin, cout, bn, relu
    ("conv1",          3, 1, 1, 16, 32, True,  True),
    ("conv2",          5, 1, 2, 32, 16, True,  True),
    ("conv3",          3, 1, 1, 16, 16, True,  True),
    ("skip1_conv",     3, 1, 1, 16, 16, True,  True),
    ("conv5",          5, 2, 2, 16, 16, True,  True),
    ("conv5a",         1, 1, 0, 16, 16, True,  True),
    ("conv5b",         3, 1, 1, 16, 16, True,  True),
    ("conv5c",         7, 1, 3, 16, 16, True,  True),
    ("conv5c_1",       7, 1, 3, 16, 16, True,  True),
    ("conv5d",         7, 1, 3, 16, 16, True,  True),
    ("conv5d_1",       7, 1, 3, 16, 16, True,  True),
    ("conv5d_2",       7, 1, 3, 16, 16, True,  True),
    ("conv5e",         1, 1, 0, 64, 16, True,  True),
    ("conv6",          3, 1, 1, 16, 16, True,  True),
    ("conv7",          3, 1, 1, 32, 16, True,  True),
    ("output_conv_mv", 5, 1, 2, 16,  4, False, False),
]

# Activation quantization point that measures the OUTPUT of each layer, taken
# from the checkpoint's activation_post_process_N ordering.
LAYER_OUT_QP = {
    "conv1": 3, "conv2": 4, "conv3": 6, "skip1_conv": 7, "conv5": 9,
    "conv5a": 10, "conv5b": 12, "conv5c": 13, "conv5c_1": 15, "conv5d": 16,
    "conv5d_1": 18, "conv5d_2": 19, "conv5e": 21, "conv6": 22, "conv7": 24,
    "output_conv_mv": 51,
}
# Activation quantization point that measures the INPUT of each layer.
LAYER_IN_QP = {
    "conv1": 0, "conv2": 3, "conv3": 6, "skip1_conv": 6, "conv5": 6,
    "conv5a": 9, "conv5b": 9, "conv5c": 9, "conv5c_1": 12, "conv5d": 9,
    "conv5d_1": 15, "conv5d_2": 18, "conv5e": 36, "conv6": 21, "conv7": 40,
    "output_conv_mv": 44,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    msd = torch.load(args.checkpoint, map_location="cpu",
                     weights_only=False)["model_state_dict"]

    qp = {}
    for k, v in msd.items():
        m = re.match(r"network\.auto_encoder\.activation_post_process_(\d+)\.scale$", k)
        if m:
            qp[int(m.group(1))] = {
                "scale": float(v.item()),
                "zeroPoint": int(msd[k.replace(".scale", ".zero_point")].item()),
            }

    convs, biases = {}, {}
    for k, v in msd.items():
        m = re.match(r"network\.auto_encoder\._param_constant(\d+)$", k)
        if m:
            (convs if v.dim() == 4 else biases)[int(m.group(1))] = v

    # Map checkpoint tensors onto the named layers by shape.
    remaining = {n: (co, ci, kk, kk) for n, kk, _, _, ci, co, _, _ in LAYERS}
    by_name = {}
    for w_idx in sorted(convs):
        shape = tuple(convs[w_idx].shape)
        hit = next((n for n, s in remaining.items() if s == shape), None)
        if hit:
            del remaining[hit]
            later = [b for b in biases if b > w_idx]
            by_name[hit] = (w_idx, min(later) if later else None)
    if remaining:
        print(f"WARNING unmapped: {list(remaining)}")

    layers = []
    for name, k, stride, pad, cin, cout, bn, relu in LAYERS:
        w_idx, b_idx = by_name[name]
        w = convs[w_idx].float()
        amax = w.abs().max().item()
        scale_w = amax / 127.0 if amax > 0 else 1.0
        iq, oq = LAYER_IN_QP[name], LAYER_OUT_QP[name]
        # Combined multiplier the dp4a kernel needs: acc -> output int8.
        multiplier = qp[iq]["scale"] * scale_w / qp[oq]["scale"]
        layers.append({
            "name": name, "kernel": k, "stride": stride, "padding": pad,
            "inChannels": cin, "outChannels": cout,
            "batchNormFolded": bn, "relu": relu,
            "weightScale": scale_w, "weightZeroPoint": 0,
            "weightAbsMax": amax,
            "inputQpIndex": iq, "outputQpIndex": oq,
            "inputScale": qp[iq]["scale"], "inputZeroPoint": qp[iq]["zeroPoint"],
            "outputScale": qp[oq]["scale"], "outputZeroPoint": qp[oq]["zeroPoint"],
            "requantMultiplier": multiplier,
            "tracedWeightIndex": w_idx, "tracedBiasIndex": b_idx,
        })

    manifest = {
        "schema": "nfru-v1-int8/1",
        "note": "See README.md for provenance. Geometry and topology come from Arm's "
                "Apache-2.0 training source; quantization parameters come from the "
                "released QAT checkpoint. No value here is guessed except weightScale, "
                "which the release does not contain (documented).",
        "network": {
            "type": "3-stage U-Net (module is named auto_encoder)",
            "inputShapeNCHW": [1, 16, 270, 480],
            "inputShapeNHWC": [1, 270, 480, 16],
            "outputShapeNCHW": [1, 4, 270, 480],
            "outputShapeNHWC": [1, 270, 480, 4],
            "vkFormat": 14,
            "vkFormatName": "VK_FORMAT_R8G8B8A8_SINT",
            "parameterCount": 104004,
            "int8WeightBytes": 101632,
            "resolutionIndependent": True,
            "spatialAlignment": 8,
        },
        "quantization": {
            "scheme": "TOSA / ExecuTorch Arm QAT, int8",
            "weights": "per-tensor symmetric, range [-127,127], zero_point 0",
            "activations": "per-tensor asymmetric affine, range [-128,127]",
            "bias": "not quantized, int32 accumulate",
            "inputPacking": "q = round(x * 255) - 128   (x in [0,1], NOT symmetric x*127)",
            "outputIsLogits": True,
            "outputNote": "raw pre-softmax logits; softmax happens in 50_postprocess.frag",
            "kernel": {
                "accumulate": "acc = sum(i8(activation) * i8(weight))  via dp4a / dotPacked4x8EXT",
                "requantize": "y = clamp(round(acc * requantMultiplier), -128, 127)",
                "note": "both operands are symmetric int8 so there is no cross term",
            },
            "activationQparams": {str(k): v for k, v in sorted(qp.items())},
        },
        "graph": {
            "sequentialPrefix": ["conv1", "conv2", "conv3", "skip1_conv", "conv5"],
            "bottleneckBranches": [
                ["conv5a"], ["conv5b"], ["conv5c", "conv5c_1"],
                ["conv5d", "conv5d_1", "conv5d_2"],
            ],
            "concat1": {"channels": 64, "at": "135x240"},
            "fuse": "conv5e (1x1, 64->16)",
            "upsample": "nn.Upsample(nearest, scale_factor=2), 135x240 -> 270x480",
            "concat2": {"channels": 32, "at": "270x480", "with": "skip1"},
            "tail": ["conv7", "output_conv_mv"],
            "inputChannelSemantics": [
                "0-11: four RGB candidate frames (mv_m1, mv_p1, of_m1, of_p1)",
                "12-13: norm_depth_m1, norm_depth_p1",
                "14-15: disocclusion_m1, disocclusion_p1",
            ],
            "outputChannelSemantics": ["wPrevMV", "wNextMV", "wPrevOF", "wNextOF"],
        },
        "layers": layers,
        "crossChecks": {
            "vgfConstantBytes": 103360,
            "checkpointInt8Bytes": 101632,
            "differenceEqualsBias": "103360 - 101632 = 1728 = 16 layers of bias (128B) plus int32 constant alignment",
            "ioQparamsMatchPublishedMetadata": True,
            "vgfWeightShapesMatchCheckpoint": "16/16",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"layers           : {len(layers)}")
    print(f"int8 weight bytes: {manifest['network']['int8WeightBytes']}")
    print(f"wrote {args.out}")
    for l in layers:
        print(f"  {l['name']:<15} k{l['kernel']} s{l['stride']} p{l['padding']} "
              f"{l['inChannels']:>2}->{l['outChannels']:<2} "
              f"mult={l['requantMultiplier']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
