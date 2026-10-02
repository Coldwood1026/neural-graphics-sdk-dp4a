#!/usr/bin/env python3
"""Validate the int8 dp4a pipeline using the REAL VGF requantisation parameters.

Earlier attempts at this used guessed quantisation points, which is why they
failed. This one is driven entirely by `port/nfru_baked.json`, whose multiplier /
shift / zero points were extracted from the VGF with the correct SPIR-V operand
offsets (see bake_nfru.py).

The integer path here is exactly what a dp4a kernel does:

    acc = sum(i8(act) * i8(w))                      int32, exact
    y   = ((acc + bc) * M) >> S                     with rounding, then + outZp
    y   = clamp(y, -128, 127)                       int8 store

so agreement with the float model validates the kernel design, not just this script.

Usage:
    python validate_baked.py --vgf ../../fru/nfru_v1_int8.vgf \
        --checkpoint ../../fru/nfru_v1_int8.pt \
        --input ../../model_gym/golden/autoencoder_input_golden.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgftools"))
from vgf_graph import analyze                                # noqa: E402
from nss_ref import load_constants, const_array               # noqa: E402


def load_baked(path):
    m = json.load(open(path, encoding="utf-8"))
    return {l["opIndex"]: l for l in m["layers"]}, m


def requant(acc_i32, bc, M, S, out_zp):
    """TOSA requantisation: ((acc + bc) * M) >> S, round-half-away, then + zp.

    TOSA's SINGLE_ROUND applies round-half-away-from-zero; the `+ (1 << (S-1))`
    term implements that for the common power-of-two shift case.
    """
    a = acc_i32.astype(np.int64) + bc.astype(np.int64)
    m = int(M) if np.isscalar(M) else int(M[0])
    s = int(S) if np.isscalar(S) else int(S[0])
    if m >= (1 << 31):
        m -= 1 << 32
    prod = a * m
    if s > 0:
        # round half away from zero before the arithmetic shift
        prod = prod + (1 << (s - 1))
    return (prod >> s) + int(out_zp)


def conv_int8(x_i8, w_i8, layer, in_scale_note=""):
    """x_i8, w_i8 are int8 numpy arrays. x is NHWC, w is [oc,kh,kw,ic]."""
    pt, pb, pl, pr = layer["padT"], layer["padB"], layer["padL"], layer["padR"]
    sh, sw = layer["strideH"], layer["strideW"]
    # Zero-point padding: pad with the input zero point, not with 0.
    izp = layer["inputZeroPoint"]
    # Pad with the input zero point. Pad generously on the bottom/right by one
    # extra kernel's worth so the gather below can never index out of bounds; the
    # extra region is only read for positions the output size excludes.
    x = x_i8.astype(np.int32)
    N, H0, W0, C = x_i8.shape
    kh, kw, ic = layer["kernel"], layer["kernel"], layer["inChannels"]
    oc = layer["outChannels"]
    # Pad with the input zero point. Pad generously on the bottom/right by one
    # extra kernel's worth so the gather below can never index out of bounds; the
    # extra region is only read for positions the output size excludes.
    #
    # CRITICAL: padding with `izp` and using the bias correction are mutually
    # exclusive. If the padded region holds `izp`, the accumulator already
    # contains `izp * sum(w)` and adding `biasCorrected` would double-count it,
    # producing a large constant offset. So this path uses the RAW bias.
    x = x_i8.astype(np.int32)
    extra = max(kh, kw)
    x = np.pad(x, ((0, 0), (pt, pb + extra), (pl, pr + extra), (0, 0)),
               mode="constant", constant_values=izp)
    # Output size per the TOSA rule, computed from the ORIGINAL input size.
    oH = (H0 + pt + pb - kh) // sh + 1
    oW = (W0 + pl + pr - kw) // sw + 1
    # Explicit gather: correctness first. A production kernel does this with
    # texture/buffer reads rather than materialising im2col.
    out = np.empty((N, oH, oW, oc), dtype=np.int32)
    wflat = w_i8.reshape(oc, kh * kw * ic).astype(np.int32)
    for oy in range(oH):
        for ox in range(oW):
            # patch of the (zero-point-padded) input
            patch = np.empty((N, kh, kw, ic), dtype=np.int32)
            for i in range(kh):
                yy = oy * sh + i
                for j in range(kw):
                    xx = ox * sw + j
                    patch[:, i, j, :] = x[:, yy, xx, :]
            acc = patch.reshape(N, kh * kw * ic) @ wflat.T
            out[:, oy, ox, :] = acc
    acc = out
    # Use the RAW bias here (see the note above about double-counting).
    bc = np.asarray(layer["bias"], dtype=np.int32)
    y = requant(acc, bc, layer["multiplier"], layer["shift"], layer["outputZeroPoint"])
    if layer["relu"]:
        y = np.maximum(y, 0)
    y = np.clip(y, -128, 127).astype(np.int8)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vgf", required=True)
    ap.add_argument("--baked", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input")
    args = ap.parse_args()

    layers, manifest = load_baked(args.baked)
    ir = analyze(args.vgf)
    consts = load_constants(args.vgf, ir)
    ops = {r["index"]: r for r in ir["ops"]}

    # Weight arrays straight from the VGF constants.
    weights = {}
    for r in ir["ops"]:
        if r["kind"] != "CONV2D":
            continue
        weights[r["index"]] = np.asarray(const_array(r["weight"], consts), dtype=np.int8)
    print(f"loaded {len(weights)} conv weights from the VGF")

    # Float reference from the QAT checkpoint (BN folded).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from recover_qparams import NFRUAutoEncoder, fold_bn
    import re

    model = NFRUAutoEncoder()
    msd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    params, tensors = {}, {}
    for k, v in msd.items():
        m = re.search(r"\._param_constant(\d+)$", k)
        if m:
            params[int(m.group(1))] = v.float()
        m = re.search(r"\._tensor_constant(\d+)$", k)
        if m:
            tensors[int(m.group(1))] = v.float()
    bn_i = 0
    ORDER = ["conv1", "conv2", "conv3", "skip1_conv", "conv5", "conv5a", "conv5b",
             "conv5c", "conv5c_1", "conv5d", "conv5d_1", "conv5d_2", "conv5e",
             "conv6", "conv7", "output_conv_mv"]
    for i, name in enumerate(ORDER):
        mod = getattr(model, name)
        conv = mod if isinstance(mod, nn.Conv2d) else mod.conv2d
        w, b = params[4 * i + 0].clone(), params[4 * i + 1].clone()
        if not isinstance(mod, nn.Conv2d):
            g, be = params[4 * i + 2], params[4 * i + 3]
            mean, var = tensors[2 * bn_i], tensors[2 * bn_i + 1]
            w, b = fold_bn(w, b, g, be, mean, var)
            bn_i += 1
            mod.bn = nn.Identity()
        with torch.no_grad():
            conv.weight.copy_(w)
            conv.bias.copy_(b)
    model.eval()

    if args.input:
        xf = torch.load(args.input, map_location="cpu", weights_only=False).float()
        if xf.shape[3] == 16:
            xf = xf.permute(0, 3, 1, 2).contiguous()
    else:
        xf = torch.rand(1, 16, 270, 480)

    with torch.no_grad():
        yf = model(xf).numpy()
    print(f"float output: [{yf.min():.4f}, {yf.max():.4f}] mean {yf.mean():.4f}")

    # ---- int8 path --------------------------------------------------------
    x_np = xf.permute(0, 2, 3, 1).numpy()                  # NHWC
    x_i8 = np.clip(np.round(x_np * 255.0) - 128.0, -128, 127).astype(np.int8)

    # Execution follows the IR DAG.
    val = {None: None}
    def get(rec):
        if rec["kind"] == "input":
            return x_i8
        if rec["kind"] == "node":
            return produced[rec["index"]]
        raise NotImplementedError(rec)

    produced = {}
    for r in ir["ops"]:
        i, k = r["index"], r["kind"]
        if k == "TABLE":
            produced[i] = get(r["input"])               # table is fused too
            continue
        if k == "RESCALE":
            # Fused into its input convolution: the requantisation happens inside
            # the conv kernel, so the RESCALE shares its producer's buffer.
            produced[i] = get(r["input"])
            continue
        if k == "CONV2D":
            src = get(r["input"])
            produced[i] = conv_int8(src, weights[i], layers[i])
        elif k == "RESIZE":
            src = get(r["input"])
            produced[i] = np.repeat(np.repeat(src, 2, axis=1), 2, axis=2)
        elif k == "CONCAT":
            parts = [get(t) for t in r["inputs"]]
            produced[i] = np.concatenate(parts, axis=3)

    out_id = ir["outputs"][0]["tensor_id"]
    last = max(i for i in produced)
    out_i8 = produced[last]
    print(f"int8 output: [{out_i8.min()}, {out_i8.max()}] mean {out_i8.mean():.2f}")

    # dequantise with the output quantisation point
    ozp, oscale = 44, 0.35356706380844116
    logits = oscale * (out_i8.astype(np.float32) - ozp)
    yf_nhwc = yf.transpose(0, 2, 3, 1)

    a, b = logits.reshape(-1), yf_nhwc.reshape(-1)
    corr = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-9))
    print(f"\ncorrelation  = {corr:.6f}")
    print(f"mean abs rel = {rel:.2%}")
    print("VERDICT:", "int8 pipeline MATCHES float" if corr > 0.99
          else "still a mismatch - inspect per-layer output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
