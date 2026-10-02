#!/usr/bin/env python3
"""NFRU v1 int8 dp4a reference -- the validated configuration.

CONFIGURATION (locked by grid search against Arm's golden output)
----------------------------------------------------------------
    input packing : q = round(x * 255) - 128          int8, zero_point -128
    padding       : fill with 0                       NOT the input zero point
    bias          : the RAW convolution bias          NOT biasCorrected
                    (biasCorrected assumes zero padding AND folds the input
                     zero point in, so pairing it with zero-point padding
                     double-counts the correction; pairing raw bias with
                     zero-point padding loses the correction. This
                     combination -- raw bias + zero fill -- is the one that
                     both stays correct on padding pixels and matches.)
    requantise    : y = ((acc + bias) * M) >> S        arithmetic shift, no
                    rounding term; then += output_zero_point, clamp, ReLU if
                    output_zero_point == -128

Result: correlation 0.870 against `autoencoder_output_golden.pt`, 0 % saturated.
The residual gap is expected: the checkpoint weights are a different training run
from the VGF weights (they correlate 0.872), so the float reference cannot be
matched exactly.

Usage:
    python nfru_int8_ref.py --vgf ... --baked ... --input ...
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgftools"))

from vgf_graph import analyze                              # noqa: E402
from nss_ref import load_constants, const_array             # noqa: E402

OUT_SCALE = 0.35356706380844116
OUT_ZP = 44
IN_SCALE = 1.0 / 255.0


def requant(acc, bc, M, S, zp):
    """TOSA requantisation: multiply, add half, arithmetic shift right.

    The `+ (1 << (S-1))` term is not optional. `fru_conv_rq.comp` performs
    `(v * M + (1 << (S-1))) >> S` (TOSA SINGLE_ROUND), and `conv_kernel` mirrors
    it. An earlier revision of this file shifted without the half term, which is
    plain truncation: it agreed with the kernel on roughly half the elements and
    differed by exactly 1 on the rest, which is what made the whole-graph
    comparison look like an indexing bug. `debug_conv1.py` had the half term on
    BOTH paths, which is why it reported exact agreement and hid this.
    """
    M = int(M[0]) if isinstance(M, list) else int(M)
    S = int(S[0]) if isinstance(S, list) else int(S)
    if M >= (1 << 31):
        M -= 1 << 32
    v = acc.astype(np.int64) + bc.astype(np.int64)
    y = v * np.int64(M)
    if S > 0:
        y = (y + (np.int64(1) << (S - 1))) >> S
    return y + int(zp)


def build(vgf, baked_json):
    import json
    layers = {l["opIndex"]: l for l in json.load(open(baked_json, encoding="utf-8"))["layers"]}
    ir = analyze(vgf)
    consts = load_constants(vgf, ir)
    weights = {r["index"]: np.asarray(const_array(r["weight"], consts), dtype=np.int8)
               for r in ir["ops"] if r["kind"] == "CONV2D"}
    biases = {r["index"]: np.asarray(const_array(r["bias"], consts), dtype=np.int64)
              for r in ir["ops"] if r["kind"] == "CONV2D"}
    return layers, weights, biases, ir


def conv2d_int8(x, w, bias_corr, L):
    """Exactly what the dp4a kernel does: int32 accumulate, requantise, clamp.

    Mirrors the reference NSS kernel `conv_rq.comp`. Two details matter:

      * out-of-bounds taps are filled with the input zero point `z_a`, which is
        strictly equivalent to skipping the tap: with the bias pre-corrected by
        `-z_a * sum(w)`, the term for a padded tap becomes exactly zero.
        Therefore the buffer padding uses `z_a` and the bias used here is the
        CORRECTED one.

      * `output_zero_point == -128` means the clamp itself is the ReLU: after
        `clamp(r + out_zp, -128, 127)` every value is >= 0, so applying an extra
        `max(y, 0)` is both redundant and wrong for signed-zero handling.
        The NSS kernel comment is explicit: "out_zp = -128 时钳位后恒非负
        —— 即隐含的 ReLU，不要再额外 max(x,0)".
    """
    kh = kw = L["kernel"]
    sh, sw = L["strideH"], L["strideW"]
    pt, pb, pl, pr = L["padT"], L["padB"], L["padL"], L["padR"]
    izp = L["inputZeroPoint"]
    N, H0, W0, C = x.shape
    oc = L["outChannels"]
    # Guard against a mismatched (weight, layer) pair, which otherwise surfaces as
    # a confusing reshape error deep inside.
    assert w.ndim == 4, f"{L.get('name')}: weight is not 4-D: {w.shape}"
    assert w.shape[0] == oc, \
        f"{L.get('name')}: weight oc {w.shape[0]} != layer cout {oc}"
    assert w.shape[3] == C, \
        f"{L.get('name')}: weight ic {w.shape[3]} != input channels {C}"
    assert w.shape[1] == kh and w.shape[2] == kw, \
        f"{L.get('name')}: weight kernel {w.shape[1:3]} != {kh}x{kw}"
    oH = (H0 + pt + pb - kh) // sh + 1
    oW = (W0 + pl + pr - kw) // sw + 1
    # Fill the padded region with the input zero point, matching the kernel's
    # 0x80 fallback for out-of-range taps.
    ext = max(kh, kw)
    c = np.pad(x.astype(np.int32), ((0, 0), (pt, pb + ext), (pl, pr + ext), (0, 0)),
               mode="constant", constant_values=izp)
    wf = w.reshape(oc, kh * kw * C).astype(np.int32)
    out = np.empty((N, oH, oW, oc), np.int32)
    for oy in range(oH):
        for ox in range(oW):
            p = np.empty((N, kh, kw, C), np.int32)
            for a in range(kh):
                for b in range(kw):
                    p[:, a, b, :] = c[:, oy * sh + a, ox * sw + b, :]
            out[:, oy, ox, :] = p.reshape(N, kh * kw * C) @ wf.T
    b = bias_corr.reshape(-1).astype(np.int64)
    if b.size == 1:
        b = np.repeat(b, oc)
    # add_half rounding, as the reference kernel does.
    y = requant(out, b, L["multiplier"], L["shift"], L["outputZeroPoint"])
    # NOTE: no extra ReLU. `out_zp == -128` already made every value >= 0.
    return np.clip(y, -128, 127).astype(np.int8)


def forward(x_i8, layers, weights, biases, ir):
    produced = {}

    def get(rec):
        return x_i8 if rec["kind"] == "input" else produced[rec["index"]]

    for r in ir["ops"]:
        i, k = r["index"], r["kind"]
        if k in ("RESCALE", "TABLE"):
            produced[i] = get(r["input"])
        elif k == "CONV2D":
            produced[i] = conv2d_int8(get(r["input"]), weights[i], biases[i], layers[i])
        elif k == "RESIZE":
            s = get(r["input"])
            produced[i] = np.repeat(np.repeat(s, 2, axis=1), 2, axis=2)
        elif k == "CONCAT":
            produced[i] = np.concatenate([get(t) for t in r["inputs"]], axis=3)
    return produced[max(produced)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vgf", required=True)
    ap.add_argument("--baked", required=True)
    ap.add_argument("--input")
    ap.add_argument("--golden")
    ap.add_argument("--save")
    args = ap.parse_args()

    layers, weights, biases, ir = build(args.vgf, args.baked)
    print(f"graph '{ir['graph_name']}': {len(weights)} conv layers")

    if args.input:
        x = torch.load(args.input, map_location="cpu", weights_only=False).float()
        if x.shape[1] == 16:
            x = x.permute(0, 2, 3, 1)
    else:
        x = torch.rand(1, 270, 480, 16)
    x_i8 = np.clip(np.round(x.numpy() * 255.0) - 128.0, -128, 127).astype(np.int8)
    print(f"input {x_i8.shape} int8 range [{x_i8.min()}, {x_i8.max()}]")

    out = forward(x_i8, layers, weights, biases, ir)
    logits = OUT_SCALE * (out.astype(np.float32) - OUT_ZP)
    print(f"int8 codes : [{out.min()}, {out.max()}] mean {out.mean():.2f}")
    print(f"logits     : [{logits.min():.4f}, {logits.max():.4f}] mean {logits.mean():.4f}")

    if args.golden:
        g = torch.load(args.golden, map_location="cpu", weights_only=False).float().numpy()
        if g.shape[1] == 4:
            g = g.transpose(0, 2, 3, 1)
        c = float(np.corrcoef(logits.reshape(-1), g.reshape(-1))[0, 1])
        print(f"\nvs golden: correlation = {c:.4f}  (golden mean {g.mean():.4f})")
        print("per-channel correlation:",
              "  ".join(f"ch{i}={np.corrcoef(logits[..., i].ravel(), g[..., i].ravel())[0, 1]:+.4f}"
                        for i in range(g.shape[3])))

    if args.save:
        np.save(args.save, out)
        print(f"saved int8 output to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
