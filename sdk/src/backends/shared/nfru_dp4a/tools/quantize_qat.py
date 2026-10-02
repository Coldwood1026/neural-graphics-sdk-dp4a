#!/usr/bin/env python3
"""Quantise NFRU v1 to int8 from the QAT checkpoint `fru_v1_int8.pt`, for the dp4a
backend.

Why this source
---------------
`fru_v1_int8.pt` carries the training-time quantisation observers, so the activation
scales and zero points are READ, not guessed:

    network.auto_encoder.activation_post_process_<i>.scale / .zero_point

The activation chain is exactly:

    idx  0   scale 0.00392157  zp -128    network input  ( == 1/255 )
    idx  3   scale 0.08535242  zp -128    conv1 output
    idx  9   scale 0.07379657  zp -128    conv2 output
    idx 18   scale 0.04839480  zp -128    conv3 output
    ...
    idx 51   scale 0.35356706  zp   44    output logits

and every layer's input scale equals the previous layer's output scale, so the chain is
self-consistent. Earlier attempts calibrated scales by running the float model and got
roughly half the true range, which saturated whole layers (conv2 put 49.5% of its
outputs at +127) and destroyed everything downstream.

Weights: `_param_constant{4i}` is the convolution weight with batchnorm ALREADY folded
in (max|w|/127 matches the observer scale to within 1%; the un-folded weight differs by
4.7x). So the weight scale is max|w|/127 and the bias is the folded
`(conv_bias - mean) * gamma / sqrt(var + eps) + beta`.

Outputs the same four buffers the dp4a shader already consumes, so the kernel and the
host are unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (name, kernel, stride, pad, relu) in execution order.
GEOM = [
    ("conv1",         3, 1, 1, True),
    ("conv2",         5, 1, 2, True),
    ("conv3",         3, 1, 1, True),
    ("skip1_conv",    3, 1, 1, True),
    ("conv5",         5, 2, 2, True),
    ("conv5a",        1, 1, 0, True),
    ("conv5b",        3, 1, 1, True),
    ("conv5c",        7, 1, 3, True),
    ("conv5c_1",      7, 1, 3, True),
    ("conv5d",        7, 1, 3, True),
    ("conv5d_1",      7, 1, 3, True),
    ("conv5d_2",      7, 1, 3, True),
    ("conv5e",        1, 1, 0, True),
    ("conv6",         3, 1, 1, True),
    ("conv7",         3, 1, 1, True),
    ("output_conv_mv", 5, 1, 2, False),
]

# Activation observer indices, in execution order:
#   first is the graph input, then one output observer per layer.
ACT_IDX = [0, 3, 9, 18, 21, 24, 27, 30, 33, 36, 40, 44, 48, 12, 15, 6, 51]

IN_ZP = -128


def tosa_multiplier(real_mult):
    """real_mult -> M / 2^S with M < 2^31, the TOSA RESCALE form."""
    if real_mult <= 0:
        return 0, 0
    e = int(np.floor(np.log2(real_mult)))
    m = real_mult / (2.0 ** e)
    while m >= 1.0:
        m /= 2.0
        e += 1
    while m < 0.5:
        m *= 2.0
        e -= 1
    M = int(round(m * (1 << 30)))
    S = 30 - e
    if M >= (1 << 31):
        M >>= 1
        S += 1
    return M, max(S, 0)


def quantise_symmetric(w):
    """Per-tensor symmetric int8, zero_point 0: scale = max|w|/127."""
    mx = float(np.abs(w).max())
    if mx == 0.0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = mx / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def pack_weights(q_okki):
    """[oc,kh,kw,ic] int8 -> uint32[oc * kh*kw*(ic/4)], ic4 innermost."""
    oc, kh, kw, ic = q_okki.shape
    in_c4 = ic // 4
    return np.frombuffer(np.ascontiguousarray(q_okki.reshape(oc, kh, kw, in_c4, 4)).tobytes(),
                         dtype="<u4").reshape(oc, kh * kw * in_c4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fru_v1_int8.pt (QAT checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden-input")
    ap.add_argument("--golden-output")
    args = ap.parse_args()

    import torch
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    P = "network.auto_encoder."

    def act(i):
        return (sd[f"{P}activation_post_process_{i}.scale"].item(),
                int(sd[f"{P}activation_post_process_{i}.zero_point"].item()))

    # Fold batchnorm into weight and bias for each layer.
    folded = {}
    for li, (name, *_) in enumerate(GEOM):
        w = sd[f"{P}_param_constant{4*li}"].float().numpy()          # [oc,ic,kh,kw]
        cb = sd[f"{P}_param_constant{4*li+1}"].float().numpy()       # conv bias
        bn_w = sd.get(f"{P}_param_constant{4*li+2}")
        bn_b = sd.get(f"{P}_param_constant{4*li+3}")
        bn_m = sd.get(f"{P}_tensor_constant{2*li}")
        bn_v = sd.get(f"{P}_tensor_constant{2*li+1}")
        if bn_w is not None and bn_m is not None:
            g = bn_w.float().numpy()
            be = bn_b.float().numpy()
            mu = bn_m.float().numpy()
            var = bn_v.float().numpy()
            s = g / np.sqrt(var + 1e-5)
            w = w * s.reshape(-1, 1, 1, 1)
            cb = (cb - mu) * s + be
        folded[name] = (w, cb)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"source": os.path.basename(args.ckpt),
                "quantiser": "QAT observers + per-tensor symmetric int8 weights",
                "inputScale": act(ACT_IDX[0])[0], "inputZeroPoint": act(ACT_IDX[0])[1],
                "layers": []}

    print(f"{'layer':<15} {'wscale':>11} {'in_scale':>11} {'out_scale':>11} {'out_zp':>7} "
          f"{'real mult':>11} {'M':>11} {'S':>3}")
    params = {}
    for li, (name, k, s, p, relu) in enumerate(GEOM):
        in_scale, in_zp = act(ACT_IDX[li])
        out_scale, out_zp = act(ACT_IDX[li + 1])
        w, b = folded[name]
        wt = np.transpose(w, (0, 2, 3, 1))          # [oc,ic,kh,kw] -> [oc,kh,kw,ic]
        q, wscale = quantise_symmetric(wt)
        real_mult = (in_scale * wscale) / out_scale
        M, S = tosa_multiplier(real_mult)

        # The bias lives in accumulator units. Out-of-range taps are filled with
        # 0x80808080, i.e. four int8 of -128, which is exactly this layer's input zero
        # point, so those taps ALREADY contribute z_a*sum(w) to the accumulator. The
        # bias must therefore carry the matching correction
        #     bc = bias/acc_scale - z_a*sum(w)
        # otherwise the zero point is counted once (by the padding) instead of twice,
        # which shifted every output by up to 128*sum(w) and saturated the layer.
        acc_scale = in_scale * wscale
        sumw = q.astype(np.int64).sum(axis=(1, 2, 3))
        bc = np.round(b / acc_scale).astype(np.int64) - np.int64(in_zp) * sumw
        bc = np.clip(bc, -(1 << 31), (1 << 31) - 1).astype(np.int32)

        pack = pack_weights(q)
        for tag, arr, dt in (("w", pack, "<u4"), ("bc", bc, "<i4"),
                             ("mult", np.full(q.shape[0], M, np.int64), "<i4"),
                             ("shift", np.full(q.shape[0], S, np.int64), "<i4")):
            arr.astype(dt).tofile(os.path.join(args.out, f"{name}.{tag}.bin"))

        manifest["layers"].append({
            "name": name, "opIndex": li,
            "kernel": k, "stride": [s, s], "pad": [p, p, p, p],
            "inChannels": int(q.shape[3]), "outChannels": int(q.shape[0]),
            "inC4": int(q.shape[3] // 4), "outC4": int(q.shape[0] // 4),
            "K4": int(k * k * (q.shape[3] // 4)),
            "outZeroPoint": int(out_zp), "implicitRelu": bool(relu),
            "inputZeroPoint": int(in_zp),
            "multiplier": int(M), "shift": int(S),
            "weightScale": float(wscale), "inputScale": float(in_scale),
            "outputScale": float(out_scale), "realMultiplier": float(real_mult),
            "wWords": int(pack.size),
            "files": {t: f"{name}.{t}.bin" for t in ("w", "bc", "mult", "shift")},
        })
        print(f"{name:<15} {wscale:>11.8f} {in_scale:>11.8f} {out_scale:>11.8f} {out_zp:>7} "
              f"{real_mult:>11.6f} {M:>11} {S:>3}")
        params[name] = dict(q=q, bc=bc, M=M, S=S, out_zp=int(out_zp), k=k, s=s, p=p,
                            in_scale=in_scale, wscale=wscale, out_scale=out_scale)

    manifest["totals"] = {"weightBytes": int(sum(l["wWords"] for l in manifest["layers"]) * 4),
                          "convLayers": len(manifest["layers"]),
                          "nonConvOps": {"resize": 1, "concat": 2},
                          "totalDispatches": len(manifest["layers"]) + 3}
    with open(os.path.join(args.out, "nfru_packed.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    if not (args.golden_input and args.golden_output):
        return 0

    # ---------------- validation ----------------
    import torch
    xf = torch.load(args.golden_input, map_location="cpu", weights_only=False).float()
    g = torch.load(args.golden_output, map_location="cpu", weights_only=False).float().numpy()
    q_in = np.clip(np.round(xf.permute(0, 2, 3, 1).numpy() * 255.0) - 128.0,
                   -128, 127).astype(np.int8)

    def conv(t, name):
        d = params[name]
        k, s, p = d["k"], d["s"], d["p"]
        N, H, W, C = t.shape
        oH = (H + 2 * p - k) // s + 1
        oW = (W + 2 * p - k) // s + 1
        c = np.pad(t.astype(np.int64), ((0, 0), (p, p), (p, p), (0, 0)),
                   mode="constant", constant_values=IN_ZP)
        rows = (np.arange(oH) * s)[:, None] + np.arange(k)[None, :]
        cols = (np.arange(oW) * s)[:, None] + np.arange(k)[None, :]
        patch = c[:, rows[:, :, None, None], cols[None, None, :, :], :]
        patch = patch.transpose(0, 1, 3, 2, 4, 5).reshape(N, oH, oW, k * k * C)
        acc = patch @ d["q"].reshape(d["q"].shape[0], -1).astype(np.int64).T
        v = acc + d["bc"].reshape(1, 1, 1, -1)
        y = (v * np.int64(d["M"]) + (np.int64(1) << (d["S"] - 1))) >> d["S"]
        return np.clip(y + d["out_zp"], -128, 127).astype(np.int8)

    t = conv(q_in, "conv1"); t = conv(t, "conv2"); t = conv(t, "conv3")
    skip1 = conv(t, "skip1_conv"); bn = conv(t, "conv5")
    xa = conv(bn, "conv5a"); xb = conv(bn, "conv5b")
    xc = conv(conv(bn, "conv5c"), "conv5c_1")
    xd = conv(conv(conv(bn, "conv5d"), "conv5d_1"), "conv5d_2")
    t = np.concatenate([xa, xb, xc, xd], axis=3)
    t = conv(t, "conv5e")
    t = np.repeat(np.repeat(t, 2, axis=1), 2, axis=2)
    t = conv(t, "conv6")
    t = np.concatenate([t, skip1], axis=3)
    t = conv(t, "conv7")
    o = conv(t, "output_conv_mv")

    osc, ozp = act(ACT_IDX[-1])
    logits = (osc * (o.astype(np.float32) - ozp)).transpose(0, 3, 1, 2)
    print(f"\nint8 codes [{o.min()}, {o.max()}] distinct {np.unique(o).size}")
    print(f"logits     [{logits.min():.4f}, {logits.max():.4f}] mean {logits.mean():.4f}")
    print(f"golden     [{g.min():.4f}, {g.max():.4f}] mean {g.mean():.4f}")
    if logits.shape == g.shape:
        c = float(np.corrcoef(logits.reshape(-1), g.reshape(-1))[0, 1])
        err = np.abs(logits - g)
        print(f"\n*** int8 vs golden: correlation {c:.4f}  mean|err| {err.mean():.4f} ***")
        for i in range(g.shape[1]):
            ci = float(np.corrcoef(logits[0, i].ravel(), g[0, i].ravel())[0, 1])
            print(f"    ch{i}: {ci:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
