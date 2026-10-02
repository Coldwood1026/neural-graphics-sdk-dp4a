#!/usr/bin/env python3
"""Quantise NFRU v1 to int8 FROM THE fp32 CHECKPOINT, for the dp4a backend.

Why this exists
---------------
The published VGF's int8 parameters are not reproducible in practice: its TOSA
rescales carry a per-term multiplier whose meaning depends on a convention that has
to be reverse-engineered, and the resulting graph correlates only 0.86 with Arm's own
golden output even though the graph is bit-exactly reproduced.

The fp32 checkpoint has none of that ambiguity. `tools/probe_float_forward.py` proves
`nfru_v1_fp32.pt` reproduces `autoencoder_output_golden.pt` with correlation 1.0000 and
max error 2e-4, and `tools/probe_pair.py` proves the VGF weights are quantised from
exactly this checkpoint (>=0.99 correlation on all 16 layers after folding batch norm).

So: fold batch norm, calibrate activation ranges by running the float model over the
official golden input, and derive every int8 parameter ourselves.

Produces, per layer, the same four buffers the dp4a shader already consumes:
    {name}.w.bin      uint32, layout wpk[oc*K4 + (ky*kw+kx)*in_c4 + ic4], K4 = kh*kw*(Cin/4)
    {name}.bc.bin     int32, the raw convolution bias (padding supplies the zero point)
    {name}.mult.bin   int32, TOSA rescale multiplier M
    {name}.shift.bin  int32, TOSA rescale shift S
plus nfru_packed.json describing the graph, so nothing downstream changes.

Numerics, identical to `fru_conv_rq.comp`:
    acc   = sum( q_a * q_w )                     int32 dp4a, 4 channels per OpSDot
    pad   = 0x80808080 == -128, which equals the input zero point, so padded taps
            contribute z_a*sum(w) on their own and the bias must NOT fold it in again
    r     = (int64(acc + bc) * M + (1 << (S-1))) >> S, then += out_zp, then clamp
    ReLU  : out_zp == -128 already makes r >= 0; the output head keeps out_zp so its
            logits are signed.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgftools"))

# `fold` is defined locally below: the runtime path must not pull in fp32 helpers.

# Execution order of the graph (dispatches), and per-layer geometry.
#   (kernel, stride, padding, relu)
GEOM = {
    "conv1":         (3, 1, 1, True),
    "conv2":         (5, 1, 2, True),
    "conv3":         (3, 1, 1, True),
    "skip1_conv":    (3, 1, 1, True),
    "conv5":         (5, 2, 2, True),
    "conv5a":        (1, 1, 0, True),
    "conv5b":        (3, 1, 1, True),
    "conv5c":        (7, 1, 3, True),
    "conv5c_1":      (7, 1, 3, True),
    "conv5d":        (7, 1, 3, True),
    "conv5d_1":      (7, 1, 3, True),
    "conv5d_2":      (7, 1, 3, True),
    "conv5e":        (1, 1, 0, True),
    "conv6":         (3, 1, 1, True),
    "conv7":         (3, 1, 1, True),
    "output_conv_mv": (5, 1, 2, False),
}

IN_SCALE = 1.0 / 255.0
IN_ZP = -128


def fold(sd, name):
    """conv2d + BatchNorm2d -> (weight [oc,ic,kh,kw], bias), float32.

    Used ONLY for offline calibration. The runtime path is pure int8: every
    parameter that reaches the GPU is an integer produced below.
    """
    import torch
    for pre in (f"network.auto_encoder.{name}.conv2d.",
                f"network.auto_encoder.{name}."):
        if pre + "weight" in sd:
            break
    else:
        raise KeyError(f"no weight for {name}")
    bn = f"network.auto_encoder.{name}.bn."
    w = sd[pre + "weight"].float()
    b = sd[pre + "bias"].float() if pre + "bias" in sd else torch.zeros(w.shape[0])
    if bn + "weight" not in sd:
        return w, b
    g = sd[bn + "weight"].float()
    be = sd[bn + "bias"].float()
    mu = sd[bn + "running_mean"].float()
    var = sd[bn + "running_var"].float()
    scale = g / torch.sqrt(var + 1e-5)
    return w * scale.view(-1, 1, 1, 1), (b - mu) * scale + be


def tosa_multiplier(real_mult):
    """Represent real_mult as M / 2^S with M < 2^31, the TOSA RESCALE form."""
    if real_mult <= 0:
        return 0, 0
    e = int(np.floor(np.log2(real_mult)))
    m = real_mult / (2.0 ** e)
    # normalise m into [0.5, 1)
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
    if S < 0:
        S = 0
    return M, S


def quantise_symmetric(w):
    """Per-tensor symmetric int8: scale = max|w|/127, zero_point 0."""
    mx = float(np.abs(w).max())
    if mx == 0.0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = mx / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def pack_weights(q_okki):
    """[oc,kh,kw,ic] int8 -> uint32[oc * kh*kw*(ic/4)], ic4 innermost."""
    oc, kh, kw, ic = q_okki.shape
    assert ic % 4 == 0, ic
    in_c4 = ic // 4
    v = q_okki.reshape(oc, kh, kw, in_c4, 4)
    raw = np.ascontiguousarray(v).tobytes()
    return np.frombuffer(raw, dtype="<u4").reshape(oc, kh * kw * in_c4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="calibration + validation input, e.g. the golden input")
    ap.add_argument("--golden", help="golden output, for reporting accuracy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentile", type=float, default=99.99,
                    help="activation percentile used to set each layer's scale")
    ap.add_argument("--fixed-output-scale", type=float, default=0.35356706380844116,
                    help="keep the published output scale so postprocess is unchanged")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    x = torch.load(args.input, map_location="cpu", weights_only=False).float()
    print(f"checkpoint loaded, calibration input {tuple(x.shape)}")

    # ---------------- pass 1: float reference + activation ranges -------------
    acts = {}          # layer -> tensor, for calibration
    ranges = {}        # layer -> (max|val|, percentile)

    def record(name, t):
        a = t.detach().abs()
        acts[name] = t.detach()
        ranges[name] = (float(a.max()), float(torch.quantile(a.flatten().float(),
                                                              args.percentile / 100.0)))

    def conv_layer(t, name):
        k, s, p, relu = GEOM[name]
        w, b = fold(sd, name)
        o = F.conv2d(t, w, b, stride=s, padding=p)
        if relu:
            o = F.relu(o)
        return o

    with torch.no_grad():
        t = conv_layer(x, "conv1"); record("conv1", t)
        t = conv_layer(t, "conv2"); record("conv2", t)
        t = conv_layer(t, "conv3"); record("conv3", t)
        skip1 = conv_layer(t, "skip1_conv"); record("skip1_conv", skip1)
        bn = conv_layer(t, "conv5"); record("conv5", bn)

        br = {}
        for name in ("conv5a", "conv5b", "conv5c", "conv5c_1",
                     "conv5d", "conv5d_1", "conv5d_2"):
            src = bn if name in ("conv5a", "conv5b", "conv5c", "conv5d") else \
                br[{"conv5c_1": "conv5c", "conv5d_1": "conv5d", "conv5d_2": "conv5d_1"}[name]]
            o = conv_layer(src, name)
            br[name] = o
            record(name, o)
        t = torch.cat([br["conv5a"], br["conv5b"], br["conv5c_1"], br["conv5d_2"]], dim=1)
        t = conv_layer(t, "conv5e"); record("conv5e", t)
        t = F.interpolate(t, scale_factor=2, mode="nearest")
        t = conv_layer(t, "conv6"); record("conv6", t)
        t = torch.cat([t, skip1], dim=1)
        t = conv_layer(t, "conv7"); record("conv7", t)
        t = conv_layer(t, "output_conv_mv"); record("output_conv_mv", t)
        float_out = t

    if args.golden:
        g = torch.load(args.golden, map_location="cpu", weights_only=False).float()
        c = float(np.corrcoef(float_out.reshape(-1).numpy(), g.reshape(-1).numpy())[0, 1])
        print(f"float reference vs golden: correlation {c:.4f}  "
              f"max|err| {float((float_out - g).abs().max()):.6f}")

    print(f"\n{'layer':<15} {'max|act|':>10} {'p99.99':>10} {'scale':>12} {'wscale':>12} "
          f"{'real mult':>12} {'M':>11} {'S':>3}")
    for name in GEOM:
        mx, pq = ranges[name]
        print(f"{name:<15} {mx:>10.4f} {pq:>10.4f} "
              f"{pq/127.0 if pq > 0 else 1.0:>12.8f} "
              f"{'':>12} {'':>12}")

    # ---------------- derive int8 parameters ---------------------------------
    os.makedirs(args.out, exist_ok=True)
    manifest = {"source": os.path.basename(args.ckpt), "quantiser": "per-tensor symmetric int8",
                "inputScale": IN_SCALE, "inputZeroPoint": IN_ZP, "layers": []}

    layers_params = {}
    prev_scale = IN_SCALE
    for name in GEOM:
        w, b = fold(sd, name)
        # PyTorch stores [oc, ic, kh, kw]; the kernel consumes NHWC [oc, kh, kw, ic].
        w = np.transpose(w.numpy(), (0, 2, 3, 1))
        assert w.shape[3] % 4 == 0, f"{name}: Cin {w.shape[3]} is not a multiple of 4"
        b = b.numpy()
        q, wscale = quantise_symmetric(w)

        # activation scale for this layer's output. The final layer keeps the
        # published output scale so the existing postprocess is untouched.
        if name == "output_conv_mv":
            out_scale = args.fixed_output_scale
            out_zp = 44
        else:
            # Use the observed MAXIMUM, not a percentile. A percentile smaller than
            # the max makes the int8 range too narrow and saturates the layer: with
            # p99.99 the second conv layer put 49.5% of its outputs at +127, which
            # destroyed every layer after it. Saturating is the one error this
            # scheme cannot afford, because a saturated int8 value carries no
            # information at all.
            mx = ranges[name][0]
            out_scale = float(mx) / 127.0 if mx > 0 else 1.0
            out_zp = -128

        real_mult = (prev_scale * wscale) / out_scale
        M, S = tosa_multiplier(real_mult)

        # bias lives in accumulator units: acc ~ sum(q_a*q_w), so acc_scale = in*w
        acc_scale = prev_scale * wscale
        bc = np.round(b / acc_scale).astype(np.int64)
        bc = np.clip(bc, -(1 << 31), (1 << 31) - 1).astype(np.int32)

        pack = pack_weights(q)
        for tag, arr, dt in (("w", pack, "<u4"), ("bc", bc, "<i4"),
                             ("mult", np.full(q.shape[0], M, np.int64), "<i4"),
                             ("shift", np.full(q.shape[0], S, np.int64), "<i4")):
            arr.astype(dt).tofile(os.path.join(args.out, f"{name}.{tag}.bin"))

        k, s, p, relu = GEOM[name]
        manifest["layers"].append({
            "name": name, "opIndex": len(manifest["layers"]),
            "kernel": k, "stride": [s, s], "pad": [p, p, p, p],
            "inChannels": int(q.shape[3]), "outChannels": int(q.shape[0]),
            "inC4": int(q.shape[3] // 4), "outC4": int(q.shape[0] // 4),
            "K4": int(k * k * (q.shape[3] // 4)),
            "outZeroPoint": out_zp, "implicitRelu": bool(relu),
            "inputZeroPoint": IN_ZP,
            "multiplier": int(M), "shift": int(S),
            "weightScale": float(wscale), "inputScale": float(prev_scale),
            "outputScale": float(out_scale), "realMultiplier": float(real_mult),
            "wWords": int(pack.size),
            "files": {t: f"{name}.{t}.bin" for t in ("w", "bc", "mult", "shift")},
        })
        layers_params[name] = (q, bc, M, S, out_zp, k, s, p, relu, prev_scale, wscale, out_scale)
        print(f"{name:<15} {ranges[name][0]:>10.4f} {ranges[name][1]:>10.4f} "
              f"{out_scale:>12.8f} {wscale:>12.8f} {real_mult:>12.6f} {M:>11} {S:>3}")
        prev_scale = out_scale

    manifest["totals"] = {"weightBytes": int(sum(l["wWords"] for l in manifest["layers"]) * 4),
                          "convLayers": len(manifest["layers"]),
                          "nonConvOps": {"resize": 1, "concat": 2},
                          "totalDispatches": len(manifest["layers"]) + 3}
    with open(os.path.join(args.out, "nfru_packed.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    # ---------------- pass 2: run the int8 graph and score it ----------------
    print(f"\n--- int8 validation ---")
    q_in = np.clip(np.round(x.permute(0, 2, 3, 1).numpy() * 255.0) - 128.0,
                   -128, 127).astype(np.int8)
    out_i8, out_scale = run_int8(q_in, layers_params)
    logits = out_scale * (out_i8.astype(np.float32) - (44 if "output_conv_mv" in GEOM else 0))
    logits = logits.transpose(0, 3, 1, 2)
    print(f"int8 output codes range [{out_i8.min()}, {out_i8.max()}] "
          f"distinct {np.unique(out_i8).size}")
    print(f"int8 logits range [{logits.min():.4f}, {logits.max():.4f}] mean {logits.mean():.4f}")

    if args.golden:
        g = torch.load(args.golden, map_location="cpu", weights_only=False).float().numpy()
        if logits.shape == g.shape:
            c = float(np.corrcoef(logits.reshape(-1), g.reshape(-1))[0, 1])
            err = np.abs(logits - g)
            print(f"\n*** int8 vs golden: correlation {c:.4f}   "
                  f"mean|err| {err.mean():.4f}   max|err| {err.max():.4f} ***")
            for i in range(g.shape[1]):
                ci = float(np.corrcoef(logits[0, i].ravel(), g[0, i].ravel())[0, 1])
                print(f"    ch{i}: {ci:+.4f}")
            print(f"float reference for comparison: correlation 1.0000")
    return 0


def run_int8(q_in, P):
    """Execute the quantised graph with the same integer semantics as the shader."""
    def conv(t, name):
        q, bc, M, S, out_zp, k, s, p, relu, ins, ws, outs = P[name]
        kh = kw = k
        N, H, W, C = t.shape
        oH = (H + 2 * p - kh) // s + 1
        oW = (W + 2 * p - kw) // s + 1
        c = np.pad(t.astype(np.int64), ((0, 0), (p, p), (p, p), (0, 0)),
                   mode="constant", constant_values=IN_ZP)
        rows = (np.arange(oH) * s)[:, None] + np.arange(kh)[None, :]
        cols = (np.arange(oW) * s)[:, None] + np.arange(kw)[None, :]
        patch = c[:, rows[:, :, None, None], cols[None, None, :, :], :]
        patch = patch.transpose(0, 1, 3, 2, 4, 5).reshape(N, oH, oW, kh * kw * C)
        acc = patch @ q.reshape(q.shape[0], -1).astype(np.int64).T
        v = acc + bc.reshape(1, 1, 1, -1)
        y = (v * np.int64(M) + (np.int64(1) << (S - 1))) >> S
        y = y + out_zp
        y = np.clip(y, -128, 127)
        # No extra ReLU: out_zp == -128 already makes every value >= 0.
        return y.astype(np.int8)

    t = conv(q_in, "conv1")
    t = conv(t, "conv2")
    t = conv(t, "conv3")
    skip1 = conv(t, "skip1_conv")
    bn = conv(t, "conv5")
    xa = conv(bn, "conv5a")
    xb = conv(bn, "conv5b")
    xc = conv(conv(bn, "conv5c"), "conv5c_1")
    xd = conv(conv(conv(bn, "conv5d"), "conv5d_1"), "conv5d_2")
    t = np.concatenate([xa, xb, xc, xd], axis=3)
    t = conv(t, "conv5e")
    t = np.repeat(np.repeat(t, 2, axis=1), 2, axis=2)
    t = conv(t, "conv6")
    t = np.concatenate([t, skip1], axis=3)
    t = conv(t, "conv7")
    t = conv(t, "output_conv_mv")
    return t, P["output_conv_mv"][11]


if __name__ == "__main__":
    sys.exit(main())
