#!/usr/bin/env python3
"""AUTHORITATIVE per-layer chain comparison for the NFRU int8 port.

Several rounds of debugging failed because each probe covered only part of the chain
(accumulator here, bias there, requant somewhere else), so an inconsistency between
probes could always be blamed on the probe. This program prints the ENTIRE chain for
one pixel side by side, from the scalar dot product through the requantised value, and
cross-checks it against the packed-buffer kernel path (conv_kernel) on the same pixel.

Chain, per output channel:
    acc      = sum_ic q_a[ic] * q_w[oc,ic]      (over the 3x3 patch, pad = pad_value)
    v_ref    = acc + bias[oc]
    pre_ref  = ((v_ref * M) + (1 << (S-1))) >> S
    out_ref  = clamp(pre_ref + out_zp, -128, 127)
The kernel path writes acc_k, v_k, pre_k, out_k from its own buffers; all four are
printed so the first divergence is visible directly rather than inferred.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgftools"))

from vgf_graph import analyze                              # noqa: E402
from nss_ref import load_constants, const_array             # noqa: E402
from verify_packed import pack_activations, sdot, Z_A_WORD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vgf", required=True)
    ap.add_argument("--baked", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--op", type=int, default=0)
    ap.add_argument("--pad-value", type=int, default=None,
                    help="value for out-of-range taps; default = inputZeroPoint")
    ap.add_argument("--bias", choices=["raw", "raw+corr", "bcfile"], default="raw")
    ap.add_argument("--oy", type=int, default=10)
    ap.add_argument("--ox", type=int, default=12)
    args = ap.parse_args()

    layers = {l["opIndex"]: l for l in json.load(open(args.baked, encoding="utf-8"))["layers"]}
    packed = json.load(open(os.path.join(args.packed, "nfru_packed.json"), encoding="utf-8"))
    pl = next(p for p in packed["layers"] if p["opIndex"] == args.op)
    L = layers[args.op]
    name = pl["name"]
    ir = analyze(args.vgf)
    consts = load_constants(args.vgf, ir)
    r = next(x for x in ir["ops"] if x["index"] == args.op)
    w = np.asarray(const_array(r["weight"], consts), dtype=np.int8)
    raw_bias = np.asarray(const_array(r["bias"], consts), dtype=np.int64).reshape(-1)
    if raw_bias.size == 1:
        raw_bias = np.repeat(raw_bias, w.shape[0])
    oc_n = w.shape[0]
    izp = L["inputZeroPoint"]
    pad_value = izp if args.pad_value is None else args.pad_value
    corr = -np.int64(izp) * w.astype(np.int64).sum(axis=(1, 2, 3))
    bc_file = np.fromfile(os.path.join(args.packed, f"{name}.bc.bin"), dtype="<i4").astype(np.int64)
    bias = {"raw": raw_bias, "raw+corr": raw_bias + corr, "bcfile": bc_file}[args.bias]
    mult = np.fromfile(os.path.join(args.packed, f"{name}.mult.bin"), dtype="<i4")
    shift = np.fromfile(os.path.join(args.packed, f"{name}.shift.bin"), dtype="<i4")
    wpk = np.fromfile(os.path.join(args.packed, f"{name}.w.bin"), dtype="<u4").reshape(-1)

    kh = kw = L["kernel"]
    sh, sw = L["strideH"], L["strideW"]
    pt, plf, pr = L["padT"], L["padL"], L["padR"]
    in_c4 = pl["inC4"]
    K4 = kh * kw * in_c4
    M, S, zp = int(mult[0]), int(shift[0]), int(L["outputZeroPoint"])

    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, 24, 32, 16), dtype=np.int8)
    act = pack_activations(x)
    _, in_h, in_w, _ = x.shape

    oy, ox = args.oy, args.ox
    print(f"{name} op{args.op}  pixel ({oy},{ox})  pad_value={pad_value}  bias={args.bias}")
    print(f"  M={M} S={S} out_zp={zp}  kh=kw={kh} stride=({sh},{sw}) padT={pt} padL={plf} "
          f"in_c4={in_c4} K4={K4} oc_n={oc_n}")

    print(f"\n{'oc':>3} {'acc':>8} {'bias':>8} {'v':>8} {'pre':>8} {'[ref]':>6} | "
          f"{'acc_k':>8} {'v_k':>8} {'pre_k':>8} {'[krn]':>6} | flag")
    for oc in range(min(12, oc_n)):
        # --- scalar reference chain -------------------------------------
        acc = 0
        for ky in range(kh):
            iy = oy * sh - pt + ky
            for kx in range(kw):
                ix = ox * sw - plf + kx
                if iy < 0 or iy >= in_h or ix < 0 or ix >= in_w:
                    av = [pad_value] * 16
                else:
                    av = [int(v) for v in x[0, iy, ix, :]]
                for ic in range(16):
                    acc += av[ic] * int(w[oc, ky, kx, ic])
        v = acc + int(bias[oc])
        pre = ((np.int64(v) * np.int64(M) + (np.int64(1) << (S - 1))) >> S)
        out_ref = int(max(np.int64(-128), min(np.int64(127), pre + np.int64(zp))))

        # --- kernel chain, using its own packed buffers -----------------
        acc_k = 0
        for ky in range(kh):
            iy = oy * sh - pt + ky
            for kx in range(kw):
                ix = ox * sw - plf + kx
                tap = (iy < 0 or iy >= in_h or ix < 0 or ix >= in_w)
                woff = (ky * kw + kx) * in_c4
                for ic4 in range(in_c4):
                    a = Z_A_WORD if tap else int(act[0, iy, ix, ic4])
                    acc_k += sdot(a, int(wpk[oc * K4 + woff + ic4]))
        v_k = acc_k + int(bc_file[oc])
        pre_k = ((np.int64(v_k) * np.int64(M) + (np.int64(1) << (S - 1))) >> S)
        out_k = int(max(np.int64(-128), min(np.int64(127), pre_k + np.int64(zp))))

        flag = ""
        if acc != acc_k:
            flag += "ACC "
        if v != v_k:
            flag += "V "
        if int(pre) != int(pre_k):
            flag += "PRE "
        if out_ref != out_k:
            flag += "OUT "
        print(f"{oc:>3} {acc:>8} {int(bias[oc]):>8} {v:>8} {int(pre):>8} {out_ref:>6} | "
              f"{acc_k:>8} {v_k:>8} {int(pre_k):>8} {out_k:>6} | {flag or 'ok'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
