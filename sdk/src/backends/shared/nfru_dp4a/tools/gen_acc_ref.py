#!/usr/bin/env python3
"""Generate the RAW int32 accumulator of one convolution, for divergence attribution.

The packed int8 output conflates three stages: accumulation, bias addition and
requantisation. When the GPU output is wrong there is no way to tell which stage is at
fault by looking at the codes alone.

With `NFRU_DP4A_DUMP_ACC=1` the kernel takes its debug path (has_lut == 2) and writes
one int32 accumulator per output channel instead of the packed code, so the two can be
compared directly:

  * accumulators equal, codes differ   -> the bias or the requantiser is wrong
  * accumulators differ                -> the accumulation (weights or activation
                                          addressing) is wrong

Layout: for each output pixel, 4 output channels per group, `outC4` groups, so
`outW * outH * outC4 * 4` int32 values in total, group-major within the pixel.

Usage:
    python gen_acc_ref.py --layer conv5 --out port/nfru_qat/acc_conv5.bin
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEOM = {
    "conv1": (3, 1, 1), "conv2": (5, 1, 2), "conv3": (3, 1, 1),
    "skip1_conv": (3, 1, 1), "conv5": (5, 2, 2),
    "conv5a": (1, 1, 0), "conv5b": (3, 1, 1),
    "conv5c": (7, 1, 3), "conv5c_1": (7, 1, 3),
    "conv5d": (7, 1, 3), "conv5d_1": (7, 1, 3), "conv5d_2": (7, 1, 3),
    "conv5e": (1, 1, 0), "conv6": (3, 1, 1), "conv7": (3, 1, 1),
    "output_conv_mv": (5, 1, 2),
}
ZP = -128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--input", required=True, help="int8 NHWC tensor this layer reads")
    ap.add_argument("--packed", default="port/nfru_qat")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.packed, "nfru_packed.json"), encoding="utf-8"))
    L = next(l for l in man["layers"] if l["name"] == args.layer)
    k, s, p = GEOM[args.layer]
    oc = L["outChannels"]
    out_c4 = L["outC4"]
    K4 = L["K4"]

    wraw = np.fromfile(os.path.join(args.packed, f"{args.layer}.w.bin"), dtype="<u4")
    w8 = np.ascontiguousarray(wraw).view(np.int8).reshape(oc, k, k, L["inChannels"])

    x = np.fromfile(args.input, dtype=np.int8).reshape(1, args.height, args.width, -1)
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    assert C == L["inChannels"], (C, L["inChannels"])
    oH = (H + 2 * p - k) // s + 1
    oW = (W + 2 * p - k) // s + 1

    c = np.pad(x.astype(np.int64), ((0, 0), (p, p), (p, p), (0, 0)),
               mode="constant", constant_values=ZP)
    rows = (np.arange(oH) * s)[:, None] + np.arange(k)[None, :]
    cols = (np.arange(oW) * s)[:, None] + np.arange(k)[None, :]
    patch = c[:, rows[:, :, None, None], cols[None, None, :, :], :]
    patch = patch.transpose(0, 1, 3, 2, 4, 5).reshape(oH, oW, k * k * C)
    acc = patch @ w8.reshape(oc, k * k * C).astype(np.int64).T      # [oH,oW,oc]

    # Layout must match the kernel's debug path exactly: one int32 per output CHANNEL,
    # raster order within the pixel, i.e. word (pixel*oc + channel). The accumulator array
    # already has that shape, so it is written straight out.
    np.ascontiguousarray(acc.astype("<i4")).tofile(args.out)
    print(f"{args.layer}: acc {acc.shape} -> {args.out} ({acc.size * 4} bytes)")
    print(f"  acc range [{acc.min()}, {acc.max()}]  K4={K4} inC4={L['inC4']} outC4={out_c4}")
    print(f"  first pixel accumulators: {acc[0, 0, :8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
