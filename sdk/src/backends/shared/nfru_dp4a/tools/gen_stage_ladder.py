#!/usr/bin/env python3
"""Generate the CPU reference tensor produced after the first N graph dispatches.

`NFRU_DP4A_STOP_AFTER=<n>` makes the backend record only the first n dispatches and bind
the last one to the host output buffer. Comparing that against the tensor computed here
pinpoints the first dispatch after which the GPU diverges -- the only reliable way to
localise a fault in a 19-dispatch graph, since the end result hides where the error was
introduced.

The op list and tensor-slot assignment below mirror `fru-port/src/nfru_graph.cpp`
exactly:

    conv1->2  conv2->3  conv3->4  skip1_conv->5  conv5->6
    conv5a->7 conv5b->8 conv5c->9 conv5c_1->10   conv5d->11
    conv5d_1->12 conv5d_2->13
    concat(7,8,10,13)->14   conv5e->15
    resize(15)->3   conv6->4   concat(4,5)->6
    conv7->7   output_conv_mv->1(host)

Usage:
    python gen_stage_ladder.py --counts 1,13,14,15,16,17,18,19 --out port/nfru_qat/stage
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (name, kernel, stride, pad) in execution order
GEOM = {
    "conv1": (3, 1, 1), "conv2": (5, 1, 2), "conv3": (3, 1, 1),
    "skip1_conv": (3, 1, 1), "conv5": (5, 2, 2),
    "conv5a": (1, 1, 0), "conv5b": (3, 1, 1),
    "conv5c": (7, 1, 3), "conv5c_1": (7, 1, 3),
    "conv5d": (7, 1, 3), "conv5d_1": (7, 1, 3), "conv5d_2": (7, 1, 3),
    "conv5e": (1, 1, 0), "conv6": (3, 1, 1), "conv7": (3, 1, 1),
    "output_conv_mv": (5, 1, 2),
}
# (kind, layerName or None, dst, src, [srcs])
OPS = [
    ("conv", "conv1", 2, 0, []),
    ("conv", "conv2", 3, 2, []),
    ("conv", "conv3", 4, 3, []),
    ("conv", "skip1_conv", 5, 4, []),
    ("conv", "conv5", 6, 4, []),
    ("conv", "conv5a", 7, 6, []),
    ("conv", "conv5b", 8, 6, []),
    ("conv", "conv5c", 9, 6, []),
    ("conv", "conv5c_1", 10, 9, []),
    ("conv", "conv5d", 11, 6, []),
    ("conv", "conv5d_1", 12, 11, []),
    ("conv", "conv5d_2", 13, 12, []),
    ("concat", None, 14, -1, [7, 8, 10, 13]),
    ("conv", "conv5e", 15, 14, []),
    ("resize", None, 3, 15, []),
    ("conv", "conv6", 4, 3, []),
    ("concat", None, 6, -1, [4, 5]),
    ("conv", "conv7", 7, 6, []),
    ("conv", "output_conv_mv", 1, 7, []),
]
ZP = -128
INPUT_SLOT = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="port/nfru_qat")
    ap.add_argument("--input", default="port/nfru_qat/golden_input_codes.bin")
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", default="port/nfru_qat/stage")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.packed, "nfru_packed.json"), encoding="utf-8"))
    byname = {l["name"]: l for l in man["layers"]}

    def load(name):
        p = os.path.join(args.packed, name)
        return (np.ascontiguousarray(np.fromfile(p + ".w.bin", dtype="<u4")).view(np.int8),
                np.fromfile(p + ".bc.bin", dtype="<i4").astype(np.int64),
                int(np.fromfile(p + ".mult.bin", dtype="<i4")[0]),
                int(np.fromfile(p + ".shift.bin", dtype="<i4")[0]))

    def conv(t, name):
        w8, bc, M, S = load(name)
        k, s, p = GEOM[name]
        oc = byname[name]["outChannels"]
        out_zp = byname[name]["outZeroPoint"]
        N, H, W, C = t.shape
        oH = (H + 2 * p - k) // s + 1
        oW = (W + 2 * p - k) // s + 1
        c = np.pad(t.astype(np.int64), ((0, 0), (p, p), (p, p), (0, 0)),
                   mode="constant", constant_values=ZP)
        rows = (np.arange(oH) * s)[:, None] + np.arange(k)[None, :]
        cols = (np.arange(oW) * s)[:, None] + np.arange(k)[None, :]
        patch = c[:, rows[:, :, None, None], cols[None, None, :, :], :]
        patch = patch.transpose(0, 1, 3, 2, 4, 5).reshape(N, oH, oW, k * k * C)
        acc = patch @ w8.reshape(oc, k * k * C).astype(np.int64).T
        v = acc + bc.reshape(1, 1, 1, -1)
        y = (v * np.int64(M) + (np.int64(1) << (S - 1))) >> S
        return np.clip(y + out_zp, -128, 127).astype(np.int8)

    slot = {INPUT_SLOT: np.fromfile(args.input, dtype=np.int8)
            .reshape(1, args.height, args.width, 16)}
    counts = sorted(int(c) for c in args.counts.split(","))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    written = {}
    for n, (kind, name, dst, src, srcs) in enumerate(OPS, start=1):
        if kind == "conv":
            slot[dst] = conv(slot[src], name)
        elif kind == "resize":
            slot[dst] = np.repeat(np.repeat(slot[src], 2, axis=1), 2, axis=2)
        else:
            slot[dst] = np.concatenate([slot[s] for s in srcs], axis=3)
        if n in counts:
            t = slot[dst]
            path = f"{args.out}_{n:02d}.bin"
            np.ascontiguousarray(t).tofile(path)
            written[n] = (path, t.shape, int(t.min()), int(t.max()))
            print(f"n={n:>2} {kind:<7} {name or '':<15} -> shape {t.shape} "
                  f"range {t.min()}..{t.max()}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
