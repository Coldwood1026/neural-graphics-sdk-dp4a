#!/usr/bin/env python3
"""CPU reference for the NFRU v1 int8 pipeline, mirroring the dp4a arithmetic.

Purpose
-------
The GPU kernel is pure integer arithmetic, so it can be reproduced exactly on
the CPU. This validates three things before any Vulkan code exists:
  1. the checkpoint's constant layout is understood (see below);
  2. the quantisation contract (pack / accumulate / requantise) is self-consistent;
  3. the int8 network tracks its own float path closely.

Checkpoint layout, verified empirically
---------------------------------------
The QAT checkpoint flattens the traced graph into indexed constants. Each of the
16 convolutions (traced in graph order) owns four consecutive `_param_constant`
entries, and each of the 15 BatchNorm layers owns two consecutive
`_tensor_constant` entries:

    _param_constant[4i + 0]  convolution weight
    _param_constant[4i + 1]  convolution bias
    _param_constant[4i + 2]  BatchNorm gamma
    _param_constant[4i + 3]  BatchNorm beta
    _tensor_constant[2i + 0] BatchNorm running_mean
    _tensor_constant[2i + 1] BatchNorm running_var          (all positive)

The output head (`output_conv_mv`) has no BatchNorm, hence (16*4 - 2) = 62
`_param_constant` and 15*2 = 30 `_tensor_constant`.

Usage:
    python int8_sim.py --int8 ../../fru/nfru_v1_int8.pt \
                       [--input <golden input .pt>] [--json report.json]
"""
import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

# Execution order, which is also the constant-group order in the checkpoint:
# group i occupies _param_constant[4i .. 4i+3] and _tensor_constant[2i .. 2i+1].
# name, kernel, stride, padding, cin, cout, has_bn, relu
LAYERS = [
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

# Quantisation index per layer, in execution order (from the op stream).
OUT_QP = {"conv1": 3, "conv2": 4, "conv3": 6, "skip1_conv": 7, "conv5": 9,
          "conv5a": 10, "conv5b": 12, "conv5c": 13, "conv5c_1": 15,
          "conv5d": 16, "conv5d_1": 18, "conv5d_2": 19, "conv5e": 21,
          "conv6": 22, "conv7": 24, "output_conv_mv": 51}
IN_QP = {"conv1": 0, "conv2": 3, "conv3": 6, "skip1_conv": 6, "conv5": 6,
         "conv5a": 9, "conv5b": 9, "conv5c": 9, "conv5c_1": 12, "conv5d": 9,
         "conv5d_1": 15, "conv5d_2": 18, "conv5e": 36, "conv6": 21,
         "conv7": 40, "output_conv_mv": 44}


def q_input(x):
    """Input packing from the shipped preprocess shader: (x*255) - 128.

    NOTE the offset convention, which is the subtle part of the whole pipeline:
    quantization points are stored as an affine (scale, zero_point) pair and the
    dequantization is

        x = scale * (q - zero_point)

    (not `scale * q`). For the graph input, scale = 1/255 and zero_point = -128,
    so

        q = round(x / scale) + zero_point = round(x*255) - 128

    which is exactly what the shader computes. Treating it as plain `scale * q`
    introduces a constant factor of 128/... error and was the bug that broke the
    first attempts at this simulation.
    """
    return torch.clamp(torch.round(x * 255.0) - 128.0, -128, 127)


def q_symmetric(t, bits=127.0):
    """Per-tensor symmetric int8 quantisation; zero_point is 0.

    Model Gym's `get_qat_quantization_profile` forces
    `per_channel_weight_quantization=False`, selecting `per_tensor_symmetric` with
    range [-127, 127] and a true min/max observer (averaging_constant 1.0), so
    `scale = max|w| / 127` is the exact rule, not an approximation.
    """
    amax = t.abs().max().item()
    scale = amax / bits if amax > 0 else 1.0
    return torch.clamp(torch.round(t / scale), -128, 127), scale


def fold_bn(w, b, gamma, beta, mean, var, eps=1e-5):
    std = torch.sqrt(var + eps)
    s = gamma / std
    return w * s.reshape(-1, 1, 1, 1), (b - mean) * s + beta


def build(checkpoint):
    msd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    P = "network.auto_encoder."
    params, tensors = {}, {}
    for k, v in msd.items():
        m = re.search(r"\._param_constant(\d+)$", k)
        if m:
            params[int(m.group(1))] = v.float()
        m = re.search(r"\._tensor_constant(\d+)$", k)
        if m:
            tensors[int(m.group(1))] = v.float()

    qp = {}
    for k, v in msd.items():
        m = re.search(r"activation_post_process_(\d+)\.scale$", k)
        if m:
            qp[int(m.group(1))] = float(v.item())

    prepared, bn_i = {}, 0
    for i, (name, k, stride, pad, cin, cout, has_bn, relu) in enumerate(LAYERS):
        w = params[4 * i + 0]
        b = params[4 * i + 1]
        # These assertions are what caught the original operand-order bug; keep them.
        assert w.shape[0] == cout and w.shape[1] == cin, \
            f"{name}: weight {tuple(w.shape)} vs cout={cout} cin={cin}"
        assert k == w.shape[2] == w.shape[3], \
            f"{name}: kernel {k} vs weight {tuple(w.shape)}"
        if has_bn:
            gamma, beta = params[4 * i + 2], params[4 * i + 3]
            mean, var = tensors[2 * bn_i], tensors[2 * bn_i + 1]
            assert bool((var > 0).all()), f"{name}: running_var not all positive"
            w, b = fold_bn(w, b, gamma, beta, mean, var)
            bn_i += 1
        qw, sw = q_symmetric(w)
        prepared[name] = dict(
            kernel=k, stride=stride, pad=pad, relu=relu,
            w_fold=w, b_fold=b,           # folded float weights (for the float path)
            w_q=qw, w_scale=sw,           # int8 weights
            mult=qp[IN_QP[name]] * sw / qp[OUT_QP[name]],
            in_scale=qp[IN_QP[name]], out_scale=qp[OUT_QP[name]],
        )
    return prepared, qp, bn_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", required=True)
    ap.add_argument("--input")
    ap.add_argument("--json")
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--w", type=int, default=480)
    args = ap.parse_args()

    prepared, qp, used_bn = build(args.int8)
    print(f"constant layout check: {len(prepared)} conv layers, {used_bn} BatchNorms consumed")

    def conv_float(x_nhwc, name):
        """Float path, operating on NHWC and returning NHWC."""
        p = prepared[name]
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        y = F.conv2d(x, p["w_fold"], p["b_fold"], stride=p["stride"], padding=p["pad"])
        if p["relu"]:
            y = torch.clamp(y, 0.0, None)
        return y.permute(0, 2, 3, 1).contiguous()

    def conv_int8(x_q, name):
        """Int8 path: NHWC in, NHWC out. Mirrors dp4a accumulate + requantise.

        Algebra, with activations and weights both symmetric int8 (zero_point 0):

            acc      = conv(i8(act), i8(w))          <- raw int32 dot products
            y_float  = acc * scale_in * scale_w + bias
            y_code   = round(y_float / scale_out)    <- requantise to output scale

        So the kernel multiplies the accumulator by
            m = scale_in * scale_w / scale_out
        and needs the bias pre-divided by (scale_in * scale_w) so it lands in the
        accumulator's integer domain:

            bias_acc = bias / (scale_in * scale_w)

        where the accumulator value is `acc + bias_acc`.

        Two earlier mistakes lived here and both produced large errors:
          * treating F.conv2d's output as already scaled (it is a raw accumulator);
          * adding bias after scaling, which effectively multiplied it by m again.
        """
        p = prepared[name]
        x = x_q.permute(0, 3, 1, 2).float()
        acc = F.conv2d(x, p["w_q"].float(), None, stride=p["stride"], padding=p["pad"])
        acc = acc + p["b_fold"].reshape(1, -1, 1, 1) / (p["in_scale"] * p["w_scale"])
        y = acc * p["mult"]
        if p["relu"]:
            y = torch.clamp(y, 0.0, None)
        return torch.clamp(torch.round(y), -128, 127).permute(0, 2, 3, 1).contiguous()

    if args.input:
        x_raw = torch.load(args.input, map_location="cpu", weights_only=False).float()
        # The released golden input is NCHW (1,16,270,480). Everything inside this
        # script works NHWC, matching the layout the VGF declares.
        if x_raw.dim() == 4 and x_raw.shape[1] == 16:
            x = x_raw.permute(0, 2, 3, 1).contiguous()
        elif x_raw.dim() == 4 and x_raw.shape[3] == 16:
            x = x_raw.contiguous()
        else:
            raise SystemExit(f"unexpected input shape {tuple(x_raw.shape)}")
        print(f"input raw {tuple(x_raw.shape)} -> NHWC {tuple(x.shape)}")
    else:
        x = torch.rand(1, args.h, args.w, 16)
        print(f"input NHWC {tuple(x.shape)}")

    def run(conv):
        """Both paths use NHWC internally; cat is on the channel axis."""
        a = conv(x, "conv1")
        a = conv(a, "conv2")
        a = conv(a, "conv3")
        skip1 = conv(a, "skip1_conv")
        b = conv(a, "conv5")
        xa = conv(b, "conv5a")
        xb = conv(b, "conv5b")
        xc = conv(conv(b, "conv5c"), "conv5c_1")
        xd = conv(conv(conv(b, "conv5d"), "conv5d_1"), "conv5d_2")
        cat1 = torch.cat([xa, xb, xc, xd], dim=3)
        y = conv(cat1, "conv5e")
        y = F.interpolate(y.permute(0, 3, 1, 2), scale_factor=2, mode="nearest")
        y = y.permute(0, 2, 3, 1).contiguous()
        y = conv(y, "conv6")
        y = torch.cat([y, skip1], dim=3)
        y = conv(y, "conv7")
        return conv(y, "output_conv_mv"), cat1.shape[3]

    with torch.no_grad():
        y_float, cat1_ch = run(conv_float)
        q = q_input(x)
        y_i8_q, cat1_ch_i8 = run(conv_int8)

    out_scale, out_zp = qp[51], 44
    logits = out_scale * (y_i8_q.permute(0, 3, 1, 2).float() - out_zp)

    print(f"cat1 channels: float={cat1_ch} int8={cat1_ch_i8}  (expect 64)")
    print(f"int8 codes : min={y_i8_q.min():.0f} max={y_i8_q.max():.0f}")
    print(f"logits     : [{logits.min():.4f}, {logits.max():.4f}] mean={logits.mean():.4f}")
    print(f"float out  : [{y_float.min():.4f}, {y_float.max():.4f}] mean={y_float.mean():.4f}")

    a_flat, b_flat = logits.reshape(-1), y_float.reshape(-1)
    corr = torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()
    rel = ((a_flat - b_flat).abs().mean() / b_flat.abs().mean()).item()
    mx = (a_flat - b_flat).abs().max().item()
    print(f"\n== int8 path vs this checkpoint's own float path ==")
    print(f"  correlation      = {corr:.6f}")
    print(f"  mean abs rel err = {rel:.4%}")
    print(f"  max abs diff     = {mx:.4f}")
    print("  VERDICT:", "int8 pipeline tracks float" if corr > 0.99 else "INVESTIGATE")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"correlation": corr, "meanAbsRelError": rel, "maxAbsDiff": mx,
                       "logitsRange": [logits.min().item(), logits.max().item()],
                       "floatRange": [y_float.min().item(), y_float.max().item()],
                       "cat1Channels": cat1_ch_i8, "bnLayersConsumed": used_bn},
                      fh, indent=1)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
