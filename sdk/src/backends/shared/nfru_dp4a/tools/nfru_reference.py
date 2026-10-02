#!/usr/bin/env python3
"""NFRU v1 autoencoder: exact PyTorch reference, weight cross-check, golden validation.

This mirrors `NFRUAutoEncoder.forward` from Arm's Neural Graphics Model Gym
(`src/ng_model_gym/usecases/nfru/model/nfru_v1_nn.py`) and the constants from
`.../model/constants.py`. It exists to:

  1. prove the VGF-extracted weight shapes correspond 1:1 to the PyTorch module
     (independent confirmation that the reverse-engineered architecture is right);
  2. compute the exact convolution geometry (kernel, stride, padding, channels)
     needed by a GPU port, instead of inferring it;
  3. validate a forward pass against Arm's own golden tensors;
  4. emit the int8 weights repacked for dp4a plus the per-layer requantisation
     parameters.

Usage:
    python nfru_reference.py --weights ../../fru/nfru_v1_fp32.pt \
        [--input golden/autoencoder_input_golden.bin] \
        [--expected golden/autoencoder_output_golden.bin] \
        [--emit-weights <dir>]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- constants, copied from ng_model_gym/usecases/nfru/model/constants.py ----
IN_CH, HIDDEN_CH, BRANCH_CH, CONCAT_CH, OUT_CH = 16, 32, 16, 64, 4
K1, K3, K5, K7 = 1, 3, 5, 7
PAD_NONE, PAD_3X3, PAD_5X5, PAD_7X7 = 0, (1, 1), (2, 2), (3, 3)
STRIDE_1, STRIDE_2 = (1, 1), (2, 2)


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU, matching model gym's ConvBlock default."""

    def __init__(self, cin, cout, k, padding=(1, 1), stride=(1, 1),
                 batch_norm=True, activation="relu"):
        super().__init__()
        self.conv2d = nn.Conv2d(cin, cout, kernel_size=k, padding=padding, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(cout, momentum=0.1) if batch_norm else nn.Identity()
        self.act = {"relu": nn.ReLU, "": nn.Identity}[activation]()

    def forward(self, x):
        return self.act(self.bn(self.conv2d(x)))


class NFRUAutoEncoder(nn.Module):
    """Exactly the module tree in ng_model_gym's nfru_v1_nn.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(IN_CH, HIDDEN_CH, K3, PAD_3X3)
        self.conv2 = ConvBlock(HIDDEN_CH, BRANCH_CH, K5, PAD_5X5, STRIDE_1)
        self.skip1_conv = ConvBlock(BRANCH_CH, BRANCH_CH, K3)
        self.conv3 = ConvBlock(BRANCH_CH, BRANCH_CH, K3)
        self.conv5 = ConvBlock(BRANCH_CH, BRANCH_CH, K5, PAD_5X5, STRIDE_2)
        self.conv5a = ConvBlock(BRANCH_CH, BRANCH_CH, K1, PAD_NONE)
        self.conv5b = ConvBlock(BRANCH_CH, BRANCH_CH, K3)
        self.conv5c = ConvBlock(BRANCH_CH, BRANCH_CH, K7, PAD_7X7)
        self.conv5c_1 = ConvBlock(BRANCH_CH, BRANCH_CH, K7, PAD_7X7)
        self.conv5d = ConvBlock(BRANCH_CH, BRANCH_CH, K7, PAD_7X7)
        self.conv5d_1 = ConvBlock(BRANCH_CH, BRANCH_CH, K7, PAD_7X7)
        self.conv5d_2 = ConvBlock(BRANCH_CH, BRANCH_CH, K7, PAD_7X7)
        self.conv5e = ConvBlock(CONCAT_CH, BRANCH_CH, K1, PAD_NONE)
        self.upsample1 = nn.Upsample(mode="nearest", scale_factor=2)
        self.conv6 = ConvBlock(BRANCH_CH, BRANCH_CH, K3, PAD_3X3)
        self.conv7 = ConvBlock(HIDDEN_CH, BRANCH_CH, K3)
        self.output_conv_mv = nn.Conv2d(BRANCH_CH, OUT_CH, kernel_size=K5, padding=PAD_5X5)

    def forward(self, x):
        x = self.conv1(x)          # E-0
        x = self.conv2(x)
        x = self.conv3(x)          # E-2
        skip1 = self.skip1_conv(x)
        b = self.conv5(x)          # B, stride 2

        xa = self.conv5a(b)
        xb = self.conv5b(b)
        xc = self.conv5c_1(self.conv5c(b))
        xd = self.conv5d_2(self.conv5d_1(self.conv5d(b)))

        x = torch.cat([xa, xb, xc, xd], dim=1)   # 64 channels
        x = self.conv5e(x)                       # 1x1, 64 -> 16
        x = self.upsample1(x)                    # nearest x2
        x = self.conv6(x)
        x = torch.cat([x, skip1], dim=1)         # 32 channels
        x = self.conv7(x)
        return self.output_conv_mv(x)            # [N,4,H,W]


# Layer order along the execution path -> weight tensor in the VGF.
# (name, kernel, stride, padding, in_ch, out_ch) computed from the module tree.
LAYER_SPEC = [
    # name          k  stride  pad      cin cout
    ("conv1",       3, 1, 1,  16, 32),
    ("conv2",       5, 1, 2,  32, 16),
    ("conv3",       3, 1, 1,  16, 16),
    ("skip1_conv",  3, 1, 1,  16, 16),
    ("conv5",       5, 2, 2,  16, 16),
    ("conv5a",      1, 1, 0,  16, 16),
    ("conv5b",      3, 1, 1,  16, 16),
    ("conv5c",      7, 1, 3,  16, 16),
    ("conv5c_1",    7, 1, 3,  16, 16),
    ("conv5d",      7, 1, 3,  16, 16),
    ("conv5d_1",    7, 1, 3,  16, 16),
    ("conv5d_2",    7, 1, 3,  16, 16),
    ("conv5e",      1, 1, 0,  64, 16),
    ("conv6",       3, 1, 1,  16, 16),
    ("conv7",       3, 1, 1,  32, 16),
    ("output_conv_mv", 5, 1, 2, 16, 4),
]


def fold_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into the preceding convolution."""
    std = torch.sqrt(bn_var + eps)
    scale = bn_w / std
    w = conv_w * scale.reshape(-1, 1, 1, 1)
    b = (conv_b - bn_mean) * scale + bn_b
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input")
    ap.add_argument("--expected")
    ap.add_argument("--emit-weights")
    ap.add_argument("--vh", type=int, default=270, help="height for a synthetic run")
    ap.add_argument("--vw", type=int, default=480, help="width for a synthetic run")
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    msd = sd.get("model_state_dict", sd)
    P = "network.auto_encoder."

    # ---- 1. cross-check the module tree against the checkpoint --------------
    print("== weight shape cross-check ==")
    import re
    found = {}
    for k, v in msd.items():
        if not k.startswith(P):
            continue
        # ConvBlock weights are "<name>.conv2d.weight"; the final output head is a
        # plain nn.Conv2d, so its key is "<name>.weight".
        m = re.match(P + r"([a-z0-9_]+)\.(?:conv2d\.)?weight$", k, re.I)
        if m:
            found[m.group(1)] = tuple(v.shape)
    for name, k, stride, pad, cin, cout in LAYER_SPEC:
        exp = (cout, cin, k, k)
        got = found.get(name)
        status = "OK" if got == exp else f"MISMATCH got={got} want={exp}"
        print(f"  {name:<16} {str(cout):>3} x {cin:<3} {k}x{k}  {status}")

    # ---- 2. build the model, load, fold BN ---------------------------------
    model = NFRUAutoEncoder()
    missing, unexpected = model.load_state_dict(msd, strict=False)
    real_missing = [m for m in missing if m.startswith("network.auto_encoder")]
    if real_missing:
        print(f"\nWARNING: {len(real_missing)} missing keys, e.g. {real_missing[:5]}")
    model.eval()

    # ---- 3. forward pass ---------------------------------------------------
    if args.input:
        blob = torch.load(args.input, map_location="cpu", weights_only=False)
        print(f"\n== golden input loaded: type={type(blob)}")
        if isinstance(blob, dict):
            print("   keys:", list(blob.keys())[:10])
            x = None
            for key in ("input", "x", "tensor", "data"):
                if key in blob and torch.is_tensor(blob[key]):
                    x = blob[key]
                    break
            if x is None:
                tensors = [v for v in blob.values() if torch.is_tensor(v)]
                x = tensors[0] if tensors else None
        elif torch.is_tensor(blob):
            x = blob
        else:
            x = None
        if x is None:
            print("   could not find a tensor in the golden input")
            return 1
        print(f"   input tensor: shape={tuple(x.shape)} dtype={x.dtype}")
        x = x.float()
        if x.dim() == 4 and x.shape[1] != IN_CH and x.shape[-1] == IN_CH:
            print("   note: input looks NHWC, converting to NCHW")
            x = x.permute(0, 3, 1, 2).contiguous()
    else:
        print(f"\n== synthetic input {args.vh}x{args.vw} ==")
        x = torch.randn(1, IN_CH, args.vh, args.vw)

    with torch.no_grad():
        y = model(x)
    print(f"== forward ok: output shape={tuple(y.shape)} "
          f"min={y.min().item():.4f} max={y.max().item():.4f}")

    if args.expected:
        exp = torch.load(args.expected, map_location="cpu", weights_only=False)
        if isinstance(exp, dict):
            tensors = [v for v in exp.values() if torch.is_tensor(v)]
            exp = tensors[0] if tensors else None
        if exp is not None:
            exp = exp.float()
            if exp.dim() == 4 and exp.shape[1] != OUT_CH and exp.shape[-1] == OUT_CH:
                exp = exp.permute(0, 3, 1, 2).contiguous()
            print(f"\n== golden comparison: expected shape={tuple(exp.shape)}")
            if tuple(exp.shape) != tuple(y.shape):
                print("   SHAPE MISMATCH - cannot compare directly")
            else:
                diff = (y - exp).abs()
                print(f"   max abs diff  = {diff.max().item():.6e}")
                print(f"   mean abs diff = {diff.mean().item():.6e}")
                print("   RESULT:", "MATCH" if diff.max().item() < 1e-4 else "MISMATCH")

    # ---- 4. emit dp4a-ready int8 weights + requant params ------------------
    if args.emit_weights:
        os.makedirs(args.emit_weights, exist_ok=True)
        manifest = {"layers": [], "note": "int8 symmetric weights, per-tensor; "
                                          "BN folded into conv; weight layout [K,K,CIN,COUT]"}
        total = 0
        for name, k, stride, pad, cin, cout in LAYER_SPEC:
            w = msd[P + name + ".conv2d.weight"]
            b = msd[P + name + ".conv2d.bias"]
            bnw = msd.get(P + name + ".bn.weight")
            if bnw is not None:
                w, b = fold_bn(w, b, bnw, msd[P + name + ".bn.bias"],
                               msd[P + name + ".bn.running_mean"],
                               msd[P + name + ".bn.running_var"])
            # per-tensor symmetric int8 quantisation of the folded weights
            amax = w.abs().max().item()
            scale = amax / 127.0 if amax > 0 else 1.0
            q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
            # repack [COUT,CIN,K,K] -> [K,K,CIN,COUT] for dp4a
            qp = q.permute(2, 3, 1, 0).contiguous()
            fn = os.path.join(args.emit_weights, f"{name}.int8")
            qp.numpy().tofile(fn)
            total += qp.numel()
            manifest["layers"].append({
                "name": name, "kernel": k, "stride": stride, "padding": pad,
                "inChannels": cin, "outChannels": cout,
                "weightScale": scale, "bytes": int(qp.numel()),
                "file": os.path.basename(fn),
                "biasFolded": [float(v) for v in b.flatten()[:4]],
            })
            print(f"  emit {name:<16} {tuple(qp.shape)} -> {fn}")
        manifest["totalWeightBytes"] = total
        with open(os.path.join(args.emit_weights, "dp4a_weights.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"  total int8 weight bytes = {total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
