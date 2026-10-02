#!/usr/bin/env python3
"""把 numpy 黄金参考的输出与 GPU 实现（仿真层 / 各自的 DP4A 后端）对拍。

两边都输出 .npz（key: out0=KPN, out1=temporal，int8），逐元素比较。
**判据是位精确相等**，不是「看起来差不多」——量化网络里任何一处舍入差异都会累积。

用法:
    python nss_compare.py ref.npz gpu.npz
"""
import sys

import numpy as np


def load(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.startswith("out"):
            out[k] = np.squeeze(z[k])
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) | set(b))
    rc = 0
    for k in keys:
        if k not in a or k not in b:
            print(f"{k}: 只有一边有  (ref={k in a} gpu={k in b})")
            rc = 1
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k}: 形状不一致  ref={x.shape} gpu={y.shape}")
            rc = 1
            continue
        xi, yi = x.astype(np.int32), y.astype(np.int32)
        d = np.abs(xi - yi)
        n = xi.size
        nz = int(np.count_nonzero(d))
        print(f"{k}: shape={x.shape}  ref range=[{x.min()},{x.max()}]  gpu range=[{y.min()},{y.max()}]")
        print(f"    逐元素不一致: {nz}/{n}  ({100.0 * nz / n:.4f}%)")
        print(f"    最大绝对差: {int(d.max())}   平均绝对差: {d.mean():.6f}")
        if nz:
            idx = np.unravel_index(np.argmax(d), d.shape)
            print(f"    最大差位置 {idx}: ref={int(x[idx])} gpu={int(y[idx])}")
            rc = 1
        else:
            print("    位精确一致 ✓")
    print()
    print("结果:", "不一致 ✗" if rc else "全部位精确一致 ✓")
    return rc


if __name__ == "__main__":
    sys.exit(main())
