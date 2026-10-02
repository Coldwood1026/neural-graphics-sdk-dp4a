#!/usr/bin/env python3
"""NSS 的 numpy 黄金参考实现（阶段 2）。

按 TOSA 语义逐算子复现 VGF 图，用于与 GPU 实现对拍。
算式的依据是 ai-ml-emulation-layer-for-vulkan 的 graph_op/tosa/*.comp：

  CONV2D : acc[o] = Σ_{ky,kx,ic} (in - in_zp)·(w - w_zp) + bias[o]
           越界 tap 跳过 ≡ 用 in_zp 填充（两者都贡献 0）
  RESCALE: v = acc - in_zp
           r = (v·multiplier + (1 << (shift-1))) >> shift      # 算术右移
           r = clamp(r + out_zp, int8_min, int8_max)
           out_zp = -128 时即隐含 ReLU
  RESIZE : 最近邻，scale=[n,d,n,d] → iy = (2*ry >= n) ? iy+1 : iy
  CONCAT : 通道轴拼接（NHWC 的 axis=3）
  TABLE  : lut[value + 128]

用法:
    python nss_ref.py <model>.vgf [--size 64] [--seed 0] [-o out.npz]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vgf_graph import analyze  # noqa: E402

DTYPE_BY_FORMAT = {
    "R8_SINT": (np.int8, 1), "R8_UINT": (np.uint8, 1),
    "R32_SINT": (np.int32, 4), "R32_UINT": (np.uint32, 4), "R32_SFLOAT": (np.float32, 4),
    "R16_SINT": (np.int16, 2), "R16_UINT": (np.uint16, 2), "R16_SFLOAT": (np.float16, 2),
}


def load_constants(vgf_path, ir):
    """按资源表的格式与形状把常量段切成 numpy 数组。"""
    data = open(vgf_path, "rb").read()
    out = {}
    for ent in ir["constants"]:
        cid = ent["id"]
        res = ir["resources"][cid]
        fmt = res["vk_format"]
        if fmt not in DTYPE_BY_FORMAT:
            out[cid] = None
            continue
        dtype, _ = DTYPE_BY_FORMAT[fmt]
        blob = data[ent["abs_offset"]:ent["abs_offset"] + ent["size"]]
        arr = np.frombuffer(blob, dtype=dtype)
        shape = res["shape"]
        if shape:
            n = int(np.prod(shape))
            if arr.size != n:
                arr = arr[:n] if arr.size > n else np.resize(arr, n)
            arr = arr.reshape(shape)
        out[cid] = arr
    return out


def as_int8(v):
    """量化零点在 VGF 里存的是原始字节：128 实为 int8 的 -128。"""
    v = int(v)
    return v - 256 if v > 127 else v


def as_int32(v):
    """VGF 里的 int32 字面量有时以原始 32 位字存储，需要还原为有符号。"""
    v = int(v)
    return v - (1 << 32) if v >= (1 << 31) else v


def const_array(desc, consts):
    """从 describe() 的结果取出数组（图常量或内联字面量）。"""
    if desc["kind"] == "const":
        return consts[desc["const_id"]]
    if desc["kind"] == "lit":
        v = desc["value"]
        if isinstance(v, list):
            return np.asarray([as_int32(x) for x in v], dtype=np.int32)
        return np.int32(as_int32(v))
    raise ValueError(f"不能作为常量使用: {desc}")


def conv2d(x, w, b, pad, stride, in_zp, w_zp, out_channels):
    """x: NHWC int8；w: OHWI int8；b: int32[O]；pad=[t,b,l,r]。返回 int32 NHWC。"""
    n, h, wd, cin = x.shape
    pt, pb, pl, pr = pad
    sh, sw = stride
    kh, kw = w.shape[1], w.shape[2]
    oh = (h + pt + pb - kh) // sh + 1
    ow = (wd + pl + pr - kw) // sw + 1

    # 用输入零点填充 → 越界处 (in - in_zp) == 0，与「跳过 tap」等价
    if PAD_MODE == "clamp":
        xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode="edge")
    else:
        xp = np.full((n, h + pt + pb, wd + pl + pr, cin),
                     in_zp if PAD_MODE == "zp" else 0, dtype=np.int8)
        xp[:, pt:pt + h, pl:pl + wd, :] = x

    xv = xp.astype(np.int32) - np.int32(in_zp)          # 激活去零点（越界处恰为 0）
    wv = w.astype(np.int32) - np.int32(w_zp)            # 权重点为 0

    # im2col: [n, oh, ow, kh*kw*cin]
    idx_y = np.arange(oh)[:, None] * sh + np.arange(kh)[None, :]      # (oh, kh)
    idx_x = np.arange(ow)[:, None] * sw + np.arange(kw)[None, :]      # (ow, kw)
    cols = xv[:, idx_y[:, None, :, None], idx_x[None, :, None, :], :]  # (n, oh, ow, kh, kw, cin)
    cols = cols.reshape(n, oh, ow, kh * kw * cin)

    acc = cols.astype(np.int32) @ wv.reshape(w.shape[0], -1).T.astype(np.int32)
    acc = acc.astype(np.int32)

    if b is not None and np.size(b) > 0:
        bb = np.asarray(b, dtype=np.int32).reshape(-1)
        acc = acc + (bb if bb.size > 1 else bb[0])
    return acc


ROUNDING_MODE = "single"      # single / double / trunc / away
PAD_MODE = "zp"               # zp = 填激活零点（= 跳过越界 tap）; 0 = 填 0; clamp = 边缘复制
RESIZE_OFFSET = -1            # TOSA RESIZE 的 offset（实测为 -1，不是 0！）


def rescale(acc, mult, shift, in_zp, out_zp, out_dtype=np.int8):
    """acc: int32；mult/shift 逐通道；返回 int8。

    in_zp 是 RESCALE 自己的输入零点（对 conv 输出来说实测为 0，
    因为激活零点已在卷积内部减掉），不是卷积的 input_zero_point。

    舍入方式由模块级 ROUNDING_MODE 控制，用于与官方实现对拍时逐一变体排查。
    """
    m = np.asarray(mult, dtype=np.int32).reshape(-1)
    s = np.asarray(shift, dtype=np.int32).reshape(-1)
    if m.size == 1:
        m = np.repeat(m, acc.shape[-1])
    if s.size == 1:
        s = np.repeat(s, acc.shape[-1])

    v = acc.astype(np.int64) - np.int64(in_zp)
    prod = v * m.astype(np.int64)
    s64 = s.astype(np.int64)

    if ROUNDING_MODE == "single":
        # 仿真层 applyScale：round = 1<<(shift-1)，加完做算术右移
        r = (prod + np.left_shift(np.int64(1), s64 - 1)) >> s64
    elif ROUNDING_MODE == "double":
        # doubleRound=true 且 shift>31 时额外 ±1<<30
        extra = np.where(v >= 0, np.int64(1) << 30, -(np.int64(1) << 30))
        extra = np.where(s64 > 31, extra, np.int64(0))
        r = (prod + np.left_shift(np.int64(1), s64 - 1) + extra) >> s64
    elif ROUNDING_MODE == "trunc":
        r = prod >> s64
    elif ROUNDING_MODE == "away":
        # 四舍五入、临界处远离零
        half = np.left_shift(np.int64(1), s64 - 1)
        mag = (np.abs(prod) + half) >> s64
        r = np.where(prod < 0, -mag, mag)
    else:
        raise ValueError(ROUNDING_MODE)

    r = r + np.int64(out_zp)
    info = np.iinfo(out_dtype)
    return np.clip(r, info.min, info.max).astype(out_dtype)


def resize_nearest(x, sn, sd):
    """scale = [n, d]（H/W 同比例）。RESIZE_OFFSET 为 TOSA 的 offset 操作数。"""
    n, h, w, c = x.shape
    oh = (h * sn) // sd
    ow = (w * sn) // sd
    off = RESIZE_OFFSET

    def axis_index(o, size):
        y = o * sd + off
        i = y // sn                      # idivFloor
        ry = y - i * sn
        i = i + 1 if 2 * ry >= sn else i
        return min(max(i, 0), size - 1)

    iy = np.array([axis_index(o, h) for o in range(oh)])
    ix = np.array([axis_index(o, w) for o in range(ow)])
    return x[:, iy[:, None], ix[None, :], :]


def conv2d_dp4a(x, w, b, pad, stride, in_zp, w_zp, correction):
    """与 conv2d 等价的 DP4A 形式，用于验证内核算式。

    内核里算的是 Σ q_a·q_w（激活用 in_zp 填充、权重直接读），
    而 TOSA 语义要的是 Σ (q_a - in_zp)q_w，两者相差 -in_zp·Σ_all w —— 逐输出通道常量。
    本函数走 DP4A 的算法路径，用来证明修正项推导正确。
    """
    n, h, wd, cin = x.shape
    pt, pb, pl, pr = pad
    sh, sw = stride
    kh, kw = w.shape[1], w.shape[2]
    oh = (h + pt + pb - kh) // sh + 1
    ow = (wd + pl + pr - kw) // sw + 1

    xp = np.full((n, h + pt + pb, wd + pl + pr, cin),
                 in_zp if PAD_MODE == "zp" else 0, dtype=np.int8)
    xp[:, pt:pt + h, pl:pl + wd, :] = x

    idx_y = np.arange(oh)[:, None] * sh + np.arange(kh)[None, :]
    idx_x = np.arange(ow)[:, None] * sw + np.arange(kw)[None, :]
    cols = xp[:, idx_y[:, None, :, None], idx_x[None, :, None, :], :]
    cols = cols.reshape(n, oh, ow, kh * kw * cin)

    # 内核做的是 Σ q_a·q_w（q_w 即原始权重，权重点为 0）
    acc = cols.astype(np.int32) @ w.reshape(w.shape[0], -1).T.astype(np.int32)
    acc = acc.astype(np.int32)
    acc = acc + correction.astype(np.int32).reshape(1, 1, 1, -1)   # 常量修正
    if b is not None and np.size(b) > 0:
        bb = np.asarray(b, dtype=np.int32).reshape(-1)
        acc = acc + (bb if bb.size > 1 else bb[0])
    return acc


def dp4a_correction(w, in_zp, w_zp):
    """逐输出通道的修正项：-in_zp · Σ_all w（补零前的权重和）。"""
    return (-np.int64(in_zp) * w.astype(np.int64).sum(axis=(1, 2, 3))).astype(np.int32)


def run(vgf_path, size, seed, use_dp4a=False, input_array=None, dump_dir=None):
    ir = analyze(vgf_path)
    consts = load_constants(vgf_path, ir)

    h = w = size
    if input_array is not None:
        a = np.ascontiguousarray(input_array, dtype=np.int8)
        inp = a[None] if a.ndim == 3 else a
        h, w = inp.shape[1], inp.shape[2]
    else:
        rng = np.random.default_rng(seed)
        # 输入量化：zp=-128，值域 [0, 1]，这里给个偏亮的随机激活
        inp = rng.integers(-128, 128, size=(1, h, w, 12), dtype=np.int8)
        inp = np.clip(inp.astype(np.int16) // 2 + 64, -128, 127).astype(np.int8)

    vals = {}       # op index -> ndarray
    graph_input = inp
    print(f"  输入 {inp.shape} 范围 [{inp.min()}, {inp.max()}]")

    for rec in ir["ops"]:
        i, k = rec["index"], rec["kind"]

        def get(desc):
            if desc["kind"] == "input":
                return graph_input
            if desc["kind"] == "node":
                return vals[desc["index"]]
            if desc["kind"] == "const":
                return consts[desc["const_id"]]
            raise ValueError(f"算子 {i}({k}) 的张量操作数不可执行: {desc}")

        if k == "CONV2D":
            x = get(rec["input"])
            wgt = const_array(rec["weight"], consts)
            bias = (const_array(rec["bias"], consts)
                    if rec["bias"]["kind"] != "none" else None)
            izp = rec["input_zero_point"]
            wzp = rec["weight_zero_point"]
            izp = as_int8(izp[0] if isinstance(izp, list) else izp)
            wzp = as_int8(wzp[0] if isinstance(wzp, list) else wzp)
            if use_dp4a:
                corr = dp4a_correction(wgt, izp, wzp)
                acc = conv2d_dp4a(x, wgt, bias, rec["pad"], rec["stride"], izp, wzp, corr)
            else:
                acc = conv2d(x, wgt, bias, rec["pad"], rec["stride"], izp, wzp, wgt.shape[0])
            # 注意：卷积内部已减掉激活零点，这里不再重复减
            vals[i] = acc
            tag = f"w{wgt.shape} izp={izp}"

        elif k == "RESCALE":
            acc = get(rec["input"])
            mult = const_array(rec["multiplier"], consts)
            shift = const_array(rec["shift"], consts)
            izp_r = rec["input_zero_point"]["value"]
            izp_r = as_int8(izp_r[0] if isinstance(izp_r, list) else izp_r)
            ozp = rec["output_zero_point"]["value"]
            ozp = as_int8(ozp[0] if isinstance(ozp, list) else ozp)
            vals[i] = rescale(acc, mult, shift, izp_r, ozp)
            tag = f"in_zp={izp_r} out_zp={ozp}"
        elif k == "RESIZE":
            x = get(rec["input"])
            sc = rec["scale"]
            if isinstance(sc, list) and len(sc) >= 4:
                vals[i] = resize_nearest(x, sc[0], sc[1])
            else:
                vals[i] = x
            tag = f"scale={sc}"
        elif k == "CONCAT":
            parts = [get(t) for t in rec["inputs"]]
            vals[i] = np.concatenate(parts, axis=3)
            tag = f"+{rec['axis']} {[p.shape[-1] for p in parts]}"
        elif k == "TABLE":
            x = get(rec["input"])
            lut = get(rec["table"]).reshape(-1)
            idx = x.astype(np.int32) + 128
            vals[i] = lut[np.clip(idx, 0, lut.size - 1)].astype(np.int8)
            tag = f"lut{lut.shape}"
        else:
            raise NotImplementedError(k)

        print(f"  {i:3d} {k:<8} {str(vals[i].shape):<22} dtypes={vals[i].dtype}  {tag}")

    outs = []
    for o in ir["outputs"]:
        tid = o["tensor_id"]
        found = next((r["index"] for r in ir["ops"] if r["result"] == tid), None)
        if found is None:
            for r in ir["ops"]:
                d = r.get("input")
                if isinstance(d, dict) and d.get("kind") == "node":
                    pass
            raise SystemExit(f"输出张量 {tid} 找不到生产者")
        outs.append(vals[found])

    print(f"\n  输出: {[o['shape'] for o in ir['outputs']]}")
    for o, v in zip(ir["outputs"], outs):
        print(f"    tensor_id={o['tensor_id']} shape={v.shape} range=[{v.min()}, {v.max()}]")

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        for r in ir["ops"]:
            v = vals.get(r["index"])
            if v is None:
                continue
            np.save(os.path.join(dump_dir, f"op{r['index']:02d}_{r['kind']}.npy"),
                    np.squeeze(v))
        print(f"  已导出 {len(vals)} 个中间张量到 {dump_dir}/")
    return outs, ir


def export_weights(ir, consts, outdir):
    """把图用到的常量导出成 .npy，供写 DP4A 内核时取用。"""
    os.makedirs(outdir, exist_ok=True)
    manifest = []
    for rec in ir["ops"]:
        i, k = rec["index"], rec["kind"]
        if k == "CONV2D":
            w = const_array(rec["weight"], consts)
            b = const_array(rec["bias"], consts) if rec["bias"]["kind"] != "none" else None
            izp = rec["input_zero_point"]
            izp = as_int8(izp[0] if isinstance(izp, list) else izp)
            wzp = rec["weight_zero_point"]
            wzp = as_int8(wzp[0] if isinstance(wzp, list) else wzp)
            corr = dp4a_correction(w, izp, wzp)
            np.save(os.path.join(outdir, f"op{i:02d}_conv_w.npy"), w)
            if b is not None:
                np.save(os.path.join(outdir, f"op{i:02d}_conv_b.npy"), b.reshape(-1))
            np.save(os.path.join(outdir, f"op{i:02d}_conv_dp4a_corr.npy"), corr)
            manifest.append({"op": i, "kind": "conv",
                             "weight": f"op{i:02d}_conv_w.npy", "weight_shape": list(w.shape),
                             "weight_const_id": rec["weight"].get("const_id"),
                             "weight_packed_u32": int(w.size // 4),
                             "bias": f"op{i:02d}_conv_b.npy" if b is not None else None,
                             "dp4a_correction": f"op{i:02d}_conv_dp4a_corr.npy",
                             "pad": rec["pad"], "stride": rec["stride"],
                             "input_zero_point": izp, "weight_zero_point": wzp,
                             "K": int(np.prod(w.shape[1:])),
                             "dp4a_per_output": int(np.prod(w.shape[1:]) // 4)})
        elif k == "RESCALE":
            m = const_array(rec["multiplier"], consts).reshape(-1)
            s = const_array(rec["shift"], consts).reshape(-1)
            np.save(os.path.join(outdir, f"op{i:02d}_rescale_mult.npy"), m)
            np.save(os.path.join(outdir, f"op{i:02d}_rescale_shift.npy"), s)
            ozp = rec["output_zero_point"]["value"]
            ozp = as_int8(ozp[0] if isinstance(ozp, list) else ozp)
            manifest.append({"op": i, "kind": "rescale", "channels": int(m.size),
                             "multiplier": f"op{i:02d}_rescale_mult.npy",
                             "shift": f"op{i:02d}_rescale_shift.npy",
                             "output_zero_point": ozp, "rounding": rec["rounding"],
                             "per_channel": rec["per_channel"], "scale32": rec["scale32"]})
        elif k == "TABLE":
            lut = const_array(rec["table"], consts).reshape(-1)
            np.save(os.path.join(outdir, f"op{i:02d}_lut.npy"), lut)
            manifest.append({"op": i, "kind": "table", "lut": f"op{i:02d}_lut.npy",
                             "entries": int(lut.size), "index_base": 128})
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"model": ir["vgf"], "graph": ir["graph_name"],
                   "input_shape": ir["inputs"][0]["shape"],
                   "output_shapes": [o["shape"] for o in ir["outputs"]],
                   "layers": manifest}, f, indent=2, ensure_ascii=False)
    print(f"\n已导出 {len(manifest)} 层参数到 {outdir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("--size", type=int, default=32, help="输入 H/W（必须是 8 的倍数）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output")
    ap.add_argument("--export", metavar="DIR", help="导出权重与量化参数后退出")
    ap.add_argument("--dump-all", metavar="DIR", help="导出所有中间张量")
    ap.add_argument("--input", help="用指定 npz/npy 里的 in0 作为输入，用于与 GPU 实现对拍同一份数据")
    ap.add_argument("--pad", default="zp", choices=["zp", "0", "clamp"], help="卷积 padding 填充方式")
    ap.add_argument("--resize-offset", type=int, default=-1, help="TOSA RESIZE 的 offset（官方为 -1）")
    ap.add_argument("--rounding", default="single", choices=["single", "double", "trunc", "away"],
                    help="RESCALE 舍入方式（默认 single，即仿真层 applyScale）")
    ap.add_argument("--dp4a-verify", action="store_true",
                    help="用 DP4A 算式重跑一遍并与标准路径位精确比对")
    args = ap.parse_args()
    globals()["ROUNDING_MODE"] = args.rounding
    globals()["PAD_MODE"] = args.pad
    globals()["RESIZE_OFFSET"] = args.resize_offset

    if args.export:
        ir = analyze(args.vgf)
        export_weights(ir, load_constants(args.vgf, ir), args.export)
        return

    if args.dp4a_verify:
        print(f"=== {os.path.basename(args.vgf)}  DP4A 算式验证 (size={args.size}) ===\n")
        a, _ = run(args.vgf, args.size, args.seed, use_dp4a=False)
        b, _ = run(args.vgf, args.size, args.seed, use_dp4a=True)
        ok = True
        for i, (x, y) in enumerate(zip(a, b)):
            d = np.abs(x.astype(np.int32) - y.astype(np.int32))
            same = not d.any()
            ok &= same
            print(f"  out{i}: 最大差 {int(d.max())}  {'位精确一致 ✓' if same else '不一致 ✗'}")
        print(f"\nDP4A 算式的修正项推导：{'成立 ✓' if ok else '有问题 ✗'}")
        return

    inp = None
    if args.input:
        if args.input.endswith(".npz"):
            z = np.load(args.input)
            inp = z["in0"] if "in0" in z.files else z[z.files[0]]
        else:
            inp = np.load(args.input)
        print(f"=== {os.path.basename(args.vgf)}  输入来自 {os.path.basename(args.input)} "
              f"shape={inp.shape} ===")
    else:
        print(f"=== {os.path.basename(args.vgf)}  size={args.size}x{args.size} ===")
    outs, ir = run(args.vgf, args.size, args.seed, input_array=inp, dump_dir=args.dump_all)
    if args.output:
        np.savez(args.output, **{f"out{i}": v for i, v in enumerate(outs)})
        print(f"\n已写出 {args.output}")


if __name__ == "__main__":
    main()
