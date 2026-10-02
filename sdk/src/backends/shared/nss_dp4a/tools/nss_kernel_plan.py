#!/usr/bin/env python3
"""NSS 的 DP4A 内核规划：逐层几何、算力、带宽与打包尺寸（阶段 3 的输入）。

从图 IR 做形状推导与成本建模，输出：
  · 540p（实际用 544 高，8 的倍数）下每层的张量形状、K 维、MAC 数
  · DP4A 打包尺寸（K/4 的 uint32 组数、修正项表）
  · 融合后的 dispatch 数与显存流量估算

用法:
    python nss_kernel_plan.py <model>.vgf [--h 544] [--w 960] [-o plan.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vgf_graph import analyze  # noqa: E402

MB = 1024 * 1024


def shape_of(desc, op_shapes, input_shape):
    """解析一个张量操作数描述符对应的形状。"""
    if desc["kind"] == "input":
        return input_shape
    if desc["kind"] == "node":
        return op_shapes[desc["index"]]
    if desc["kind"] == "const":
        return tuple(desc["shape"])
    raise KeyError(f"无法确定形状: {desc}")


def propagate_shape(op, op_shapes, input_shape):
    """返回该算子的输出形状 (n,h,w,c)。"""
    k = op["kind"]
    if k == "CONV2D":
        n, h, w, _ = shape_of(op["input"], op_shapes, input_shape)
        _, kh, kw, cin = op["weight"]["shape"]
        pt, pb, pl, pr = op["pad"]
        sh, sw = op["stride"]
        oh = (h + pt + pb - kh) // sh + 1
        ow = (w + pl + pr - kw) // sw + 1
        return (n, oh, ow, op["weight"]["shape"][0])
    if k in ("RESCALE", "TABLE"):
        return shape_of(op["input"], op_shapes, input_shape)
    if k == "RESIZE":
        n, h, w, c = shape_of(op["input"], op_shapes, input_shape)
        sc = op["scale"]
        if isinstance(sc, list) and len(sc) >= 4:
            return (n, h * sc[0] // sc[1], w * sc[2] // sc[3], c)
        return (n, h, w, c)
    if k == "CONCAT":
        parts = [shape_of(t, op_shapes, input_shape) for t in op["inputs"]]
        n, h, w, _ = parts[0]
        return (n, h, w, sum(p[3] for p in parts))
    raise NotImplementedError(k)


def count_valid_taps(op, in_shape):
    """精确统计该层实际发生的乘加次数（考虑越界 tap 被跳过）。"""
    k = op["kind"]
    if k != "CONV2D":
        n, h, w, c = in_shape if k != "CONCAT" else (0, 0, 0, 0)
        return 0
    n, h, w, cin = in_shape
    _, kh, kw, _ = op["weight"]["shape"]
    pt, pb, pl, pr = op["pad"]
    sh, sw = op["stride"]
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    oc = op["weight"]["shape"][0]
    total = 0
    for oy in range(oh):
        y0 = oy * sh - pt
        ny = max(0, min(kh, h - y0)) - max(0, -y0)
        for ox in range(ow):
            x0 = ox * sw - pl
            nx = max(0, min(kw, w - x0)) - max(0, -x0)
            total += max(0, ny) * max(0, nx) * cin * oc
    return total


def plan(vgf_path, H, W):
    ir = analyze(vgf_path)
    in_shape = (1, H, W, 12)
    op_shapes = {}
    rows = []

    total_macs = 0
    total_wbytes = 0
    for op in ir["ops"]:
        i, k = op["index"], op["kind"]
        out_shape = propagate_shape(op, op_shapes, in_shape)
        op_shapes[i] = out_shape
        n, oh, ow, oc = out_shape

        row = {"op": i, "kind": k, "out_shape": list(out_shape)}
        act_out = oh * ow * oc

        if k == "CONV2D":
            ishape = shape_of(op["input"], op_shapes, in_shape)
            _, kh, kw, cin = op["weight"]["shape"]
            K = kh * kw * cin
            macs = count_valid_taps(op, ishape)
            wbytes = oc * K
            corr_bytes = oc * 4
            row.update(input_shape=list(ishape), K=K, k_div4=(K % 4 == 0),
                       dp4a_per_out=K // 4, oc=oc,
                       macs=macs, weight_bytes=wbytes, correction_bytes=corr_bytes,
                       weight_dp4a_u32=oc * (K // 4))
            total_macs += macs
            total_wbytes += wbytes + corr_bytes
        else:
            row["macs"] = 0
        row["act_bytes"] = act_out
        rows.append(row)

    # 汇总
    print(f"{'op':>3} {'kind':<8} {'输出形状':<24} {'K':>6} {'K/4':>6} {'MAC':>14} {'权重B':>9}")
    print("-" * 78)
    for r in rows:
        K = r.get("K", "-")
        print(f"{r['op']:>3} {r['kind']:<8} {str(r['out_shape']):<24} {str(K):>6} "
              f"{str(r.get('dp4a_per_out', '-')):>6} {r['macs']:>14,} {r.get('weight_bytes', 0):>9,}")

    print("-" * 78)
    print(f"总 MAC: {total_macs:,}  ({total_macs/1e9:.3f} GMAC/帧)")
    print(f"权重总量: {total_wbytes/1024:.1f} KiB  (含修正确表)")
    print(f"60fps 需求: {total_macs*60/1e9:.1f} GMAC/s")

    # 激活流量：每层读输入 + 写输出（int8，但也读 int32 累加器 —— 见融合后说明）
    print("\n激活张量规模（融合后中间 int32 不落显存）:")
    for r in rows:
        if r["kind"] == "CONV2D":
            n, h, w, c = r["input_shape"]
            print(f"  op{r['op']:>2} 读 {h:>4}×{w:>4}×{c:>3} = {h*w*c/1024:>8.1f} KiB"
                  f"   写 {r['out_shape'][1]:>4}×{r['out_shape'][2]:>4}×{r['out_shape'][3]:>3}"
                  f" = {r['act_bytes']/1024:>8.1f} KiB")

    return {"input_shape": list(in_shape), "total_macs": total_macs,
            "weight_bytes": total_wbytes, "layers": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("--h", type=int, default=544, help="输入高（必须 8 的倍数，540 对齐后为 544）")
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    print(f"=== {os.path.basename(args.vgf)}  输入 {args.h}×{args.w} ===\n")
    result = plan(args.vgf, args.h, args.w)
    if args.output:
        json.dump(result, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n已写出 {args.output}")


if __name__ == "__main__":
    main()
