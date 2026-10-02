#!/usr/bin/env python3
"""AOT 生成器：把 VGF 烘焙成 C++ 头文件，供 nss_dp4a.dll 直接使用。

设计原则 —— DLL 里不做任何 VGF 解析，也不绑定固定分辨率：
  实测发现**权重与量化参数完全与输入尺寸无关**，只有张量形状依赖输入 H/W。
  所以这里烘焙「尺寸无关」的部分，C++ 侧运行时做形状推导（十几行算术）。
  结果：DLL 支持任意 8 的倍数尺寸，且不需要重新烘焙。

产物（均在 --outdir 下）：
  nss_model_data_<name>.h   权重/bias/修正项/multiplier/shift/LUT
  nss_model_plan_<name>.h   逐层几何参数（尺寸无关）+ 缓冲拓扑
  nss_spirv_embed.h         三个内核的 SPIR-V 字节码

用法:
    python bake_model.py <model.vgf> --name high -o ../nss-dll/generated --shaders ../nss-vk/shaders
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nss-tools"))
from vgf_graph import analyze                                    # noqa: E402
from nss_ref import load_constants, const_array, as_int8         # noqa: E402


# ------------------------------------------------------------------ 工具
def c_array_u8(name, arr, per_line=16):
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1)
    lines = [f"static const unsigned char {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        lines.append("    " + "".join(f"0x{b:02x}," for b in flat[i:i + per_line]))
    lines.append("};")
    return "\n".join(lines)


def c_array_i32(name, arr, per_line=12):
    flat = np.asarray(arr, dtype=np.int32).reshape(-1)
    lines = [f"static const int {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        lines.append("    " + ", ".join(f"{int(v)}" for v in flat[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def c_array_u32(name, arr, per_line=12):
    flat = np.asarray(arr, dtype=np.uint32).reshape(-1)
    lines = [f"static const unsigned int {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        lines.append("    " + ", ".join(f"0x{int(v):08x}u" for v in flat[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


# ------------------------------------------------------------------ 规划
def build_plan(vgf_path):
    """输出「尺寸无关」的层表 + 缓冲拓扑。形状留到 C++ 侧推导。"""
    ir = analyze(vgf_path)
    consts = load_constants(vgf_path, ir)
    ops = ir["ops"]

    # --- 融合：conv + rescale (+ table) ---
    fused, skip = {}, set()
    for r in ops:
        if r["kind"] != "CONV2D":
            continue
        j = next((s["index"] for s in ops
                  if s["kind"] == "RESCALE" and s["input"]["kind"] == "node"
                  and s["input"]["index"] == r["index"]), None)
        k = None
        if j is not None:
            k = next((t["index"] for t in ops
                      if t["kind"] == "TABLE" and t["input"]["kind"] == "node"
                      and t["input"]["index"] == j), None)
        fused[r["index"]] = (j, k)
        if j is not None:
            skip.add(j)
        if k is not None:
            skip.add(k)

    # --- 缓冲拓扑 ---
    # 按「产生它的 dispatch」分配，而不是按算子下标。
    # 融合组（conv + rescale [+ table]）只占一个缓冲，组内成员都指向它；
    # 否则被融合的 RESCALE/TABLE 会白白占掉槽位、留下空洞。
    buf_of = {}                       # op index -> buffer index
    buffers = []
    group_of = {}                     # 融合组主算子 -> 组内全部算子下标
    for ci, (j, k) in fused.items():
        members = [ci] + ([j] if j is not None else []) + ([k] if k is not None else [])
        group_of[ci] = members
    claimed = set()
    for r in ops:
        i = r["index"]
        if i in claimed:
            continue
        if i in group_of:                       # 融合组主算子
            buffers.append(None)
            bi = len(buffers) - 1
            for m in group_of[i]:
                buf_of[m] = bi
                claimed.add(m)
        else:                                   # 独立 dispatch（resize / concat）
            buffers.append(None)
            buf_of[i] = len(buffers) - 1
            claimed.add(i)

    def ref(rec):
        if rec["kind"] == "node":
            return buf_of[rec["index"]]
        if rec["kind"] == "input":
            return -1
        return -2

    layers = []
    const_blobs = {}

    for r in ops:
        i = r["index"]
        if i in skip:
            continue
        k = r["kind"]

        if k == "CONV2D":
            j, tk = fused[i]
            rs = ops[j]
            wgt = const_array(r["weight"], consts)
            bias = const_array(r["bias"], consts)
            izp = r["input_zero_point"]
            izp = as_int8(izp[0] if isinstance(izp, list) else izp)
            corr = (-np.int64(izp) * wgt.astype(np.int64).sum(axis=(1, 2, 3)))
            bias_i64 = np.asarray(bias, dtype=np.int64).reshape(-1)
            if bias_i64.size == 1:
                bias_i64 = np.repeat(bias_i64, wgt.shape[0])
            bc = (bias_i64 + corr).astype(np.int32)

            mult = np.ascontiguousarray(const_array(rs["multiplier"], consts),
                                        dtype=np.int32).reshape(-1)
            shv = np.ascontiguousarray(const_array(rs["shift"], consts),
                                       dtype=np.int32).reshape(-1)
            ozp = rs["output_zero_point"]["value"]
            ozp = as_int8(ozp[0] if isinstance(ozp, list) else ozp)
            lut = (np.ascontiguousarray(const_array(ops[tk]["table"], consts),
                                        dtype=np.int32).reshape(-1)
                   if tk is not None else np.zeros(256, dtype=np.int32))

            tag = f"op{i:02d}"
            const_blobs[f"{tag}_w"] = ("u32", np.ascontiguousarray(wgt).view(np.uint32).reshape(-1))
            const_blobs[f"{tag}_bc"] = ("i32", bc)
            const_blobs[f"{tag}_m"] = ("i32", mult)
            const_blobs[f"{tag}_s"] = ("i32", shv)
            const_blobs[f"{tag}_lut"] = ("i32", lut)

            pt, pb, pl, pr = r["pad"]
            sh, sw = r["stride"]
            layers.append({
                "kind": "conv", "tag": tag,
                "in_buf": ref(r["input"]), "out_buf": buf_of[i],
                "cin": wgt.shape[3], "cout": wgt.shape[0],
                "kh": wgt.shape[1], "kw": wgt.shape[2],
                "pad_t": pt, "pad_l": pl, "stride_h": sh, "stride_w": sw,
                "out_zp": ozp & 0xFFFFFFFF, "has_lut": 1 if tk is not None else 0,
                "w_words": const_blobs[f"{tag}_w"][1].size,
                "mult_count": mult.size,
            })

        elif k == "RESIZE":
            sc = r["scale"]
            layers.append({
                "kind": "resize", "tag": None,
                "in_buf": ref(r["input"]), "out_buf": buf_of[i],
                "scale_n": sc[0], "scale_d": sc[1],
            })

        elif k == "CONCAT":
            parts = r["inputs"]
            # 记录各输入的缓冲与其通道数（通道数尺寸无关，可直接烘焙）
            ins = []
            for t in parts:
                tid = t["index"]
                c = None
                for rr in ops:
                    if rr["index"] == tid and rr["kind"] == "CONV2D":
                        c = rr["weight"]["shape"][0]
                if c is None:
                    # 输入来自 resize/concat，需回溯到最终通道数
                    c = _channels_of(ops, tid)
                ins.append({"buf": ref(t), "c": c})
            layers.append({
                "kind": "concat", "tag": None,
                "in_buf": ins[0]["buf"], "out_buf": buf_of[i],
                "n_inputs": len(ins),
                "inputs": ins,
            })

    outputs = []
    for o in ir["outputs"]:
        tid = o["tensor_id"]
        idx = next(r["index"] for r in ops if r["result"] == tid)
        outputs.append({"buf": buf_of[idx]})

    return {
        "ops_total": len(ops), "disp_total": len(layers),
        "buffers": buffers, "layers": layers,
        "const_blobs": const_blobs, "outputs": outputs,
        "input_c": 12,
    }


def _channels_of(ops, idx):
    """回溯某个算子的输出通道数（尺寸无关）。"""
    r = ops[idx]
    if r["kind"] == "CONV2D":
        return r["weight"]["shape"][0]
    if r["kind"] in ("RESCALE", "TABLE", "RESIZE"):
        return _channels_of(ops, r["input"]["index"])
    if r["kind"] == "CONCAT":
        return sum(_channels_of(ops, t["index"]) for t in r["inputs"])
    raise NotImplementedError(r["kind"])


# ------------------------------------------------------------------ 输出
def emit_common(out_path):
    """共享类型定义。两个模型的层表都用同一套类型，否则它们是不同类型、无法互转。"""
    L = ["// 自动生成，请勿手改。所有烘焙模型共用的类型定义。",
         "#pragma once", "",
         "namespace nss_baked {", "",
         "enum KernelKind { KERNEL_CONV = 0, KERNEL_RESIZE = 1, KERNEL_CONCAT = 2 };",
         "",
         "struct OutDesc { int buf; };",
         "",
         "struct LayerDesc {",
         "    int kind;              // KernelKind",
         "    int inBuf;             // 缓冲下标，-1 = 图输入",
         "    int outBuf;",
         "    int cin, cout;         // 通道数（conv/resize）",
         "    int kh, kw;            // 卷积核",
         "    int padT, padL;",
         "    int strideH, strideW;",
         "    unsigned int outZp;    // RESCALE 输出零点（-128 的原始字节 = ReLU 标记）",
         "    int hasLut;            // 是否走 sigmoid 查表",
         "    int scaleN, scaleD;    // resize 比例",
         "    int nInputs;           // concat 输入数",
         "    const int *concatBufs; // concat 各输入的缓冲下标",
         "    const int *concatChans;",
         "    // 常量缓冲（仅 conv）",
         "    const unsigned int *w;",
         "    const int *bc, *mult, *shift, *lut;",
         "    unsigned int wWords, multCount;",
         "};",
         "",
         "}  // namespace nss_baked",
         ""]
    txt = "\n".join(L)
    open(out_path, "w", encoding="utf-8").write(txt)
    return len(txt)


def emit_data(plan, model_name, out_path, ns):
    L = [f"// 自动生成，请勿手改。模型: {model_name}",
         "// 由 nss-tools/bake_model.py 从 VGF 烘焙而来。",
         "// 内容与输入分辨率无关 —— 形状在运行时推导。",
         "#pragma once", "",
         '#include "nss_model_common.h"',
         "",
         f"namespace {ns} {{", ""]
    for name, (kind, data) in sorted(plan["const_blobs"].items()):
        L.append(c_array_u32(name, data) if kind == "u32" else c_array_i32(name, data))
        L.append("")

    L.append(f"static const nss_baked::OutDesc kOutputs[{len(plan['outputs'])}] = {{")
    for o in plan["outputs"]:
        L.append(f"    {{ {o['buf']} }},")
    L.append("};")

    L.append(f"static const unsigned int kOpsTotal = {plan['ops_total']}u;")
    L.append(f"static const unsigned int kDispatchTotal = {plan['disp_total']}u;")
    L.append(f"static const unsigned int kBufferCount = {len(plan['buffers'])}u;")
    L.append(f"static const unsigned int kOutputCount = {len(plan['outputs'])}u;")
    L.append(f"static const unsigned int kInputChannels = {plan['input_c']}u;")
    L.append("")
    L.append(f"}}  // namespace {ns}")
    L.append("")
    txt = "\n".join(L)
    open(out_path, "w", encoding="utf-8").write(txt)
    return len(txt)


def emit_plan(plan, model_name, out_path, ns):
    L = [f"// 自动生成，请勿手改。模型: {model_name}",
         "// 逐层几何参数（尺寸无关）。张量形状由 nss_dp4a_model.cpp 在运行时推导。",
         "#pragma once", "",
         '#include "nss_model_common.h"',
         "",
         f"namespace {ns} {{", "",
         "using nss_baked::LayerDesc;",
         "using nss_baked::KernelKind;",
         "using nss_baked::KERNEL_CONV;",
         "using nss_baked::KERNEL_RESIZE;",
         "using nss_baked::KERNEL_CONCAT;", ""]

    used = sorted(d["tag"] for d in plan["layers"] if d["tag"])
    for tag in used:
        L.append(f"extern const unsigned int {tag}_w[];")
        L.append(f"extern const int {tag}_bc[];")
        L.append(f"extern const int {tag}_m[];")
        L.append(f"extern const int {tag}_s[];")
        L.append(f"extern const int {tag}_lut[];")
    L.append("")

    # 层描述类型定义在 nss_model_common.h（两个模型共用，否则是不同类型无法互转）

    # concat 的输入表
    for n, d in enumerate(plan["layers"]):
        if d["kind"] == "concat":
            bufs = ", ".join(str(x["buf"]) for x in d["inputs"])
            chans = ", ".join(str(x["c"]) for x in d["inputs"])
            L.append(f"static const int kConcatBufs_{n}[{d['n_inputs']}] = {{ {bufs} }};")
            L.append(f"static const int kConcatChans_{n}[{d['n_inputs']}] = {{ {chans} }};")
    L.append("")

    kmap = {"conv": "KERNEL_CONV", "resize": "KERNEL_RESIZE", "concat": "KERNEL_CONCAT"}
    L.append(f"static const nss_baked::LayerDesc kLayers[{len(plan['layers'])}] = {{")
    for n, d in enumerate(plan["layers"]):
        kk = kmap[d["kind"]]
        # 用 C++20 指定初始化器：字段名显式写出，与 LayerDesc 的声明顺序解耦，
        # 以后往结构体里插字段不会再静默错位。
        f = [f".kind = {kk}", f".inBuf = {d['in_buf']}", f".outBuf = {d['out_buf']}"]
        if d["kind"] == "conv":
            t = d["tag"]
            f += [f".cin = {d['cin']}", f".cout = {d['cout']}",
                  f".kh = {d['kh']}", f".kw = {d['kw']}",
                  f".padT = {d['pad_t']}", f".padL = {d['pad_l']}",
                  f".strideH = {d['stride_h']}", f".strideW = {d['stride_w']}",
                  f".outZp = {d['out_zp']}u", f".hasLut = {d['has_lut']}",
                  f".w = {t}_w", f".bc = {t}_bc", f".mult = {t}_m",
                  f".shift = {t}_s", f".lut = {t}_lut",
                  f".wWords = {d['w_words']}u", f".multCount = {d['mult_count']}u"]
        elif d["kind"] == "resize":
            f += [f".scaleN = {d['scale_n']}", f".scaleD = {d['scale_d']}"]
        else:
            f += [f".nInputs = {d['n_inputs']}",
                  f".concatBufs = kConcatBufs_{n}",
                  f".concatChans = kConcatChans_{n}"]
        L.append("    { " + ", ".join(f) + " },")
    L.append("};")
    L.append("")
    L.append(f"}}  // namespace {ns}")
    L.append("")
    txt = "\n".join(L)
    open(out_path, "w", encoding="utf-8").write(txt)
    return len(txt)


def emit_spirv(out_path, spv_paths):
    L = ["// 自动生成，请勿手改。内嵌 SPIR-V 字节码。", "#pragma once", "",
         "namespace nss_model {", ""]
    for name, path in spv_paths.items():
        words = list(struct.unpack(f"<{os.path.getsize(path) // 4}I", open(path, "rb").read()))
        L.append(f"static const unsigned int kSpirv_{name}[{len(words)}] = {{")
        for i in range(0, len(words), 8):
            L.append("    " + ", ".join(f"0x{w:08x}u" for w in words[i:i + 8]) + ",")
        L.append("};")
        L.append(f"static const unsigned int kSpirv_{name}_words = {len(words)}u;")
        L.append("")
    L.append("}  // namespace nss_model")
    L.append("")
    txt = "\n".join(L)
    open(out_path, "w", encoding="utf-8").write(txt)
    return len(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("--name", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--shaders", help="着色器目录（含 *.spv）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plan = build_plan(args.vgf)

    # 两个模型各自一个命名空间，避免同一编译单元里符号冲突
    ns = "nss_model" if args.name == "high" else f"nss_model_{args.name}"

    kinds = {}
    for d in plan["layers"]:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    total_bytes = sum(d.nbytes for _, d in plan["const_blobs"].values())

    print(f"模型 {args.name}: {plan['ops_total']} 算子 -> {plan['disp_total']} 次 dispatch")
    print(f"  命名空间 {ns}")
    print(f"  内核分布 {kinds}，缓冲 {len(plan['buffers'])}，输出 {len(plan['outputs'])}")
    print(f"  常量数据 {total_bytes / 1024:.1f} KiB")

    p0 = os.path.join(args.outdir, "nss_model_common.h")
    if not os.path.exists(p0):
        emit_common(p0)
        print(f"  -> nss_model_common.h")

    p1 = os.path.join(args.outdir, f"nss_model_data_{args.name}.h")
    p2 = os.path.join(args.outdir, f"nss_model_plan_{args.name}.h")
    print(f"  -> {os.path.basename(p1)} ({emit_data(plan, args.name, p1, ns) / 1024:.1f} KiB)")
    print(f"  -> {os.path.basename(p2)} ({emit_plan(plan, args.name, p2, ns) / 1024:.1f} KiB)")

    if args.shaders:
        spvs = {}
        for n in ("conv_rq", "resize2x", "concat_copy"):
            p = os.path.join(args.shaders, f"{n}.spv")
            if not os.path.exists(p):
                sys.exit(f"缺少着色器 {p}")
            spvs[n] = p
        p3 = os.path.join(args.outdir, "nss_spirv_embed.h")
        print(f"  -> nss_spirv_embed.h ({emit_spirv(p3, spvs) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
