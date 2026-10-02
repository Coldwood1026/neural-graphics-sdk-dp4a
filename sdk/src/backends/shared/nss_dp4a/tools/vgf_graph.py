#!/usr/bin/env python3
"""VGF → 计算图 IR（阶段 1 收官）。

直接从 .vgf 中取出 SPIR-V 图模块并重建计算图，把每个张量操作数解析成
「常量 / 图输入 / 上游算子 / 字面量」四类之一，并解析出所有标量与向量字面量的真实数值。

不再依赖 Model_Parser.exe。用法:

    python vgf_graph.py <model>.vgf [-o ir.json] [--verbose]
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vgf_extract import FlatBuffer, parse_constants, parse_resources, SECTION_NAMES  # noqa: E402

TOSA = {
    0: "ARGMAX", 1: "AVG_POOL2D", 2: "CONV2D", 3: "CONV3D", 4: "DEPTHWISE_CONV2D",
    5: "FFT2D", 6: "MATMUL", 7: "MAX_POOL2D", 8: "RFFT2D", 9: "TRANSPOSE_CONV2D",
    10: "CLAMP", 11: "ERF", 12: "SIGMOID", 13: "TANH", 14: "ADD",
    15: "ARITHMETIC_RIGHT_SHIFT", 27: "MUL", 29: "SUB", 30: "TABLE", 31: "ABS",
    54: "CONCAT", 55: "PAD", 56: "RESHAPE", 58: "SLICE", 60: "TRANSPOSE",
    63: "RESIZE", 64: "CAST", 65: "RESCALE",
}
ROUNDING = {0: "UNKNOWN", 1: "SINGLE_ROUND", 2: "INEXACT_ROUND", 3: "DOUBLE_ROUND"}
RESIZE_MODE = {0: "UNKNOWN", 1: "NEAREST", 2: "BILINEAR"}

# 图扩展算子 opcode —— 来自 SPV_ARM_graph 官方规范（Khronos SPIRV-Registry）
#   OpGraphConstantARM    <Result Type> <Result id> <GraphConstantID(literal)>
#   OpGraphEntryPointARM  <Graph> <Name(string)> <Interface ids...>
#   OpGraphARM            <Result Type> <Result id>
#   OpGraphInputARM       <Result Type> <Result id> <InputIndex(id)> [ElementIndex(id)]
#   OpGraphSetOutputARM   <Value(id)> <OutputIndex(id)> [ElementIndex(id)]   ← 无结果 id
#   OpGraphEndARM         (无操作数)
#   OpTypeGraphARM        <Result id> <NumInputs> <InOut types...>            ← 无结果类型
OP_GRAPH_CONSTANT = 4181
OP_GRAPH_ENTRYPOINT = 4182
OP_GRAPH_ARM = 4183
OP_GRAPH_INPUT = 4184
OP_GRAPH_SET_OUTPUT = 4185
OP_GRAPH_END = 4186
OP_TYPE_GRAPH = 4190


class Instr:
    __slots__ = ("opcode", "ops")

    def __init__(self, opcode, ops):
        self.opcode = opcode
        self.ops = ops

    @property
    def result_type(self):
        return self.ops[0] if self.ops else None

    @property
    def result_id(self):
        return self.ops[1] if len(self.ops) > 1 else None

    def in_operand(self, i):
        k = i + 2
        return self.ops[k] if k < len(self.ops) else None


def parse_spirv(words):
    ins = []
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        if wc == 0:
            break
        ins.append(Instr(words[i] & 0xFFFF, words[i + 1:i + wc]))
        i += wc
    return ins


def decode_name(ops, start):
    """从 ops[start:] 解码一个以 NUL 结尾的字符串，返回 (字符串, 占用的 word 数)。"""
    raw, n = bytearray(), 0
    for w in ops[start:]:
        n += 1
        for c in struct.pack("<I", w):
            if c == 0:
                return raw.decode("utf-8", "replace"), n
            raw.append(c)
    return raw.decode("utf-8", "replace"), n


def build_value_table(ins):
    """收集字面量常量与整数类型（用于按有符号性还原负值）。"""
    int_types = {}
    for x in ins:
        if x.opcode == 21 and len(x.ops) >= 3:               # OpTypeInt
            int_types[x.ops[0]] = (x.ops[1], x.ops[2])       # (width, signedness)

    def as_int(word, type_id):
        info = int_types.get(type_id)
        if not info:
            return word
        width, signed = info
        if signed and width <= 32 and word >= (1 << 31):
            return word - (1 << 32)
        return word

    scalars, composites, nulls = {}, {}, set()
    for x in ins:
        if x.opcode == 43 and len(x.ops) >= 3:          # OpConstant
            scalars[x.result_id] = as_int(x.ops[2], x.ops[0])
        elif x.opcode == 41:
            scalars[x.result_id] = 1                    # OpConstantTrue
        elif x.opcode == 42:
            scalars[x.result_id] = 0                    # OpConstantFalse
        elif x.opcode == 46:
            nulls.add(x.result_id)
        elif x.opcode == 44 and len(x.ops) >= 3:        # OpConstantComposite
            composites[x.result_id] = x.ops[2:]
    return scalars, composites, nulls


def resolve_value(vid, scalars, composites, nulls, depth=0):
    """递归求一个字面量的值：标量返回 int，复合返回 list。"""
    if depth > 8:
        return f"<deep {vid}>"
    if vid in scalars:
        return scalars[vid]
    if vid in nulls:
        return 0
    if vid in composites:
        return [resolve_value(c, scalars, composites, nulls, depth + 1) for c in composites[vid]]
    return f"%{vid}"


def analyze(vgf_path, verbose=False):
    data = open(vgf_path, "rb").read()
    secs = [struct.unpack_from("<QQ", data, 0x10 + i * 16) for i in range(4)]

    # --- modules: 取出图模块的 SPIR-V ---
    off, size = secs[0]
    fb = FlatBuffer(data, off, off + size)
    root = fb.root()
    mods = fb.vec_table(root, 0, lambda p: p)
    spirv = None
    module_info = []
    for m in mods:
        code_tbl = fb.indirect(m, 4)
        words = fb.vec_scalar(code_tbl, 0, "I") if code_tbl is not None else []
        module_info.append({"name": fb.string(m, 1), "entry_point": fb.string(m, 2),
                            "type": "GRAPH" if (fb.scalar(m, 0, "B") or 0) == 1 else "COMPUTE",
                            "words": len(words)})
        if words and spirv is None:
            spirv = words
    if spirv is None:
        sys.exit("modules 段里没有找到 SPIR-V")

    # --- resources: 张量表 ---
    off, size = secs[2]
    fb = FlatBuffer(data, off, off + size)
    resources = parse_resources(fb)
    n_input = sum(1 for r in resources if r["category"] == "INPUT")

    # --- constants: 数据表 ---
    off, size = secs[3]
    consts = parse_constants(data, off, size)

    # --- 解析图 ---
    ins = parse_spirv(spirv)
    scalars, composites, nulls = build_value_table(ins)

    tosa_set = None
    for x in ins:
        if x.opcode == 11 and len(x.ops) > 1:
            name, _ = decode_name(x.ops, 1)
            if name.startswith("TOSA"):
                tosa_set = x.ops[0]

    const_of, ops, produced_by = {}, [], {}
    input_of, graph_outputs = {}, []
    graph_name, io_ids = None, []

    for x in ins:
        if x.opcode == OP_GRAPH_CONSTANT:
            const_of[x.result_id] = x.ops[2]
        elif x.opcode == 12 and x.ops[2] == tosa_set:
            kind = TOSA.get(x.ops[3], f"?{x.ops[3]}")
            rec = {"index": len(ops), "kind": kind, "result": x.result_id}
            if kind == "CONV2D":
                rec.update(pad=x.in_operand(2), stride=x.in_operand(3),
                           dilation=x.in_operand(4), input=x.in_operand(7),
                           weight=x.in_operand(8), bias=x.in_operand(9),
                           input_zero_point=x.in_operand(10), weight_zero_point=x.in_operand(11))
            elif kind == "RESCALE":
                rec.update(scale32=x.in_operand(2), rounding=x.in_operand(3),
                           per_channel=x.in_operand(4), input_unsigned=x.in_operand(5),
                           output_unsigned=x.in_operand(6), input=x.in_operand(7),
                           multiplier=x.in_operand(8), shift=x.in_operand(9),
                           input_zero_point=x.in_operand(10), output_zero_point=x.in_operand(11))
            elif kind == "RESIZE":
                rec.update(mode=x.in_operand(2), input=x.in_operand(3),
                           scale=x.in_operand(4), offset=x.in_operand(5), border=x.in_operand(6))
            elif kind == "CONCAT":
                rec.update(axis=x.in_operand(2), inputs=x.ops[5:])
            elif kind == "TABLE":
                rec.update(input=x.in_operand(2), table=x.in_operand(3))
            ops.append(rec)
            produced_by[x.result_id] = rec["index"]
        elif x.opcode == OP_GRAPH_INPUT:
            # <Result Type> <Result id> <InputIndex(id)> [ElementIndex(id)]
            idx = resolve_value(x.ops[2], scalars, composites, nulls)
            input_of[x.result_id] = idx if isinstance(idx, int) else 0
        elif x.opcode == OP_GRAPH_SET_OUTPUT:
            # <Value(id)> <OutputIndex(id)> [ElementIndex(id)]  —— 无结果 id
            oidx = resolve_value(x.ops[1], scalars, composites, nulls)
            graph_outputs.append((oidx if isinstance(oidx, int) else 0, x.ops[0]))
        elif x.opcode == OP_GRAPH_ENTRYPOINT:
            nm, nwords = decode_name(x.ops, 1)
            graph_name = nm
            io_ids = list(x.ops[1 + nwords:])

    # 图输入张量由 OpGraphInputARM 定义；图输出由 OpGraphSetOutputARM 指定
    input_ids = sorted(input_of, key=lambda k: input_of[k])
    output_ids = [v for _, v in sorted(graph_outputs)]
    if not input_ids:
        input_ids = io_ids[:n_input] if n_input else io_ids[:1]
        output_ids = output_ids or io_ids[n_input:]

    unresolved = set()

    def describe(tid):
        """把一个 id 解析成结构化引用。"""
        if tid is None:
            return {"kind": "none"}
        if tid in const_of:
            cid = const_of[tid]
            r = resources[cid] if cid < len(resources) else {}
            return {"kind": "const", "const_id": cid,
                    "vk_format": r.get("vk_format"), "shape": r.get("shape")}
        if tid in produced_by:
            return {"kind": "node", "index": produced_by[tid]}
        if tid in input_of:
            return {"kind": "input", "io_index": input_of[tid]}
        if tid in output_ids:
            return {"kind": "output", "io_index": output_ids.index(tid)}
        val = resolve_value(tid, scalars, composites, nulls)
        if isinstance(val, str):
            unresolved.add(tid)
            return {"kind": "unknown", "tensor_id": tid, "resolved": val}
        return {"kind": "lit", "value": val}

    # 把 RESCALE 的 multiplier/shift 解析成真实数组
    for rec in ops:
        if rec["kind"] == "RESCALE":
            for key in ("input", "multiplier", "shift", "output_zero_point", "input_zero_point"):
                rec[key] = describe(rec[key])
            for key in ("scale32", "rounding", "per_channel", "input_unsigned", "output_unsigned"):
                rec[key] = resolve_value(rec[key], scalars, composites, nulls)
            if isinstance(rec["rounding"], int):
                rec["rounding"] = ROUNDING.get(rec["rounding"], rec["rounding"])
        elif rec["kind"] == "CONV2D":
            for key in ("input", "weight", "bias"):
                rec[key] = describe(rec[key])
            for key in ("pad", "stride", "dilation", "input_zero_point", "weight_zero_point"):
                rec[key] = resolve_value(rec[key], scalars, composites, nulls)
        elif rec["kind"] == "RESIZE":
            rec["input"] = describe(rec["input"])
            for key in ("mode", "scale", "offset", "border"):
                rec[key] = resolve_value(rec[key], scalars, composites, nulls)
            if isinstance(rec["mode"], int):
                rec["mode"] = RESIZE_MODE.get(rec["mode"], rec["mode"])
        elif rec["kind"] == "CONCAT":
            rec["inputs"] = [describe(t) for t in rec["inputs"]]
            rec["axis"] = resolve_value(rec["axis"], scalars, composites, nulls)
        elif rec["kind"] == "TABLE":
            rec["input"] = describe(rec["input"])
            rec["table"] = describe(rec["table"])

    in_res = [r for r in resources if r["category"] == "INPUT"]
    out_res = [r for r in resources if r["category"] == "OUTPUT"]
    ir = {
        "vgf": os.path.basename(vgf_path),
        "graph_name": graph_name,
        "unresolved_tensor_ids": sorted(unresolved),
        "modules": module_info,
        "spirv": {"word_count": len(spirv),
                  "version": f"{spirv[1] >> 16}.{(spirv[1] >> 8) & 0xFF}",
                  "id_bound": spirv[3]},
        "inputs": [{"io_index": i, "tensor_id": t,
                    "shape": in_res[i]["shape"] if i < len(in_res) else None,
                    "vk_format": in_res[i]["vk_format"] if i < len(in_res) else None}
                   for i, t in enumerate(input_ids)],
        "outputs": [{"io_index": i, "tensor_id": t,
                     "shape": out_res[i]["shape"] if i < len(out_res) else None,
                     "vk_format": out_res[i]["vk_format"] if i < len(out_res) else None}
                    for i, t in enumerate(output_ids)],
        "constants": consts["entries"],
        "constant_count": consts["count"],
        "resources": resources,
        "ops": ops,
    }
    return ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("-o", "--output")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ir = analyze(args.vgf, args.verbose)

    print(f"图名: {ir['graph_name']}   模块: {[m['name'] for m in ir['modules']]}"
          f"   SPIR-V {ir['spirv']['word_count']} words")
    print(f"输入张量 id: {[i['tensor_id'] for i in ir['inputs']]}"
          f"   输出张量 id: {[o['tensor_id'] for o in ir['outputs']]}")
    print(f"常量 {ir['constant_count']} 个 / 算子 {len(ir['ops'])} 个\n")

    def fmt(r):
        if r["kind"] == "const":
            return f"c{r['const_id']}{r.get('shape') or ''}"
        if r["kind"] == "node":
            return f"op{r['index']}"
        if r["kind"] == "input":
            return "INPUT"
        if r["kind"] == "unknown":
            return f"??{r['tensor_id']}"
        if r["kind"] == "lit":
            v = r["value"]
            return f"lit{v}" if not isinstance(v, list) or len(v) <= 6 else f"lit[{len(v)}]"
        return r["kind"]

    print(f"未解析的张量 id: {ir['unresolved_tensor_ids']}\n")

    for rec in ir["ops"]:
        k = rec["kind"]
        if k == "CONV2D":
            pad = rec["pad"]
            print(f"{rec['index']:3d} CONV2D   in={fmt(rec['input']):<22} w={fmt(rec['weight']):<16}"
                  f" b={fmt(rec['bias']):<14} pad={pad}"
                  f" stride={rec["stride"]}")
        elif k == "RESCALE":
            print(f"{rec['index']:3d} RESCALE  in={fmt(rec['input']):<22} m={fmt(rec['multiplier']):<16}"
                  f" s={fmt(rec['shift']):<14} outzp={fmt(rec['output_zero_point'])}  {rec['rounding']}")
        elif k == "RESIZE":
            print(f"{rec['index']:3d} RESIZE   in={fmt(rec['input']):<22} mode={rec['mode']}"
                  f" scale={rec['scale']}")
        elif k == "CONCAT":
            print(f"{rec['index']:3d} CONCAT   axis={rec['axis']} inputs={[fmt(t) for t in rec['inputs']]}")
        elif k == "TABLE":
            print(f"{rec['index']:3d} TABLE    in={fmt(rec['input']):<22} table={fmt(rec['table'])}")
        else:
            print(f"{rec['index']:3d} {k}")

    if args.output:
        json.dump(ir, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n已写出 {args.output}")


if __name__ == "__main__":
    main()
