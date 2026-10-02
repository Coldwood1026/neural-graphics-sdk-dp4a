#!/usr/bin/env python3
"""从 VGF 图中的 SPIR-V 模块重建计算图（阶段 1 收尾）。

解析 TOSA 图模块，把每个算子的张量操作数解析成
「常量 / 图输入 / 上游算子输出」三类，输出线性算子序列。

操作数签名来源：ai-ml-emulation-layer-for-vulkan 的 graph_tosa_1_0_ext_inst.cpp
（GetInOperand(0)=指令集 id，(1)=TOSA 指令号，实际参数从 (2) 起）

用法:
    python spv_graph.py <graph>.spv [-o graph.json]
    python spv_graph.py <graph>.spv --probe      # 只做 opcode 探测
"""
import argparse
import json
import struct
import sys
from collections import Counter

TOSA = {
    0: "ARGMAX", 1: "AVG_POOL2D", 2: "CONV2D", 3: "CONV3D", 4: "DEPTHWISE_CONV2D",
    5: "FFT2D", 6: "MATMUL", 7: "MAX_POOL2D", 8: "RFFT2D", 9: "TRANSPOSE_CONV2D",
    10: "CLAMP", 11: "ERF", 12: "SIGMOID", 13: "TANH", 14: "ADD",
    15: "ARITHMETIC_RIGHT_SHIFT", 16: "BITWISE_AND", 17: "BITWISE_OR", 18: "BITWISE_XOR",
    19: "INTDIV", 20: "LOGICAL_AND", 21: "LOGICAL_LEFT_SHIFT", 22: "LOGICAL_RIGHT_SHIFT",
    23: "LOGICAL_OR", 24: "LOGICAL_XOR", 25: "MAXIMUM", 26: "MINIMUM", 27: "MUL",
    28: "POW", 29: "SUB", 30: "TABLE", 31: "ABS", 32: "BITWISE_NOT", 33: "CEIL",
    34: "CLZ", 35: "COS", 36: "EXP", 37: "FLOOR", 38: "LOG", 39: "LOGICAL_NOT",
    40: "NEGATE", 41: "RECIPROCAL", 42: "RSQRT", 43: "SIN", 44: "SELECT",
    45: "EQUAL", 46: "GREATER", 47: "GREATER_EQUAL", 48: "REDUCE_ALL", 49: "REDUCE_ANY",
    50: "REDUCE_MAX", 51: "REDUCE_MIN", 52: "REDUCE_PRODUCT", 53: "REDUCE_SUM",
    54: "CONCAT", 55: "PAD", 56: "RESHAPE", 57: "REVERSE", 58: "SLICE", 59: "TILE",
    60: "TRANSPOSE", 61: "GATHER", 62: "SCATTER", 63: "RESIZE", 64: "CAST", 65: "RESCALE",
}

ROUNDING = {0: "UNKNOWN", 1: "SINGLE_ROUND", 2: "INEXACT_ROUND", 3: "DOUBLE_ROUND"}
RESIZE_MODE = {0: "UNKNOWN", 1: "NEAREST", 2: "BILINEAR"}


class Instr:
    __slots__ = ("opcode", "ops", "word_index")

    def __init__(self, opcode, ops, word_index):
        self.opcode = opcode
        self.ops = ops
        self.word_index = word_index

    @property
    def result_type(self):
        return self.ops[0] if self.ops else None

    @property
    def result_id(self):
        return self.ops[1] if len(self.ops) > 1 else None

    def in_operand(self, i):
        """GetInOperand(i) 对应 ops[i+2]：ops[0]=result type, ops[1]=result id,
        ops[2]=指令集 id (=InOperand 0), ops[3]=TOSA 指令号 (=InOperand 1)。"""
        k = i + 2
        return self.ops[k] if k < len(self.ops) else None


def parse_spirv(data):
    words = list(struct.unpack("<%dI" % (len(data) // 4), data[:len(data) // 4 * 4]))
    magic = words[0]
    assert magic == 0x07230203, f"不是 SPIR-V 魔数: {magic:#x}"
    ins = []
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0:
            break
        ins.append(Instr(op, words[i + 1:i + wc], i))
        i += wc
    return ins


def discover_graph_const_opcode(ins, expected_count):
    """找出 OpGraphConstantARM 的 opcode：出现次数等于常量数，
    且第 3 个操作数的取值恰好是 0..N-1 的一个排列。"""
    cands = []
    for op, cnt in Counter(x.opcode for x in ins).items():
        if cnt != expected_count:
            continue
        vals = sorted(x.ops[2] for x in ins
                      if x.opcode == op and len(x.ops) >= 3)
        if len(vals) == expected_count and vals == list(range(expected_count)):
            cands.append(op)
    return cands


def collect(ins, const_opcode, tosa_set_id):
    const_of = {}       # result_id -> 常量表 id
    input_of = {}       # result_id -> 图输入序号
    produced_by = {}    # result_id -> 算子序号
    literal_of = {}     # result_id -> 字面量常量值
    ops = []
    outputs = []
    entry = None

    for x in ins:
        if x.opcode == const_opcode:
            const_of[x.result_id] = x.ops[2]
        elif x.opcode == 12:                                  # OpExtInst
            if x.ops[2] != tosa_set_id:
                continue
            kind = TOSA.get(x.ops[3], f"?{x.ops[3]}")
            idx = len(ops)
            rec = {"index": idx, "kind": kind, "result": x.result_id,
                   "operands": x.ops[4:]}
            if kind == "CONV2D":
                rec["pad"] = x.in_operand(2)
                rec["stride"] = x.in_operand(3)
                rec["dilation"] = x.in_operand(4)
                rec["input"] = x.in_operand(7)
                rec["weight"] = x.in_operand(8)
                rec["bias"] = x.in_operand(9)
                rec["input_zero_point"] = x.in_operand(10)
                rec["weight_zero_point"] = x.in_operand(11)
            elif kind == "RESCALE":
                rec["input"] = x.in_operand(7)
                rec["multiplier"] = x.in_operand(8)
                rec["shift"] = x.in_operand(9)
                rec["input_zero_point"] = x.in_operand(10)
                rec["output_zero_point"] = x.in_operand(11)
            elif kind == "RESIZE":
                rec["mode"] = x.in_operand(2)
                rec["input"] = x.in_operand(3)
                rec["scale"] = x.in_operand(4)
            elif kind == "CONCAT":
                rec["axis"] = x.in_operand(2)
                rec["inputs"] = x.ops[5:]
            elif kind == "TABLE":
                rec["input"] = x.in_operand(2)
                rec["table"] = x.in_operand(3)
            ops.append(rec)
            produced_by[x.result_id] = idx
        elif x.opcode == 43:                                  # OpConstant
            literal_of[x.result_id] = {"type": x.ops[0], "value": x.ops[2] if len(x.ops) > 2 else None,
                                       "extra": x.ops[3:]}
        elif x.opcode == 44:                                  # OpConstantComposite
            literal_of[x.result_id] = {"type": x.ops[0], "composite": x.ops[2:]}
        elif x.opcode == 46:                                  # OpConstantNull
            literal_of[x.result_id] = {"type": x.ops[0], "null": True}
        else:
            # 非 TOSA 的图算子：靠出现次数和操作数形状识别关键几种
            pass

    # 第二轮：识别图输入 / 输出 / 入口点（opcode 未知，用「操作数含 0..N 序号」特征）
    for x in ins:
        if x.opcode in (11, 12, 43, 44, 46, const_opcode) or x.opcode is None:
            continue
        if x.result_id is None or len(x.ops) < 3:
            continue
        small = x.ops[2]
        if isinstance(small, int) and small < 64 and x.result_id not in produced_by:
            input_of.setdefault(x.result_id, small)
            outputs.append(x.result_id)

    return const_of, input_of, produced_by, literal_of, ops


def describe(tid, const_of, input_of, produced_by, literal_of):
    if tid is None:
        return "none"
    if tid in const_of:
        return f"const[{const_of[tid]}]"
    if tid in produced_by:
        return f"op{produced_by[tid]}"
    if tid in input_of:
        return f"input[{input_of[tid]}]"
    if tid in literal_of:
        lit = literal_of[tid]
        if "value" in lit:
            return f"lit({lit['value']})"
        if "composite" in lit:
            return f"litComp{lit['composite']}"
        return "lit(null)"
    return f"%{tid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spv")
    ap.add_argument("-o", "--output")
    ap.add_argument("--const-count", type=int, default=43,
                    help="常量表条目数，用于识别 OpGraphConstantARM")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--dump-unknown", type=lambda s: [int(v) for v in s.split(",")],
                    help="逗号分隔的 opcode 列表，转储这些指令的全部操作数后退出")
    args = ap.parse_args()

    data = open(args.spv, "rb").read()
    ins = parse_spirv(data)

    census = Counter(x.opcode for x in ins)
    unknown = sorted(o for o in census if o not in
                     set(range(0, 100)) | {248} and census[o] <= 50)

    print(f"SPIR-V: {len(ins)} 条指令")
    print("非核心/图扩展 opcode 频次:")
    for o in sorted(census):
        if o in (11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 28, 30, 32, 41, 42, 43, 44, 46, 54, 56, 5, 7):
            continue
        print(f"  op{o:<6} x{census[o]}")

    cands = discover_graph_const_opcode(ins, args.const_count)
    print(f"\n候选 OpGraphConstantARM opcode: {cands}")
    if args.probe:
        return
    if len(cands) != 1:
        sys.exit(f"无法唯一定位 OpGraphConstantARM（候选 {cands}），请用 --const-count 调整")

    const_opcode = cands[0]
    tosa_set = None
    for x in ins:
        if x.opcode == 11 and len(x.ops) > 1:
            # OpExtInstImport 没有 result type：ops[0] 就是结果 id，ops[1:] 是名字
            raw = b"".join(struct.pack("<I", w) for w in x.ops[1:])
            if raw.split(b"\0")[0] == b"TOSA.001000.1":
                tosa_set = x.ops[0]
    print(f"TOSA 指令集 id = {tosa_set}")

    if args.dump_unknown:
        for x in ins:
            if x.opcode in args.dump_unknown:
                print(f"  op{x.opcode}  result_id={x.result_id}  ops={x.ops}")
        return

    const_of, input_of, produced_by, literal_of, ops = collect(ins, const_opcode, tosa_set)

    print(f"\n常量 {len(const_of)} 个 / 算子 {len(ops)} 个\n")
    for r in ops:
        d = describe
        if r["kind"] == "CONV2D":
            print(f"  {r['index']:2d} CONV2D   in={d(r['input'],const_of,input_of,produced_by,literal_of):<8}"
                  f" w={d(r['weight'],const_of,input_of,produced_by,literal_of):<9}"
                  f" b={d(r['bias'],const_of,input_of,produced_by,literal_of):<9}"
                  f" pad={r['pad']} stride={r['stride']}")
        elif r["kind"] == "RESCALE":
            print(f"  {r['index']:2d} RESCALE  in={d(r['input'],const_of,input_of,produced_by,literal_of):<8}"
                  f" m={d(r['multiplier'],const_of,input_of,produced_by,literal_of):<9}"
                  f" s={d(r['shift'],const_of,input_of,produced_by,literal_of):<9}"
                  f" outzp={d(r['output_zero_point'],const_of,input_of,produced_by,literal_of)}")
        elif r["kind"] == "CONCAT":
            ins_s = ",".join(d(t, const_of, input_of, produced_by, literal_of) for t in r["inputs"])
            print(f"  {r['index']:2d} CONCAT   axis={r['axis']} inputs=[{ins_s}]")
        elif r["kind"] == "TABLE":
            print(f"  {r['index']:2d} TABLE    in={d(r['input'],const_of,input_of,produced_by,literal_of):<8}"
                  f" table={d(r['table'],const_of,input_of,produced_by,literal_of)}")
        else:
            print(f"  {r['index']:2d} {r['kind']}")

    if args.output:
        json.dump({"ops": ops,
                   "const_of": {str(k): v for k, v in const_of.items()},
                   "input_of": {str(k): v for k, v in input_of.items()}},
                  open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n已写出 {args.output}")


if __name__ == "__main__":
    main()
