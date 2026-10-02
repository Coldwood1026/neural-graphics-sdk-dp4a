#!/usr/bin/env python3
"""Reconstruct the NFRU inference graph from its TOSA-ext-inst SPIR-V module.

Findings encoded here (verified against nfru_v1_int8.vgf):
  * The VGF `graph` module is SPIR-V that imports the extended instruction set
    "TOSA.001000.1" and builds the network with OpExtInst.
  * Node type ids are declared by OpGraphConstantARM, whose `IdResultType`
    operand is a *literal* opcode and whose IdResult names the node.
  * Weight tensors are declared by OpTypeTensorARM + OpGraphConstantARM.
  * Every network operation is an OpExtInst on set 368 with a TOSA opcode:
      CONV2D, RESCALE, CONCAT, RESIZE, plus CAST-style conversions.
  * The graph's real input and output are the tensors converted by the two
    outermost OpExtInst (resultType 2 / 362 at each end).

Usage:
    python nfru_graph.py module_0.spv [--json out.json]
"""
import argparse
import json
import os
import struct
import sys

TOSA = {
    2: "CONV2D", 4: "DEPTHWISE_CONV2D", 10: "CLAMP", 12: "SIGMOID",
    13: "TANH", 14: "ADD", 25: "MAXIMUM", 26: "MINIMUM", 27: "MUL",
    29: "SUB", 54: "CONCAT", 55: "PAD", 56: "RESHAPE", 60: "TRANSPOSE",
    63: "RESIZE", 64: "CAST", 65: "RESCALE",
}

CORE = {
    0x0005: "OpName", 0x000A: "OpExtension", 0x000B: "OpExtInstImport",
    0x000C: "OpExtInst", 0x000E: "OpMemoryModel", 0x0011: "OpCapability",
    0x0016: "OpTypeVoid", 0x0017: "OpTypeBool", 0x0018: "OpTypeInt",
    0x0019: "OpTypeFloat", 0x001A: "OpTypeVector", 0x001C: "OpTypeImage",
    0x001F: "OpTypeArray", 0x0020: "OpTypeRuntimeArray", 0x0021: "OpTypeStruct",
    0x0023: "OpTypePointer", 0x0024: "OpTypeFunction", 0x002B: "OpTypeForwardPointer",
    0x002C: "OpConstantSampler", 0x002E: "OpConstant", 0x002F: "OpConstantComposite",
    0x0030: "OpConstantSampler", 0x0031: "OpConstantNull", 0x0039: "OpFunction",
    0x003A: "OpFunctionParameter", 0x003B: "OpFunctionEnd", 0x003E: "OpVariable",
    0x0040: "OpLoad", 0x004B: "OpDecorate", 0x004C: "OpMemberDecorate",
    0x1043: "OpTypeTensorARM", 0x1055: "OpGraphConstantARM",
    0x1056: "OpGraphEntryPointARM", 0x1057: "OpGraphARM", 0x1058: "OpGraphInputARM",
    0x1059: "OpGraphSetOutputARM", 0x105A: "OpGraphEndARM", 0x105E: "OpTypeGraphARM",
}


def name(op):
    if op in CORE:
        return CORE[op]
    if 0x1040 <= op <= 0x104F:
        return f"OpTensor/GraphARM_0x{op:04X}"
    return f"Op_0x{op:04X}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spv")
    ap.add_argument("--json")
    args = ap.parse_args()

    data = open(args.spv, "rb").read()
    words = struct.unpack_from(f"<{len(data)//4}I", data, 0)

    insts, i = [], 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        insts.append((op, list(words[i+1:i+wc])))
        i += wc

    const_int, tensor_types, names = {}, {}, {}
    node_kind = {}          # IdResult -> TOSA opcode (declared via IdResultType literal)
    extinst = {}            # result id -> (resultType, tosaOp, operands)
    graph_inputs, graph_outputs = [], []

    for op, ops in insts:
        if op == 0x0005 and len(ops) >= 2:
            names[ops[0]] = struct.pack(f"<{len(ops)-1}I", *ops[1:]).split(b"\0")[0].decode("utf-8", "replace")
        elif op == 0x002E and len(ops) >= 3:
            const_int[ops[1]] = ops[2]
        elif op == 0x1043 and len(ops) >= 2:
            tensor_types[ops[0]] = {"format": ops[1],
                                    "shape": ops[2] if len(ops) > 2 else None,
                                    "strides": ops[3] if len(ops) > 3 else None}
        elif op == 0x1055 and len(ops) >= 3:
            node_kind[ops[1]] = ops[2]     # IdResultType is the literal TOSA opcode
        elif op == 0x000C and len(ops) >= 4:
            extinst[ops[1]] = (ops[0], ops[3], ops[4:])
        elif op == 0x1058:
            graph_inputs.append(ops)
        elif op == 0x1059:
            graph_outputs.append(ops)

    print(f"instructions={len(insts)}  declared node kinds={len(node_kind)}  "
          f"tensor types={len(tensor_types)}  ops={len(extinst)}")

    print("\n== declared node kinds (OpGraphConstantARM: literal -> id) ==")
    for nid, kind in sorted(node_kind.items(), key=lambda kv: kv[1]):
        print(f"  kind={kind:<4} ({TOSA.get(kind, '?'):<14}) declaredAsId={nid}")

    print("\n== weight tensor types ==")
    for tid, info in tensor_types.items():
        shape = info["shape"]
        shape_vals = [const_int.get(s, s) for s in (shape if isinstance(shape, list) else [shape])]
        print(f"  tensorTypeId={tid:<6} format={info['format']} "
              f"shapeIds={shape} -> values={shape_vals}")

    print("\n== graph input / output declarations ==")
    for ops in graph_inputs:
        print(f"  input   {ops}")
    for ops in graph_outputs:
        print(f"  output  {ops}")

    # ---- reconstruct op sequence in program order -------------------------
    print("\n== operation stream (program order) ==")
    conv_count = 0
    ops_list = []
    for op, ops in insts:
        if op != 0x000C:
            continue
        rt, rid, setid, inst = ops[0], ops[1], ops[2], ops[3]
        operands = ops[4:]
        tname = TOSA.get(inst, f"TOSA_{inst}")
        if inst == 2:
            conv_count += 1
            print(f"  #{conv_count:<3} CONV2D   result={rid:<5} resultType={rt:<5} operands={operands}")
        else:
            print(f"       {tname:<9} result={rid:<5} resultType={rt:<5} operands={operands}")
        ops_list.append({"result": rid, "resultType": rt, "tosaOp": inst,
                         "name": tname, "operands": operands})

    print(f"\ntotal CONV2D = {conv_count}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"nodeKinds": {str(k): v for k, v in node_kind.items()},
                       "tensorTypes": {str(k): v for k, v in tensor_types.items()},
                       "operations": ops_list,
                       "graphInputs": graph_inputs, "graphOutputs": graph_outputs,
                       "names": {str(k): v for k, v in names.items()}}, fh, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
