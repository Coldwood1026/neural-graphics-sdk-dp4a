#!/usr/bin/env python3
"""Full topology reconstruction of an Arm SPV_ARM_graph module.

Verified against the NFRU model (`nfru_v1_int8.vgf`).

Instruction roles (opcodes from the official SPIR-V grammar):
  0x1055 OpGraphConstantARM  IdResultType(=nodeType literal), IdResult, LiteralInteger
                             -> registers graph node type `result` with value `literal`,
                                e.g. (type 3 = 1) declares node type 3 is a Convolution.
  0x105E OpTypeGraphARM      IdResult, LiteralInteger, IdRef*
                             -> creates the graph object; the IdRef list is the array of
                                constant IDs that hold the convolution weight tensors.
  0x1043 OpTypeTensorARM     IdResult, IdRef(format), IdRef?(shape), IdRef?(strides)
                             -> declares a tensor type; shape array reveals the weight layout.
  0x1057 OpGraphARM          IdResultType, IdResult
                             -> instantiates a node; its operands/attributes are supplied by
                                the constant composites listed in the type's table.
  0x1058 OpGraphInputARM     IdResultType, IdResult, IdRef*
                             -> declares graph inputs.
  0x1059 OpGraphSetOutputARM IdRef, IdRef, IdRef*
                             -> marks outputs.
  0x1056 OpGraphEntryPointARM IdRef, LiteralString, IdRef*

Usage:
    python graph_topo.py module_0.spv [--json out.json]
"""
import argparse
import json
import struct
import sys

OPNAMES = {
    0x0005: "OpName", 0x000A: "OpExtension", 0x0011: "OpCapability",
    0x0016: "OpTypeVoid", 0x0017: "OpTypeBool", 0x0018: "OpTypeInt",
    0x0019: "OpTypeFloat", 0x001A: "OpTypeVector", 0x001F: "OpTypeArray",
    0x0020: "OpTypeRuntimeArray", 0x0021: "OpTypeStruct", 0x0023: "OpTypePointer",
    0x0024: "OpTypeFunction", 0x002E: "OpConstant", 0x002F: "OpConstantComposite",
    0x004B: "OpDecorate", 0x004C: "OpMemberDecorate", 0x0044: "OpAccessChain",
    0x0047: "OpArrayLength", 0x0039: "OpFunction", 0x003A: "OpFunctionParameter",
    0x003B: "OpFunctionEnd", 0x003E: "OpVariable", 0x0040: "OpLoad",
    0x1043: "OpTypeTensorARM", 0x1044: "OpTensorReadARM", 0x1045: "OpTensorWriteARM",
    0x1046: "OpTensorQuerySizeARM", 0x1055: "OpGraphConstantARM",
    0x1056: "OpGraphEntryPointARM", 0x1057: "OpGraphARM", 0x1058: "OpGraphInputARM",
    0x1059: "OpGraphSetOutputARM", 0x105A: "OpGraphEndARM", 0x105E: "OpTypeGraphARM",
}


def oname(op):
    return OPNAMES.get(op, f"Op_0x{op:04X}")


def parse(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version, generator, bound, schema = struct.unpack_from("<5I", data, 0)
    if magic != 0x07230203:
        raise SystemExit(f"bad magic 0x{magic:08X}")
    words = struct.unpack_from(f"<{len(data)//4}I", data, 0)
    insts, i = [], 5
    while i < len(words):
        wc, opcode = words[i] >> 16, words[i] & 0xFFFF
        if not wc:
            raise SystemExit(f"zero word count at word {i}")
        insts.append((opcode, list(words[i+1:i+wc])))
        i += wc
    return insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spv")
    ap.add_argument("--json")
    args = ap.parse_args()

    insts = parse(args.spv)

    names, const_int, const_comp = {}, {}, {}
    for op, ops in insts:
        if op == 0x0005 and len(ops) >= 2:
            names[ops[0]] = struct.pack(f"<{len(ops)-1}I", *ops[1:]).split(b"\0")[0].decode("utf-8", "replace")
        elif op == 0x002E and len(ops) >= 3:
            const_int[ops[1]] = ops[2]
        elif op == 0x002F and len(ops) >= 2:
            const_comp[ops[1]] = ops[2:]

    def scalar(x):
        return const_int.get(x, x)

    def arr(x):
        if x in const_comp:
            return [scalar(m) for m in const_comp[x]]
        if x in const_int:
            return [const_int[x]]
        return x

    # node types:  OpGraphConstantARM(0x1055) -> result id maps a kind to a literal
    node_types = {}
    type_decls = {}      # 0x105E OpTypeGraphARM: id -> (kindLiteral, [constant ids])
    tensor_types = {}    # 0x1043 OpTypeTensorARM: id -> (format id, shape, strides)
    for op, ops in insts:
        if op == 0x1055 and len(ops) >= 3:
            node_types[ops[1]] = ops[2]
        elif op == 0x105E and len(ops) >= 2:
            type_decls[ops[0]] = (ops[1], ops[2:])
        elif op == 0x1043 and len(ops) >= 2:
            fmt = scalar(ops[1]) if len(ops) > 1 else None
            shape = arr(ops[2]) if len(ops) > 2 else None
            strides = arr(ops[3]) if len(ops) > 3 else None
            tensor_types[ops[0]] = (fmt, shape, strides)

    # graph instantiation: 0x1057 OpGraphARM(IdResultType, IdResult)
    # node type id -> list of allocated node ids
    nodes_by_type = {}
    for op, ops in insts:
        if op == 0x1057 and len(ops) >= 2:
            nodes_by_type.setdefault(ops[0], []).append(ops[1])

    # inputs / outputs / entry point
    graph_inputs, graph_outputs, entry_points = [], [], []
    for op, ops in insts:
        if op == 0x1058:
            graph_inputs.append(ops)
        elif op == 0x1059:
            graph_outputs.append(ops)
        elif op == 0x1056:
            entry_points.append(ops)

    print(f"SPIR-V words={len(insts)} instructions")
    print(f"nodes instantiated={sum(len(v) for v in nodes_by_type.values())}  "
          f"node types={len(type_decls)}  tensor types={len(tensor_types)}")

    print("\n== node type table (OpGraphConstantARM) ==")
    kind_by_typeid = {}
    for tid, kind in sorted(node_types.items(), key=lambda kv: kv[1]):
        kind_by_typeid.setdefault(kind, []).append(tid)
        print(f"  nodeTypeId={tid:<6} kind={kind}")

    print("\n== OpTypeGraphARM declarations ==")
    for gid, (kind, consts) in type_decls.items():
        print(f"  graphId={gid:<6} kind={kind:<4} constantIds={consts}")

    print("\n== tensor types (OpTypeTensorARM) ==")
    for tid, (fmt, shape, strides) in sorted(tensor_types.items()):
        print(f"  tensorTypeId={tid:<6} format={fmt} shape={shape} strides={strides}")

    print("\n== graph inputs ==")
    for ops in graph_inputs:
        print(f"  {ops}")

    print("\n== graph outputs ==")
    for ops in graph_outputs:
        print(f"  {ops}")

    if entry_points:
        print("\n== entry points ==")
        for ops in entry_points:
            raw = struct.pack(f"<{len(ops)-1}I", *ops[1:])
            s = raw.split(b"\0")[0].decode("utf-8", "replace")
            print(f"  {ops[0]}  name='{s}'  args={ops[1+ (len(s)//4) + 1:]}")

    print("\n== nodes per type ==")
    for type_id, node_ids in nodes_by_type.items():
        kind = node_types.get(type_id, "?")
        print(f"  typeId={type_id:<6} kind={kind:<4} count={len(node_ids)}  ids={node_ids}")

    print("\n== OpConstantComposite values (node operands / attributes) ==")
    for cid, members in sorted(const_comp.items()):
        print(f"  compositeId={cid:<6} members={[scalar(m) for m in members]}")

    # The OpTypeGraphARM const list is the weight tensor set.
    if type_decls:
        for gid, (kind, consts) in type_decls.items():
            print(f"\n== weight tensor ids for graph {gid} ({len(consts)}) ==")
            print("  " + ", ".join(str(c) for c in consts))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "nodeTypes": {str(k): v for k, v in node_types.items()},
                "graphDecls": {str(k): {"kind": v[0], "constants": v[1]} for k, v in type_decls.items()},
                "tensorTypes": {str(k): {"format": v[0], "shape": v[1], "strides": v[2]}
                                for k, v in tensor_types.items()},
                "nodesByType": {str(k): v for k, v in nodes_by_type.items()},
                "constants": {str(k): [scalar(m) for m in v] for k, v in const_comp.items()},
                "graphInputs": graph_inputs,
                "graphOutputs": graph_outputs,
                "names": {str(k): v for k, v in names.items()},
            }, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
