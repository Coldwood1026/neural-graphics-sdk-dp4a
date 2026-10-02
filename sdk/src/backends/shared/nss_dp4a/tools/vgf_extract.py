#!/usr/bin/env python3
"""VGF 结构提取器 —— 阶段 1 工具。

把 Arm Neural Graphics 的 .vgf 模型包完整解码成 JSON：
分段布局、模块表、dispatch 序列、资源表（张量形状/格式）、常量表。

依据 Arm 官方 FlatBuffers schema（arm/ai-ml-sdk-vgf-library 的 schema/vgf.fbs），
不依赖 flatc / pybind11 / 编译工具链，纯标准库。

用法:
    python vgf_extract.py <model>.vgf [-o out.json]
    python vgf_extract.py <model>.vgf --dump-constant 3   # 导出第 3 个常量的原始字节
"""
import argparse
import json
import struct
import sys

SECTION_NAMES = ["modules", "model_sequence", "resources", "constants"]

MODULE_TYPE = {0: "COMPUTE", 1: "GRAPH"}
RESOURCE_CATEGORY = {0: "INPUT", 1: "OUTPUT", 2: "INTERMEDIATE", 3: "CONSTANT"}

VK_DESCRIPTOR_TYPE = {
    0: "SAMPLER", 1: "COMBINED_IMAGE_SAMPLER", 2: "SAMPLED_IMAGE", 3: "STORAGE_IMAGE",
    4: "UNIFORM_TEXEL_BUFFER", 5: "STORAGE_TEXEL_BUFFER", 6: "UNIFORM_BUFFER",
    7: "STORAGE_BUFFER", 8: "UNIFORM_BUFFER_DYNAMIC", 9: "STORAGE_BUFFER_DYNAMIC",
    10: "INPUT_ATTACHMENT", 1000150000: "INLINE_UNIFORM_BLOCK",
    1000138000: "ACCELERATION_STRUCTURE_KHR", 1000440000: "TENSOR_ARM",
}

# 只收录 NSS 实际会用到的，其余输出原始数值
VK_FORMAT = {
    0: "UNDEFINED",
    9: "R8_UNORM", 10: "R8_SNORM", 13: "R8_UINT", 14: "R8_SINT", 15: "R8_SRGB",
    16: "R8G8_UNORM", 20: "R8G8_UINT", 21: "R8G8_SINT",
    23: "R8G8B8A8_UNORM", 37: "R8G8B8A8_SNORM", 41: "R8G8B8A8_UINT",
    42: "R8G8B8A8_SINT", 43: "R8G8B8A8_SRGB",
    58: "A2B10G10R10_UNORM_PACK32",
    70: "R16_UNORM", 74: "R16_UINT", 75: "R16_SINT", 76: "R16_SFLOAT",
    77: "R16G16_UNORM", 81: "R16G16_UINT", 82: "R16G16_SINT", 83: "R16G16_SFLOAT",
    88: "R16G16B16A16_UNORM", 95: "R16G16B16A16_UINT", 96: "R16G16B16A16_SINT",
    97: "R16G16B16A16_SFLOAT",
    98: "R32_UINT", 99: "R32_SINT", 100: "R32_SFLOAT",
    103: "R32G32_UINT", 104: "R32G32_SINT", 105: "R32G32_SFLOAT",
    109: "R32G32B32A32_UINT", 110: "R32G32B32A32_SINT", 111: "R32G32B32A32_SFLOAT",
    122: "B10G11R11_UFLOAT_PACK32",
}

U32_MAX = 4294967295


class FlatBuffer:
    """最小 FlatBuffers 读取器（table / vector / string / union 就够了）。"""

    def __init__(self, data, base, limit):
        self.d = data
        self.base = base
        self.limit = limit

    def _u32(self, p):
        return struct.unpack_from("<I", self.d, p)[0]

    def _i32(self, p):
        return struct.unpack_from("<i", self.d, p)[0]

    def _u64(self, p):
        return struct.unpack_from("<Q", self.d, p)[0]

    def _i64(self, p):
        return struct.unpack_from("<q", self.d, p)[0]

    def _u16(self, p):
        return struct.unpack_from("<H", self.d, p)[0]

    def root(self):
        return self.base + self._u32(self.base)

    def field_pos(self, table, idx):
        """返回字段的绝对偏移；字段不存在（或为默认值）返回 None。"""
        vt = table - self._i32(table)
        vt_size = self._u16(vt)
        n = (vt_size - 4) // 2
        if idx >= n:
            return None
        off = self._u16(vt + 4 + idx * 2)
        return None if off == 0 else table + off

    def scalar(self, table, idx, fmt="I"):
        p = self.field_pos(table, idx)
        if p is None:
            return None
        return struct.unpack_from("<" + fmt, self.d, p)[0]

    def indirect(self, table, idx):
        """偏移型字段（string / vector / sub-table）的绝对地址。"""
        p = self.field_pos(table, idx)
        return None if p is None else p + self._u32(p)

    def string(self, table, idx):
        p = self.indirect(table, idx)
        if p is None:
            return None
        n = self._u32(p)
        return self.d[p + 4:p + 4 + n].decode("utf-8", "replace")

    def vector(self, table, idx, reader):
        p = self.indirect(table, idx)
        if p is None:
            return []
        n = self._u32(p)
        start = p + 4
        return [reader(start, i) for i in range(n)]

    def vec_scalar(self, table, idx, fmt):
        return self.vector(table, idx, lambda s, i: struct.unpack_from("<" + fmt, self.d, s + i * struct.calcsize(fmt))[0])

    def vec_table(self, table, idx, fn):
        return self.vector(table, idx, lambda s, i: fn(s + i * 4 + self._u32(s + i * 4)))

    def vec_string(self, table, idx):
        def rd(s, i):
            p = s + i * 4
            p += self._u32(p)
            n = self._u32(p)
            return self.d[p + 4:p + 4 + n].decode("utf-8", "replace")
        return self.vector(table, idx, rd)


def dflt(v, d):
    """FlatBuffers 会省略等于默认值的字段，取不到时按 schema 默认值处理。"""
    return d if v is None else v


def parse_modules(fb):
    root = fb.root()
    out = []
    for m in fb.vec_table(root, 0, lambda p: p):
        code_type = fb.scalar(m, 3, "B")
        # union 的值字段指向 SPIRV table，该 table 的字段 0 才是 words 向量
        code_tbl = fb.indirect(m, 4)
        words = fb.vec_scalar(code_tbl, 0, "I") if code_tbl is not None else []
        out.append({
            "type": MODULE_TYPE.get(dflt(fb.scalar(m, 0, "B"), 0), "?"),
            "name": fb.string(m, 1),
            "entry_point": fb.string(m, 2),
            "code_kind": {0: "NONE", 1: "SPIRV", 2: "GLSL", 3: "HLSL"}.get(dflt(code_type, 0), "?"),
            "spirv_word_count": len(words),
            "spirv_magic": hex(words[0]) if words else None,
            "spirv_version": f"{words[1] >> 16}.{(words[1] >> 8) & 0xFF}" if len(words) > 1 else None,
            "spirv_id_bound": words[3] if len(words) > 3 else None,
        })
    return out


def _binding_slots(fb, table, idx):
    return fb.vec_table(table, idx, lambda p: {
        "binding": dflt(fb.scalar(p, 0, "I"), 0),
        "mrt_index": dflt(fb.scalar(p, 1, "I"), 0),
    })


def parse_model_sequence(fb):
    root = fb.root()
    segs = []
    for s in fb.vec_table(root, 0, lambda p: p):
        set_infos = []
        for si in fb.vec_table(s, 3, lambda p: p):
            set_infos.append({
                "set_index": dflt(fb.scalar(si, 1, "I"), U32_MAX),
                "bindings": _binding_slots(fb, si, 0),
            })
        segs.append({
            "type": MODULE_TYPE.get(dflt(fb.scalar(s, 0, "B"), 0), "?"),
            "name": fb.string(s, 1),
            "module_index": dflt(fb.scalar(s, 2, "I"), 0),
            "set_infos": set_infos,
            "inputs": _binding_slots(fb, s, 4),
            "outputs": _binding_slots(fb, s, 5),
            "constants": fb.vec_scalar(s, 6, "I"),
            "dispatch_shape": fb.vec_scalar(s, 7, "I"),
            "push_constant_ranges": fb.vec_table(s, 8, lambda p: {
                "stage_flags": dflt(fb.scalar(p, 0, "I"), 0),
                "offset": dflt(fb.scalar(p, 1, "I"), 0),
                "size": dflt(fb.scalar(p, 2, "I"), 0),
            }),
            "constant_bindings": fb.vec_table(s, 9, lambda p: {
                "graph_constant_id": dflt(fb.scalar(p, 0, "I"), 0),
                "constant_table_index": dflt(fb.scalar(p, 1, "I"), 0),
            }),
        })
    return {
        "segments": segs,
        "inputs": _binding_slots(fb, root, 1),
        "outputs": _binding_slots(fb, root, 2),
        "input_names": fb.vec_string(root, 3),
        "output_names": fb.vec_string(root, 4),
    }


def parse_resources(fb):
    root = fb.root()
    out = []
    for e in fb.vec_table(root, 0, lambda p: p):
        desc = fb.indirect(e, 3)
        shape, strides = [], []
        if desc is not None:
            shape = fb.vec_scalar(desc, 0, "q")
            strides = fb.vec_scalar(desc, 1, "q")
        vkf = fb.scalar(e, 1, "I")
        vkdt = fb.scalar(e, 0, "I")
        out.append({
            "category": RESOURCE_CATEGORY.get(dflt(fb.scalar(e, 2, "B"), 0), "?"),
            "vk_descriptor_type": VK_DESCRIPTOR_TYPE.get(vkdt, vkdt),
            "vk_format": VK_FORMAT.get(vkf, vkf),
            "shape": shape,
            "strides": strides,
            "alias_group_id": dflt(fb.scalar(e, 6, "I"), U32_MAX),
        })
    return out


def parse_constants(data, base, size):
    """常量段不是 FlatBuffer：'CONST00\\0' + i64 条目数 + 条目表 + 数据区。"""
    seg = data[base:base + size]
    name = seg[:8].rstrip(b"\0").decode("ascii", "replace")
    count = struct.unpack_from("<q", seg, 8)[0]
    entries = []
    for i in range(count):
        cid, alias, csize, coff = struct.unpack_from("<IIQQ", seg, 16 + i * 24)
        entries.append({
            "id": cid,
            "alias_group_id": alias,
            "size": csize,
            "offset": coff,
            "abs_offset": base + 16 + count * 24 + coff,
        })
    return {"group_name": name, "count": count, "entries": entries,
            "data_base": 16 + count * 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vgf")
    ap.add_argument("-o", "--output")
    ap.add_argument("--dump-constant", type=int, default=None,
                    help="把指定 id 的常量原始字节写到 <vgf>.<id>.bin")
    args = ap.parse_args()

    data = open(args.vgf, "rb").read()
    if data[:4] != b"VGF1":
        sys.exit(f"不是 VGF1 文件，魔数为 {data[:4]!r}")

    secs = [struct.unpack_from("<QQ", data, 0x10 + i * 16) for i in range(4)]
    result = {
        "file": args.vgf,
        "size": len(data),
        "sections": {n: {"offset": o, "size": s} for n, (o, s) in zip(SECTION_NAMES, secs)},
    }

    for i, name in enumerate(SECTION_NAMES[:3]):
        off, size = secs[i]
        fb = FlatBuffer(data, off, off + size)
        if name == "modules":
            result["modules"] = parse_modules(fb)
        elif name == "model_sequence":
            result["model_sequence"] = parse_model_sequence(fb)
        else:
            result["resources"] = parse_resources(fb)

    off, size = secs[3]
    consts = parse_constants(data, off, size)
    result["constants"] = {
        "group_name": consts["group_name"],
        "count": consts["count"],
        "data_base_rel": consts["data_base"],
        "entries": consts["entries"],
    }

    if args.dump_constant is not None:
        ent = next((e for e in consts["entries"] if e["id"] == args.dump_constant), None)
        if ent is None:
            sys.exit(f"没有 id={args.dump_constant} 的常量")
        blob = data[ent["abs_offset"]:ent["abs_offset"] + ent["size"]]
        path = f"{args.vgf}.{args.dump_constant}.bin"
        open(path, "wb").write(blob)
        print(f"已写出 {path}  ({len(blob)} 字节)")

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        open(args.output, "w", encoding="utf-8").write(text)
        print(f"已写出 {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
