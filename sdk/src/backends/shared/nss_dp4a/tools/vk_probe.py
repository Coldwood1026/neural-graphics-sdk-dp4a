#!/usr/bin/env python3
"""用 ctypes 直连 vulkan-1.dll 做设备能力探测（无需编译器、无需第三方包）。

重点查询：
  · 设备列表
  · VK_KHR_shader_integer_dot_product 是否可用
  · integerDotProduct4x8BitPackedSignedAccelerated（Turing 的 DP4A 硬件加速位）
  · VK_KHR_cooperative_matrix 的 int8 配置（为后续 Tensor Core 通路预留）

用法: python vk_probe.py
"""
import ctypes
import sys
from ctypes import POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p

VK_SUCCESS = 0
ST_APP_INFO = 0
ST_INSTANCE_CREATE_INFO = 1
ST_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001            # 注意：FEATURES_2 是 ...000，PROPERTIES_2 是 ...001
ST_SHADER_INTEGER_DOT_PRODUCT_FEATURES = 1000280000
ST_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES = 1000280001   # Features 是 ...0000，Properties 是 ...0001


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", c_uint32)]


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", c_int32), ("pNext", c_void_p),
                ("pApplicationName", c_char_p), ("applicationVersion", c_uint32),
                ("pEngineName", c_char_p), ("engineVersion", c_uint32),
                ("apiVersion", c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32),
                ("pApplicationInfo", POINTER(VkApplicationInfo)),
                ("enabledLayerCount", c_uint32), ("ppEnabledLayerNames", c_void_p),
                ("enabledExtensionCount", c_uint32), ("ppEnabledExtensionNames", c_void_p)]


class DotProps(ctypes.Structure):
    """VkPhysicalDeviceShaderIntegerDotProductProperties。

    注意：这个结构里**没有** shaderIntegerDotProduct 字段 ——
    那个 bool 属于 VkPhysicalDeviceShaderIntegerDotProductFeatures。
    多写一个字段会让后续所有偏移错 4 字节。（已按本地 vulkan_core.h 核对）
    """
    _fields_ = [
        ("sType", c_int32), ("pNext", c_void_p),
        ("integerDotProduct8BitUnsignedAccelerated", c_uint32),
        ("integerDotProduct8BitSignedAccelerated", c_uint32),
        ("integerDotProduct8BitMixedSignednessAccelerated", c_uint32),
        ("integerDotProduct4x8BitPackedUnsignedAccelerated", c_uint32),
        ("integerDotProduct4x8BitPackedSignedAccelerated", c_uint32),
        ("integerDotProduct4x8BitPackedMixedSignednessAccelerated", c_uint32),
        ("integerDotProduct16BitUnsignedAccelerated", c_uint32),
        ("integerDotProduct16BitSignedAccelerated", c_uint32),
        ("integerDotProduct16BitMixedSignednessAccelerated", c_uint32),
        ("integerDotProduct32BitUnsignedAccelerated", c_uint32),
        ("integerDotProduct32BitSignedAccelerated", c_uint32),
        ("integerDotProduct32BitMixedSignednessAccelerated", c_uint32),
        ("integerDotProduct64BitUnsignedAccelerated", c_uint32),
        ("integerDotProduct64BitSignedAccelerated", c_uint32),
        ("integerDotProduct64BitMixedSignednessAccelerated", c_uint32),
    ]


class DotFeatures(ctypes.Structure):
    """VkPhysicalDeviceShaderIntegerDotProductFeatures（sType = ...0000）。"""
    _fields_ = [("sType", c_int32), ("pNext", c_void_p),
                ("shaderIntegerDotProduct", c_uint32)]


class PhysDevProps2(ctypes.Structure):
    _fields_ = [("sType", c_int32), ("pNext", c_void_p),
                ("properties", ctypes.c_byte * 1024)]


def main():
    try:
        vk = ctypes.WinDLL("vulkan-1.dll")
    except OSError as e:
        sys.exit(f"加载 vulkan-1.dll 失败: {e}")

    # 必须声明 argtypes：64 位句柄若被当成 32 位 int 传递会被截断
    vk.vkCreateInstance.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    vk.vkCreateInstance.restype = c_int32
    vk.vkEnumeratePhysicalDevices.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
    vk.vkEnumeratePhysicalDevices.restype = c_int32
    vk.vkGetPhysicalDeviceProperties.argtypes = [c_void_p, c_void_p]
    vk.vkGetPhysicalDeviceProperties.restype = None
    vk.vkGetPhysicalDeviceProperties2.argtypes = [c_void_p, c_void_p]
    vk.vkGetPhysicalDeviceFeatures2.argtypes = [c_void_p, c_void_p]
    vk.vkGetPhysicalDeviceFeatures2.restype = None
    vk.vkGetPhysicalDeviceProperties2.restype = None
    vk.vkEnumerateDeviceExtensionProperties.argtypes = [
        c_void_p, c_char_p, POINTER(c_uint32), POINTER(VkExtensionProperties)]
    vk.vkEnumerateDeviceExtensionProperties.restype = c_int32

    WANTED = ("VK_KHR_shader_integer_dot_product", "VK_KHR_cooperative_matrix",
              "VK_NV_cooperative_matrix", "VK_KHR_shader_float16_int8",
              "VK_KHR_8bit_storage", "VK_EXT_shader_subgroup_ballot")

    app = VkApplicationInfo(ST_APP_INFO, None, b"nss-dp4a-probe", 1, b"ng", 1,
                            0x00400000)                      # Vulkan 1.0
    ci = VkInstanceCreateInfo(ST_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = c_void_p()
    r = vk.vkCreateInstance(byref(ci), None, byref(inst))
    if r != VK_SUCCESS:
        sys.exit(f"vkCreateInstance 失败: {r}")

    n = c_uint32()
    vk.vkEnumeratePhysicalDevices(inst, byref(n), None)
    print(f"物理设备数: {n.value}\n")
    devs = (c_void_p * n.value)()
    vk.vkEnumeratePhysicalDevices(inst, byref(n), devs)

    for i, d in enumerate(devs):
        buf = (ctypes.c_byte * 1024)()
        vk.vkGetPhysicalDeviceProperties(d, buf)
        api = c_uint32.from_buffer(buf, 0).value
        name = ctypes.string_at(ctypes.addressof(buf) + 20).decode("utf-8", "replace")
        dt = c_int32.from_buffer(buf, 16).value
        print(f"[{i}] {name}")
        print(f"    apiVersion={api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}  deviceType={dt}")

        dp = DotProps(ST_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES, None)
        p2 = PhysDevProps2(ST_PHYSICAL_DEVICE_PROPERTIES_2, ctypes.cast(byref(dp), c_void_p))
        vk.vkGetPhysicalDeviceProperties2(d, byref(p2))
        print("    DP4A 硬件加速位（VkPhysicalDeviceShaderIntegerDotProductProperties）:")
        print(f"      4x8BitPackedSignedAccelerated        = "
              f"{'是' if dp.integerDotProduct4x8BitPackedSignedAccelerated else '否'}")
        print(f"      4x8BitPackedMixedSignednessAccel     = "
              f"{'是' if dp.integerDotProduct4x8BitPackedMixedSignednessAccelerated else '否'}")
        print(f"      8BitSignedAccelerated                = "
              f"{'是' if dp.integerDotProduct8BitSignedAccelerated else '否'}")
        print(f"      16BitSignedAccelerated               = "
              f"{'是' if dp.integerDotProduct16BitSignedAccelerated else '否'}")

        # 特性查询：这才是 device 创建时必须打开的位
        df = DotFeatures(ST_SHADER_INTEGER_DOT_PRODUCT_FEATURES, None)
        f2 = PhysDevProps2()
        f2.sType = 1000059000
        f2.pNext = ctypes.cast(byref(df), c_void_p)
        vk.vkGetPhysicalDeviceFeatures2(d, byref(f2))
        print(f"    shaderIntegerDotProduct 特性 = "
              f"{'可启用' if df.shaderIntegerDotProduct else '不可用'}")

        # 扩展枚举交叉验证
        cnt = c_uint32()
        vk.vkEnumerateDeviceExtensionProperties(d, None, byref(cnt), None)
        buf = (VkExtensionProperties * cnt.value)()
        vk.vkEnumerateDeviceExtensionProperties(d, None, byref(cnt), buf)
        names = {buf[i].extensionName.decode() for i in range(cnt.value)}
        print(f"    设备扩展共 {len(names)} 个；关注项:")
        for w in WANTED:
            core = ""
            if w == "VK_KHR_shader_integer_dot_product" and api >= (1 << 22) | (3 << 12):
                core = "  (Vulkan 1.3 起为核心，无需扩展名)"
            print(f"      {'有' if w in names else '无'}  {w}{core}")
        print()


if __name__ == "__main__":
    main()
