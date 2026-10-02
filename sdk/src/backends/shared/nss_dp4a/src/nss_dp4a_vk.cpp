#include "nss_dp4a_vk.h"

#if defined(_WIN32)
#  include <windows.h>
#endif

namespace nss {

bool VkLoader::loadLibrary() {
#if defined(_WIN32)
    /* 优先用系统加载器。若宿主已经把 vulkan-1.dll 载入，这里会复用同一份。 */
    HMODULE m = ::LoadLibraryA("vulkan-1.dll");
    if (!m) return false;
    module = m;
    getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        ::GetProcAddress(m, "vkGetInstanceProcAddr"));
    return getInstanceProcAddr != nullptr;
#else
    return false;   /* 非 Windows 由宿主通过 createInfo 提供 */
#endif
}

void VkLoader::unload() {
#if defined(_WIN32)
    if (module) {
        ::FreeLibrary(static_cast<HMODULE>(module));
        module = nullptr;
    }
#endif
    getInstanceProcAddr = nullptr;
}

/* 从宿主的 loader 解析一个全局/实例级函数 */
static PFN_vkVoidFunction getInstProc(PFN_vkGetInstanceProcAddr gipa,
                                      VkInstance inst, const char *name) {
    if (!gipa) return nullptr;
    return gipa(inst, name);
}

bool resolveFuncs(const VkLoader &loader,
                  VkInstance instance,
                  VkDevice device,
                  VkFuncs *out,
                  std::string *err) {
    /* 宿主传入的 loader 入口优先。原来这里只认 loader.getInstanceProcAddr，
     * 而宿主给的 gipa 从未被写回 loader，导致必然落到下面的失败分支。 */
    PFN_vkGetInstanceProcAddr gipa = loader.effective();
    if (!gipa) {
        if (err) *err = "vkGetInstanceProcAddr 不可用（宿主未提供且自行加载失败）";
        return false;
    }
    if (!instance) {
        if (err) *err = "instance 句柄为空";
        return false;
    }
    if (!device) {
        if (err) *err = "device 句柄为空";
        return false;
    }

    /* 实例级函数 */
    {
        PFN_vkGetDeviceProcAddr gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            getInstProc(gipa, instance, "vkGetDeviceProcAddr"));
        if (!gdpa) {
            if (err) *err = "取不到 vkGetDeviceProcAddr";
            return false;
        }

#define RESOLVE_INST(name)                                                        \
        out->name = reinterpret_cast<PFN_##name>(                                 \
            getInstProc(gipa, instance, #name));                                  \
        if (!out->name) { if (err) *err = "取不到 " #name; return false; }
        NSS_VK_INSTANCE_FUNCS(RESOLVE_INST)
#undef RESOLVE_INST

#define RESOLVE_DEV(name)                                                         \
        out->name = reinterpret_cast<PFN_##name>(gdpa(device, #name));            \
        if (!out->name) { if (err) *err = "取不到 " #name; return false; }
        NSS_VK_DEVICE_FUNCS(RESOLVE_DEV)
#undef RESOLVE_DEV
    }
    return true;
}

uint32_t findComputeQueueFamily(const VkFuncs &vk, VkPhysicalDevice phys) {
    uint32_t count = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    if (count == 0) return UINT32_MAX;
    std::vector<VkQueueFamilyProperties> props(count);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    return UINT32_MAX;
}

bool queryDeviceCaps(const VkFuncs &vk,
                     VkPhysicalDevice phys,
                     uint32_t apiVersion,
                     DeviceCaps *out,
                     std::string *err) {
    if (!out) { if (err) *err = "out 为空"; return false; }

    /* 基础属性 */
    VkPhysicalDeviceProperties props{};
    vk.vkGetPhysicalDeviceProperties(phys, &props);
    std::strncpy(out->deviceName, props.deviceName, sizeof(out->deviceName) - 1);
    out->deviceName[sizeof(out->deviceName) - 1] = '\0';
    out->maxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
    out->maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
    out->maxStorageBufferRange = props.limits.maxStorageBufferRange;

    /* DP4A 属性（Vulkan 1.3 核心 / VK_KHR_shader_integer_dot_product）
     * 注意：结构里没有 shaderIntegerDotProduct 字段，那个在 Features 里。 */
    VkPhysicalDeviceShaderIntegerDotProductProperties dotProps{};
    dotProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &dotProps;
    vk.vkGetPhysicalDeviceProperties2(phys, &props2);

    out->dotProduct4x8BitPackedSigned =
        dotProps.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;

    /* shaderInt64（requantize 用 64 位乘法） */
    VkPhysicalDeviceShaderIntegerDotProductFeatures dotFeat{};
    dotFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.pNext = &dotFeat;
    vk.vkGetPhysicalDeviceFeatures2(phys, &feat2);
    out->integerDotProduct = dotFeat.shaderIntegerDotProduct == VK_TRUE;
    out->shaderInt64 = feat2.features.shaderInt64 == VK_TRUE;

    /* ───────────── Tensor Core（VK_KHR_cooperative_matrix）探测 ─────────────
     * 目的：让 NSS 能在 DP4A 与 int8-MMA 两条推理路径间自动选路。
     * 缺扩展 / 缺特性 / 没有可用的 int8 形状 ⇒ cmUsable=false，退回 DP4A。
     * 注意与 shaderInt64 同一类坑：这里读到的是「宿主有没有真的启用」，
     * 而不是「硬件支不支持」—— 扩展必须由宿主的 vkCreateDevice 打开。 */
    {
        uint32_t extCount = 0;
        if (vk.vkEnumerateDeviceExtensionProperties &&
            vk.vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, nullptr) == VK_SUCCESS &&
            extCount > 0) {
            std::vector<VkExtensionProperties> exts(extCount);
            if (vk.vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, exts.data()) == VK_SUCCESS) {
                for (uint32_t i = 0; i < extCount; ++i) {
                    if (std::strcmp(exts[i].extensionName, "VK_KHR_cooperative_matrix") == 0) {
                        out->cooperativeMatrix = true;   /* 先记为「扩展已启用」*/
                        break;
                    }
                }
            }
        }
        if (out->cooperativeMatrix) {
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmFeat{};
            cmFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            VkPhysicalDeviceFeatures2 f2{};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &cmFeat;
            vk.vkGetPhysicalDeviceFeatures2(phys, &f2);
            out->cooperativeMatrix = (cmFeat.cooperativeMatrix == VK_TRUE);
        }
        if (out->cooperativeMatrix && vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            uint32_t propCount = 0;
            if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(phys, &propCount, nullptr) == VK_SUCCESS &&
                propCount > 0) {
                std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
                for (auto &p : props) {
                    p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                    p.pNext = nullptr;
                }
                if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(phys, &propCount, props.data()) == VK_SUCCESS) {
                    /* 只要 A/B 都是 sint8、累加与结果都是 sint32、scope=subgroup 的形状。
                     * 优先 16x16x32（一次 MMA 出 16 个输出通道），否则退 16x8x32。 */
                    for (const auto &p : props) {
                        if (p.AType != VK_COMPONENT_TYPE_SINT8_KHR ||
                            p.BType != VK_COMPONENT_TYPE_SINT8_KHR ||
                            p.CType != VK_COMPONENT_TYPE_SINT32_KHR ||
                            p.ResultType != VK_COMPONENT_TYPE_SINT32_KHR ||
                            p.scope != VK_SCOPE_SUBGROUP_KHR) {
                            continue;
                        }
                        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 32) {
                            out->cmM = 16; out->cmN = 16; out->cmK = 32; out->cmUsable = true;
                            break;
                        }
                        if (p.MSize == 16 && p.NSize == 8 && p.KSize == 32 && out->cmM == 0) {
                            out->cmM = 16; out->cmN = 8; out->cmK = 32; out->cmUsable = true;
                        }
                    }
                }
            }
        }
    }

    if (!out->integerDotProduct) {
        if (err) *err = "设备不支持 shaderIntegerDotProduct";
        return false;
    }
    if (!out->shaderInt64) {
        /* 注意：这里查的是「设备是否上报支持」。宿主建设备时还必须真的把
         * VkPhysicalDeviceFeatures::shaderInt64 置为 VK_TRUE，否则即便硬件支持，
         * 这里也会读到 false。SR 框架曾在 VkRenderSystem 里漏掉这一步。 */
        if (err) *err = "设备未启用 shaderInt64（requantize 需要；宿主要在建 VkDevice 时打开该特性）";
        return false;
    }
    (void)apiVersion;
    return true;
}

uint32_t findMemoryType(const VkFuncs &vk, VkPhysicalDevice phys,
                        uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vk.vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

/* ------------------------------------------------------------ OneShotCmd */

bool OneShotCmd::begin(const VkFuncs &vk, VkDevice dev, VkCommandPool pool,
                       VkQueue queue) {
    dev_ = dev;
    queue_ = queue;

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vk.vkAllocateCommandBuffers(dev, &ai, &cmd_) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk.vkBeginCommandBuffer(cmd_, &bi) != VK_SUCCESS) return false;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vk.vkCreateFence(dev, &fi, nullptr, &fence_) != VK_SUCCESS) return false;
    return true;
}

bool OneShotCmd::end(const VkFuncs &vk) {
    if (!cmd_) return false;
    if (vk.vkEndCommandBuffer(cmd_) != VK_SUCCESS) return false;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    if (vk.vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;

    VkResult r = vk.vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (fence_) { vk.vkDestroyFence(dev_, fence_, nullptr); fence_ = VK_NULL_HANDLE; }
    cmd_ = VK_NULL_HANDLE;
    return r == VK_SUCCESS;
}

}  /* namespace nss */
