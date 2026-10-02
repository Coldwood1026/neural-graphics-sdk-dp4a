/*
 * nss_dp4a_vk.h —— 内部 Vulkan 加载层
 *
 * 不链接 vulkan-1.lib，全部通过 vkGetInstanceProcAddr / vkGetDeviceProcAddr 取函数指针。
 * 这样 DLL 没有链接期 Vulkan 依赖，也不挑 Vulkan SDK 版本。
 */
#ifndef NSS_DP4A_VK_H
#define NSS_DP4A_VK_H

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nss {

/* 本库用到的全部 Vulkan 函数指针。用宏统一声明，避免手写遗漏。 */
#define NSS_VK_INSTANCE_FUNCS(F)                        \
    F(vkGetPhysicalDeviceProperties)                    \
    F(vkGetPhysicalDeviceProperties2)                   \
    F(vkGetPhysicalDeviceFeatures2)                     \
    F(vkGetPhysicalDeviceMemoryProperties)              \
    F(vkGetPhysicalDeviceQueueFamilyProperties)         \
    F(vkCreateDevice)                                   \
    F(vkGetDeviceProcAddr)                              \
    F(vkDestroyInstance)                                \
    F(vkEnumerateDeviceExtensionProperties)             \
    F(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

#define NSS_VK_DEVICE_FUNCS(F)                          \
    F(vkDestroyDevice)                                  \
    F(vkCreateBuffer)                                   \
    F(vkDestroyBuffer)                                  \
    F(vkGetBufferMemoryRequirements)                    \
    F(vkAllocateMemory)                                 \
    F(vkFreeMemory)                                     \
    F(vkBindBufferMemory)                               \
    F(vkMapMemory)                                      \
    F(vkUnmapMemory)                                    \
    F(vkCreateShaderModule)                             \
    F(vkDestroyShaderModule)                            \
    F(vkCreateDescriptorSetLayout)                      \
    F(vkDestroyDescriptorSetLayout)                     \
    F(vkCreatePipelineLayout)                           \
    F(vkDestroyPipelineLayout)                          \
    F(vkCreateComputePipelines)                         \
    F(vkDestroyPipeline)                                \
    F(vkCreateDescriptorPool)                           \
    F(vkDestroyDescriptorPool)                          \
    F(vkResetDescriptorPool)                            \
    F(vkAllocateDescriptorSets)                         \
    F(vkUpdateDescriptorSets)                           \
    F(vkCreateCommandPool)                              \
    F(vkDestroyCommandPool)                             \
    F(vkAllocateCommandBuffers)                         \
    F(vkBeginCommandBuffer)                             \
    F(vkEndCommandBuffer)                               \
    F(vkResetCommandPool)                               \
    F(vkCmdBindPipeline)                                \
    F(vkCmdBindDescriptorSets)                          \
    F(vkCmdPushConstants)                               \
    F(vkCmdDispatch)                                    \
    F(vkCmdCopyBuffer)                                  \
    F(vkCmdPipelineBarrier)                             \
    F(vkCreateFence)                                    \
    F(vkDestroyFence)                                   \
    F(vkQueueSubmit)                                    \
    F(vkWaitForFences)                                  \
    F(vkDeviceWaitIdle)                                 \
    F(vkCreateQueryPool)                                \
    F(vkDestroyQueryPool)                               \
    F(vkCmdResetQueryPool)                              \
    F(vkCmdWriteTimestamp)                              \
    F(vkGetQueryPoolResults)

/* 加载器句柄：负责把函数指针取出来 */
struct VkLoader {
    void *module = nullptr;                  /* HMODULE */
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;

    bool loadLibrary();
    void unload();

    /* 宿主通过 createInfo 传入的 loader 入口。
     * 一旦非空，优先级高于 loadLibrary() —— 宿主（VulkanMod / SR 框架）
     * 自己就是 ICD 或已持有真正的加载器，自己去 LoadLibrary("vulkan-1.dll")
     * 只会拿到另一份实现，对不上宿主传进来的 VkInstance。 */
    PFN_vkGetInstanceProcAddr hostGipa = nullptr;

    /* 取最终生效的入口：宿主优先，其次自加载 */
    PFN_vkGetInstanceProcAddr effective() const {
        return hostGipa ? hostGipa : getInstanceProcAddr;
    }
};

/* 已解析的函数指针集合 */
struct VkFuncs {
#define DECL(name) PFN_##name name = nullptr;
    NSS_VK_INSTANCE_FUNCS(DECL)
    NSS_VK_DEVICE_FUNCS(DECL)
#undef DECL
};

/* 用宿主的实例/设备解析全部函数指针。
 * getInstanceProcAddr 可为 NULL（则用加载器默认）。 */
bool resolveFuncs(const VkLoader &loader,
                  VkInstance instance,
                  VkDevice device,
                  VkFuncs *out,
                  std::string *err);

/* 设备能力探测 */
struct DeviceCaps {
    bool integerDotProduct = false;
    bool dotProduct4x8BitPackedSigned = false;
    bool shaderInt64 = false;
    uint32_t maxComputeWorkGroupInvocations = 0;
    uint32_t maxComputeSharedMemorySize = 0;
    uint32_t maxStorageBufferRange = 0;
    char deviceName[256] = {};

    /* ---- Tensor Core（VK_KHR_cooperative_matrix）----
     * 注意：这两项查的是「设备是否**上报**支持」。宿主建 VkDevice 时还必须
     * 真的启用 VK_KHR_cooperative_matrix 扩展 + VkPhysicalDeviceCooperative
     * MatrixFeaturesKHR::cooperativeMatrix，否则即便硬件支持这里也会读到 false
     * （与 shaderInt64 同一类坑）。 */
    bool     cooperativeMatrix = false;
    /* 选中的 int8 MMA 形状（A=sint8 B=sint8 C/Result=sint32，scope=SUBGROUP） */
    uint32_t cmM = 0, cmN = 0, cmK = 0;
    bool     cmUsable = false;
};

bool queryDeviceCaps(const VkFuncs &vk,
                     VkPhysicalDevice phys,
                     uint32_t apiVersion,
                     DeviceCaps *out,
                     std::string *err);

/* 找一个支持 compute 的队列族。返回 UINT32_MAX 表示失败。 */
uint32_t findComputeQueueFamily(const VkFuncs &vk, VkPhysicalDevice phys);

/* ------------------------------------------------------------ 小工具 */

/* 找一个满足 flags 的内存类型 */
uint32_t findMemoryType(const VkFuncs &vk, VkPhysicalDevice phys,
                        uint32_t typeBits, VkMemoryPropertyFlags want);

/* 一次性命令缓冲执行（用于权重上传、调试回读） */
class OneShotCmd {
public:
    bool begin(const VkFuncs &vk, VkDevice dev, VkCommandPool pool, VkQueue queue);
    bool end(const VkFuncs &vk);
    VkCommandBuffer cmd() const { return cmd_; }
private:
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
};

}  /* namespace nss */

#endif /* NSS_DP4A_VK_H */
