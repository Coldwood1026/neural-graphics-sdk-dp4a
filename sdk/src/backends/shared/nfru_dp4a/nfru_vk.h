/*
 * nfru_vk.h -- Vulkan dp4a backend, loader and function table.
 *
 * Deliberately does NOT link against vulkan-1.lib: every entry point is resolved
 * through the host's vkGetInstanceProcAddr (or a LoadLibrary of vulkan-1.dll when the
 * host does not supply one). A game can drop the backend in without matching SDK
 * headers, and the same binary runs on any Vulkan 1.3 driver with integer dot product.
 *
 * This replaces the VK_ARM_data_graph path: no VK_ARM_tensors, no
 * VkDataGraphPipelineARM, no vkCmdDispatchDataGraphARM. Those extensions only exist on
 * Arm's data-graph engine; the kernel here is an ordinary compute shader using
 * OpSDot (VK_KHR_shader_integer_dot_product), which every desktop GPU from 2018 on
 * supports.
 */
#ifndef NFRU_VK_H
#define NFRU_VK_H

#include <vulkan/vulkan.h>
#include "nfru_device.h"

namespace nfru {

struct VkLoader {
    void* module;
    PFN_vkGetInstanceProcAddr gipa;
};

/* Loads vulkan-1.dll and resolves vkGetInstanceProcAddr. `outOwned` reports whether the
 * library handle must be released by this library (true) or belongs to the host. */
bool VkLoadLibrary(VkLoader& loader, bool* outOwned);
void VkUnloadLibrary(VkLoader& loader, bool ownModule);

/*
 * The subset of Vulkan this backend uses. Resolved once per context.
 */
struct VkApi {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceFeatures GetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkResetDescriptorPool ResetDescriptorPool;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueWaitIdle QueueWaitIdle;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkResetFences ResetFences;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
};

/* Query whether a device can run the kernel. Used before context creation. */
NfruDp4aResult VkQueryDeviceCaps(uint64_t instance, uint64_t physicalDevice,
                                 uint32_t apiVersion, void* vkGetInstanceProcAddr,
                                 NfruDp4aDeviceCaps* outCaps);

/* Builds the backend. Returns null and fills outErr on failure. */
Device* VkCreateDevice(const NfruDp4aCreateInfo& createInfo, bool ownModule,
                       const VkLoader& loader, const char** outErr);

}  /* namespace nfru */

#endif /* NFRU_VK_H */
