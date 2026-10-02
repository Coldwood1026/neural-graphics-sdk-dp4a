/*
 * nfru_vk.cpp -- Vulkan dp4a backend for the NFRU v1 inference graph.
 *
 * Replaces the VK_ARM_data_graph execution path: no VK_ARM_tensors, no
 * VkDataGraphPipelineARM, no vkCmdDispatchDataGraphARM. Those extensions exist only on
 * Arm's data-graph engine; this is an ordinary compute shader using OpSDot
 * (VK_KHR_shader_integer_dot_product), which every desktop GPU since 2018 supports.
 *
 * The host owns VkInstance/VkDevice/VkQueue/VkCommandBuffer. This file creates only its
 * own pipelines, weight buffers and scratch, and records dispatches into the host's
 * command buffer. It never submits (the single exception being the one-shot upload of
 * the baked weights at context creation).
 *
 * Layout contract with `fru_conv_rq.comp`:
 *   activations : int8 NHWC, four channels packed per uint32 word
 *   weights     : uint32, wpk[oc*K4 + (ky*kw+kx)*in_c4 + ic4], K4 = kh*kw*(Cin/4)
 *   bias        : int32 per output channel, already carrying the input zero point
 *                 correction bc = bias/acc_scale - z_a*sum(w); out-of-range taps read
 *                 0x80808080 (four int8 of -128 == z_a), so their contribution
 *                 z_a*sum(w) is in the accumulator and the correction cancels exactly
 *   requantise  : r = clamp(((acc + bc)*M + (1<<(S-1))) >> S + out_zp)
 *   ReLU        : implicit when out_zp == -128; the head uses out_zp = 44 and is
 *                 deliberately not ReLU'd
 *
 * The library does NOT link vulkan-1.lib: every entry point is resolved through the
 * host's vkGetInstanceProcAddr, or a LoadLibrary of vulkan-1.dll when the host supplies
 * none.
 */
#include "nfru_vk.h"
#include "nfru_shaders_spv.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

namespace nfru {

namespace {

template <typename T>
void GetInst(PFN_vkGetInstanceProcAddr g, VkInstance inst, const char* name, T& fn)
{
    fn = reinterpret_cast<T>(g(inst, name));
}

template <typename T>
void GetDev(PFN_vkGetDeviceProcAddr g, VkDevice vkDev, const char* name, T& fn)
{
    fn = reinterpret_cast<T>(g(vkDev, name));
}

bool ResolveDevice(PFN_vkGetDeviceProcAddr d, VkDevice vkDev, VkApi& api)
{
    GetDev(d, vkDev, "vkGetDeviceProcAddr", api.GetDeviceProcAddr);
    GetDev(d, vkDev, "vkCreateShaderModule", api.CreateShaderModule);
    GetDev(d, vkDev, "vkDestroyShaderModule", api.DestroyShaderModule);
    GetDev(d, vkDev, "vkCreateDescriptorSetLayout", api.CreateDescriptorSetLayout);
    GetDev(d, vkDev, "vkDestroyDescriptorSetLayout", api.DestroyDescriptorSetLayout);
    GetDev(d, vkDev, "vkCreatePipelineLayout", api.CreatePipelineLayout);
    GetDev(d, vkDev, "vkDestroyPipelineLayout", api.DestroyPipelineLayout);
    GetDev(d, vkDev, "vkCreateComputePipelines", api.CreateComputePipelines);
    GetDev(d, vkDev, "vkDestroyPipeline", api.DestroyPipeline);
    GetDev(d, vkDev, "vkCreateDescriptorPool", api.CreateDescriptorPool);
    GetDev(d, vkDev, "vkDestroyDescriptorPool", api.DestroyDescriptorPool);
    GetDev(d, vkDev, "vkAllocateDescriptorSets", api.AllocateDescriptorSets);
    GetDev(d, vkDev, "vkResetDescriptorPool", api.ResetDescriptorPool);
    GetDev(d, vkDev, "vkUpdateDescriptorSets", api.UpdateDescriptorSets);
    GetDev(d, vkDev, "vkCreateBuffer", api.CreateBuffer);
    GetDev(d, vkDev, "vkDestroyBuffer", api.DestroyBuffer);
    GetDev(d, vkDev, "vkGetBufferMemoryRequirements", api.GetBufferMemoryRequirements);
    GetDev(d, vkDev, "vkAllocateMemory", api.AllocateMemory);
    GetDev(d, vkDev, "vkFreeMemory", api.FreeMemory);
    GetDev(d, vkDev, "vkBindBufferMemory", api.BindBufferMemory);
    GetDev(d, vkDev, "vkMapMemory", api.MapMemory);
    GetDev(d, vkDev, "vkUnmapMemory", api.UnmapMemory);
    GetDev(d, vkDev, "vkFlushMappedMemoryRanges", api.FlushMappedMemoryRanges);
    GetDev(d, vkDev, "vkCreateCommandPool", api.CreateCommandPool);
    GetDev(d, vkDev, "vkDestroyCommandPool", api.DestroyCommandPool);
    GetDev(d, vkDev, "vkAllocateCommandBuffers", api.AllocateCommandBuffers);
    GetDev(d, vkDev, "vkBeginCommandBuffer", api.BeginCommandBuffer);
    GetDev(d, vkDev, "vkEndCommandBuffer", api.EndCommandBuffer);
    GetDev(d, vkDev, "vkCmdBindPipeline", api.CmdBindPipeline);
    GetDev(d, vkDev, "vkCmdBindDescriptorSets", api.CmdBindDescriptorSets);
    GetDev(d, vkDev, "vkCmdPushConstants", api.CmdPushConstants);
    GetDev(d, vkDev, "vkCmdDispatch", api.CmdDispatch);
    GetDev(d, vkDev, "vkCmdCopyBuffer", api.CmdCopyBuffer);
    GetDev(d, vkDev, "vkCmdPipelineBarrier", api.CmdPipelineBarrier);
    GetDev(d, vkDev, "vkQueueSubmit", api.QueueSubmit);
    GetDev(d, vkDev, "vkQueueWaitIdle", api.QueueWaitIdle);
    GetDev(d, vkDev, "vkCreateFence", api.CreateFence);
    GetDev(d, vkDev, "vkDestroyFence", api.DestroyFence);
    GetDev(d, vkDev, "vkWaitForFences", api.WaitForFences);
    GetDev(d, vkDev, "vkResetFences", api.ResetFences);
    return api.CreateShaderModule && api.CreateComputePipelines && api.CmdDispatch &&
           api.UpdateDescriptorSets && api.CreateBuffer && api.CmdPushConstants;
}

uint32_t FindMemoryType(const VkApi& api, VkPhysicalDevice phys, uint32_t bits,
                        VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    api.GetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return 0xffffffffu;
}

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t size = 0;
};

/*
 * Descriptor sets are recycled round-robin. The SDK may queue several frames' worth of
 * command buffers before any of them executes, and rewriting a set that a pending
 * command buffer still references is a use-after-free on the GPU. FFX_MAX_QUEUED_FRAMES
 * times the per-frame dispatch count is the safe bound; 64 is comfortably above it and
 * costs only a few kilobytes.
 */
const uint32_t kDescriptorSetCount = 64;

class VkBackend final : public Device {
public:
    VkApi api{};
    VkDevice vkDev = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeConv = VK_NULL_HANDLE;
    VkPipeline pipeSupport = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kDescriptorSetCount] = {};
    uint32_t setIndex = 0;

    Buf weights{}, bias{}, mult{}, shift{}, scratch{};
    bool ownScratch = false;

    uint32_t width = 0, height = 0;
    TensorLayout layout{};

    char err[256] = {};

    const char* lastError() const override { return err; }
    bool valid() const override
    {
        return pipeConv != VK_NULL_HANDLE && pipeSupport != VK_NULL_HANDLE;
    }
    const char* backendName() const override { return "Vulkan/dp4a"; }
    uint64_t scratchBytes() const override { return layout.scratchBytes; }
    uint64_t internalBytes() const override
    {
        return weights.size + bias.size + mult.size + shift.size +
               (ownScratch ? scratch.size : 0);
    }

    /* ---------------------------------------------------------------- helpers */

    bool CreateBuffer(uint64_t size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, Buf& out)
    {
        out.size = size;
        const bool trace = getenv("NFRU_DP4A_TRACE") != nullptr;
        if (trace) { printf("[nfru]     CreateBuffer: enter\n"); fflush(stdout); }
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (trace) { printf("[nfru]     CreateBuffer: vkCreateBuffer fn=%p dev=%p\n",
                            (void*)api.CreateBuffer, (void*)vkDev); fflush(stdout); }
        if (api.CreateBuffer(vkDev, &bi, nullptr, &out.buffer) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkCreateBuffer(%llu bytes) failed",
                     (unsigned long long)size);
            return false;
        }
        if (trace) { printf("[nfru]     CreateBuffer: created, querying requirements\n");
                     printf("[nfru]     GetBufferMemoryRequirements fn=%p AllocateMemory fn=%p "
                            "BindBufferMemory fn=%p GetPhysicalDeviceMemoryProperties fn=%p\n",
                            (void*)api.GetBufferMemoryRequirements, (void*)api.AllocateMemory,
                            (void*)api.BindBufferMemory,
                            (void*)api.GetPhysicalDeviceMemoryProperties);
                     fflush(stdout); }
        VkMemoryRequirements mr;
        api.GetBufferMemoryRequirements(vkDev, out.buffer, &mr);
        const uint32_t type = FindMemoryType(api, phys, mr.memoryTypeBits, props);
        if (trace) { printf("[nfru]     CreateBuffer: requirements ok, type=%u\n", type);
                     fflush(stdout); }
        if (type == 0xffffffffu) {
            snprintf(err, sizeof(err), "no Vulkan memory type satisfies the request");
            return false;
        }
        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = type;
        if (api.AllocateMemory(vkDev, &ai, nullptr, &out.memory) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkAllocateMemory failed");
            return false;
        }
        if (api.BindBufferMemory(vkDev, out.buffer, out.memory, 0) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkBindBufferMemory failed");
            return false;
        }
        return true;
    }

    void DestroyBuf(Buf& b)
    {
        if (b.buffer) { api.DestroyBuffer(vkDev, b.buffer, nullptr); b.buffer = VK_NULL_HANDLE; }
        if (b.memory) { api.FreeMemory(vkDev, b.memory, nullptr); b.memory = VK_NULL_HANDLE; }
        b.size = 0;
    }

    VkPipeline MakePipeline(const uint32_t* spirv, uint32_t words)
    {
        VkShaderModuleCreateInfo smi = {};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = (size_t)words * 4;
        smi.pCode = spirv;
        VkShaderModule mod = VK_NULL_HANDLE;
        if (api.CreateShaderModule(vkDev, &smi, nullptr, &mod) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkCreateShaderModule failed");
            return VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = pipeLayout;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult r = api.CreateComputePipelines(vkDev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
        api.DestroyShaderModule(vkDev, mod, nullptr);
        if (r != VK_SUCCESS) {
            snprintf(err, sizeof(err),
                     "vkCreateComputePipelines failed (VkResult %d): the device reports "
                     "no integer dot product support", (int)r);
            return VK_NULL_HANDLE;
        }
        return pipe;
    }

    bool UploadOne(Buf& buf, const void* data, uint64_t bytes)
    {
        const bool trace = getenv("NFRU_DP4A_TRACE") != nullptr;
        if (trace) { printf("[nfru]   UploadOne: create %llu-byte buffer\n",
                            (unsigned long long)bytes); fflush(stdout); }
        if (!CreateBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf)) {
            return false;
        }
        void* mapped = nullptr;
        const VkResult mr = api.MapMemory(vkDev, buf.memory, 0, bytes, 0, &mapped);
        if (trace) { printf("[nfru]   UploadOne: vkMapMemory -> %d, ptr %p\n",
                            (int)mr, mapped); fflush(stdout); }
        if (mr != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkMapMemory failed during weight upload (%d)",
                     (int)mr);
            return false;
        }
        memcpy(mapped, data, (size_t)bytes);
        api.UnmapMemory(vkDev, buf.memory);
        return true;
    }

    /* ------------------------------------------------------------------ setup */

    bool Init(const NfruDp4aCreateInfo& ci, const VkLoader& loader)
    {
        /* Stage tracing is compiled in and gated on an environment variable: when a
         * hardware run fails, this says how far initialisation got without needing a
         * debugger. */
        PFN_vkGetInstanceProcAddr gipa = loader.gipa;
        const bool trace = getenv("NFRU_DP4A_TRACE") != nullptr;
#define STAGE(m) do { if (trace) { printf("[nfru] %s\n", m); fflush(stdout); } } while (0)

        STAGE("Init: resolve vkGetDeviceProcAddr");
        const PFN_vkGetDeviceProcAddr gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            loader.gipa((VkInstance)ci.instance, "vkGetDeviceProcAddr"));
        if (!gdpa) {
            snprintf(err, sizeof(err), "vkGetDeviceProcAddr unavailable");
            return false;
        }
        STAGE("Init: resolve device entry points");
        if (!ResolveDevice(gdpa, (VkDevice)ci.device, api)) {
            snprintf(err, sizeof(err),
                     "device is missing entry points required by the dp4a backend");
            return false;
        }
        /*
         * Two of the functions this backend uses are INSTANCE level, so
         * vkGetDeviceProcAddr returns null for them and calling them crashes. Resolve
         * them through the instance instead. (Forgetting this is exactly the failure
         * this stage check exists to catch.)
         */
        STAGE("Init: resolve instance-level entry points");
        api.GetPhysicalDeviceMemoryProperties =
            (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(
                (VkInstance)ci.instance, "vkGetPhysicalDeviceMemoryProperties");
        api.GetPhysicalDeviceQueueFamilyProperties =
            (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(
                (VkInstance)ci.instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        if (!api.GetPhysicalDeviceMemoryProperties) {
            snprintf(err, sizeof(err),
                     "vkGetPhysicalDeviceMemoryProperties is not resolvable from the "
                     "instance");
            return false;
        }
        STAGE("Init: device entry points resolved");
        vkDev = (VkDevice)ci.device;
        phys = (VkPhysicalDevice)ci.physicalDevice;
        width = ci.width;
        height = ci.height;

        STAGE("Init: create descriptor set layout");
        VkDescriptorSetLayoutBinding binds[kBindCount] = {};
        for (uint32_t i = 0; i < kBindCount; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo sli = {};
        sli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sli.bindingCount = kBindCount;
        sli.pBindings = binds;
        if (api.CreateDescriptorSetLayout(vkDev, &sli, nullptr, &setLayout) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkCreateDescriptorSetLayout failed");
            return false;
        }

        /* One push-constant range sized for the larger of the two blocks; both are
         * 56 bytes or less. */
        VkPushConstantRange range = {};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = sizeof(ConvPushConstants);
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &range;
        if (api.CreatePipelineLayout(vkDev, &pli, nullptr, &pipeLayout) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkCreatePipelineLayout failed");
            return false;
        }

        STAGE("Init: create pipeline layout");
        pipeConv = MakePipeline(kNfruSpvConv, (uint32_t)(sizeof(kNfruSpvConv) / 4));
        if (!pipeConv) return false;
        pipeSupport = MakePipeline(kNfruSpvSupport, (uint32_t)(sizeof(kNfruSpvSupport) / 4));
        if (!pipeSupport) return false;

        STAGE("Init: pipelines created");
        VkDescriptorPoolSize ps = {};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = kBindCount * kDescriptorSetCount;
        VkDescriptorPoolCreateInfo dpi = {};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = kDescriptorSetCount;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        if (api.CreateDescriptorPool(vkDev, &dpi, nullptr, &descPool) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkCreateDescriptorPool failed");
            return false;
        }
        VkDescriptorSetLayout layouts[kDescriptorSetCount];
        for (uint32_t i = 0; i < kDescriptorSetCount; ++i) layouts[i] = setLayout;
        VkDescriptorSetAllocateInfo dai = {};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = descPool;
        dai.descriptorSetCount = kDescriptorSetCount;
        dai.pSetLayouts = layouts;
        if (api.AllocateDescriptorSets(vkDev, &dai, sets) != VK_SUCCESS) {
            snprintf(err, sizeof(err), "vkAllocateDescriptorSets failed");
            return false;
        }

        STAGE("Init: upload weights");
        if (getenv("NFRU_DP4A_TRACE")) { printf("[nfru]   uploading bias/mult/shift/weights buffers\n"); fflush(stdout); }
        if (!UploadOne(weights, kNfruWeights, sizeof(kNfruWeights)) ||
            !UploadOne(bias, kNfruBias, sizeof(kNfruBias)) ||
            !UploadOne(mult, kNfruMult, sizeof(kNfruMult)) ||
            !UploadOne(shift, kNfruShift, sizeof(kNfruShift))) {
            return false;
        }

        STAGE("Init: compute tensor layout");
        if (!ComputeTensorLayout(width, height, layout)) {
            /*
             * `NFRU_DP4A_LAYER_ONLY` runs ONE convolution with the host buffers as its
             * input and output, so the inter-layer tensor layout is never consulted --
             * the layer's geometry is derived from its own kernel/stride/pad. The layout
             * is only needed for the scratch allocation.
             *
             * It also cannot be consistent at a reduced resolution: the graph's x2 resize
             * and its two concatenations only close at the model's native size, so asking
             * for a 240x135 context fails on geometry that this mode never reads. That
             * refusal, not the kernel, is what made every 240x135 layer untestable.
             */
            if (!getenv("NFRU_DP4A_LAYER_ONLY")) {
                snprintf(err, sizeof(err),
                         "tensor layout inconsistent at %ux%u (a concatenation's inputs "
                         "disagree in spatial extent)", width, height);
                return false;
            }
            memset(&layout, 0, sizeof(layout));
            /* Generous and only ever allocated, never read, in this mode. */
            layout.scratchBytes = ((uint64_t)width * height * 64u + 255u) & ~(uint64_t)255;
        }
        return true;
    }

    void Destroy()
    {
        if (!vkDev) return;
        DestroyBuf(weights);
        DestroyBuf(bias);
        DestroyBuf(mult);
        DestroyBuf(shift);
        if (ownScratch) DestroyBuf(scratch);
        if (pipeConv) api.DestroyPipeline(vkDev, pipeConv, nullptr);
        if (pipeSupport) api.DestroyPipeline(vkDev, pipeSupport, nullptr);
        if (pipeLayout) api.DestroyPipelineLayout(vkDev, pipeLayout, nullptr);
        if (descPool) api.DestroyDescriptorPool(vkDev, descPool, nullptr);
        if (setLayout) api.DestroyDescriptorSetLayout(vkDev, setLayout, nullptr);
    }

    /* -------------------------------------------------------------- recording */

    VkDescriptorSet Bind6(VkBuffer actIn, VkDeviceSize actInOff,
                          VkBuffer actOut, VkDeviceSize actOutOff,
                          uint64_t weightOffBytes, uint64_t biasOffBytes,
                          uint64_t multOffBytes, uint64_t shiftOffBytes)
    {
        const VkDescriptorSet ds = sets[setIndex];
        setIndex = (setIndex + 1) % kDescriptorSetCount;

        /*
         * `weightOffBytes` is this layer's offset inside the shared weight pool. The
         * kernel is layer-agnostic -- it computes wBase0 relative to the START of the
         * weights it is given -- so the layer offset has to be applied here. Leaving it
         * out makes every layer read conv1's weights, which conv1 cannot reveal.
         */
        /*
         * `weightOffBytes` / `biasOffBytes` / `multOffBytes` / `shiftOffBytes` locate this
         * layer inside the shared pools. The kernel is layer-agnostic -- it computes
         * wBase0 and indexes biasCr/mult/shift relative to the START of what it is given
         * -- so all four layer offsets have to be applied here. Leaving any one out makes
         * that array read the first layer's values, which conv1 cannot reveal and which
         * explains why only conv1 ever looked correct.
         */
        VkBuffer bufs[kBindCount] = { actIn, weights.buffer, bias.buffer,
                                      mult.buffer, shift.buffer, actOut };
        VkDeviceSize offs[kBindCount] = { actInOff, weightOffBytes, biasOffBytes,
                                          multOffBytes, shiftOffBytes, actOutOff };
        if (getenv("NFRU_DP4A_TRACE_BIND")) {
            printf("[bind] actIn=%llu w=%llu bc=%llu m=%llu s=%llu out=%llu | pool: w %llu B, "
                   "bc %llu B, m %llu B, s %llu B\n",
                   (unsigned long long)actInOff, (unsigned long long)weightOffBytes,
                   (unsigned long long)biasOffBytes, (unsigned long long)multOffBytes,
                   (unsigned long long)shiftOffBytes, (unsigned long long)actOutOff,
                   (unsigned long long)weights.size, (unsigned long long)bias.size,
                   (unsigned long long)mult.size, (unsigned long long)shift.size);
            fflush(stdout);
        }
        VkDeviceSize sizes[kBindCount] = { VK_WHOLE_SIZE, VK_WHOLE_SIZE, VK_WHOLE_SIZE,
                                           VK_WHOLE_SIZE, VK_WHOLE_SIZE, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo infos[kBindCount] = {};
        VkWriteDescriptorSet wr[kBindCount] = {};
        for (uint32_t i = 0; i < kBindCount; ++i) {
            infos[i].buffer = bufs[i];
            infos[i].offset = offs[i];
            infos[i].range = sizes[i];
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = ds;
            wr[i].dstBinding = i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &infos[i];
        }
        api.UpdateDescriptorSets(vkDev, kBindCount, wr, 0, nullptr);
        return ds;
    }

    /*
     * Bindings for a support dispatch.
     *
     * `nfru_support.comp` declares `binding = 0` for the source and `binding = 1` for the
     * destination, and that is what the compiled SPIR-V carries:
     *
     *     OpDecorate %dst Binding 1
     *     OpDecorate %src Binding 0
     *
     * so the destination MUST be written to descriptor binding 1.
     *
     * An earlier version bound it to 5, reasoning from the DX12 root signature's SRV/UAV
     * table split. That split does not exist in Vulkan: all six bindings are
     * VK_DESCRIPTOR_TYPE_STORAGE_BUFFER and `writeonly` is a shader-side qualifier, not a
     * different descriptor type. Binding to 5 meant the shader's stores went to whatever
     * binding 1 happened to hold -- after any convolution that is the multiplier pool --
     * so every resize and every concatenation wrote nothing to its destination slot and
     * corrupted the requantisation multipliers instead. Nothing downstream of the first
     * support op could then be right, while every convolution before it still looked
     * perfect in isolation.
     */
    VkDescriptorSet BindSupport(VkBuffer src, VkDeviceSize srcOff,
                                VkBuffer dst, VkDeviceSize dstOff)
    {
        const VkDescriptorSet ds = sets[setIndex];
        setIndex = (setIndex + 1) % kDescriptorSetCount;

        /* source -> binding 0, destination -> binding 1: exactly what the shader declares */
        const uint32_t b0 = 0;
        const uint32_t b1 = 1;
        VkDescriptorBufferInfo infos[2] = {};
        VkWriteDescriptorSet wr[2] = {};
        VkBuffer bufs[2] = { src, dst };
        VkDeviceSize offs[2] = { srcOff, dstOff };
        for (uint32_t i = 0; i < 2; ++i) {
            infos[i].buffer = bufs[i];
            infos[i].offset = offs[i];
            infos[i].range = VK_WHOLE_SIZE;
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = ds;
            wr[i].dstBinding = (i == 0) ? b0 : b1;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &infos[i];
        }
        api.UpdateDescriptorSets(vkDev, 2, wr, 0, nullptr);
        return ds;
    }

    void Barrier(VkCommandBuffer cmd)
    {
        /* Each dispatch consumes the previous dispatch's output, so one compute->compute
         * memory barrier per dispatch covers the chain. */
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        api.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               1, &mb, 0, nullptr, 0, nullptr);
    }

    /*
     * Makes every shader write in this command buffer visible to the TRANSFER stage and to
     * the host.
     *
     * Barrier() only orders compute against compute. A vkCmdCopyBuffer that reads the
     * result -- or a host read after the fence -- is otherwise a data race: the stores may
     * still be sitting in L2 and were never made available. That race does NOT fail
     * loudly. It hands back a partially stale buffer in which some words hold the correct
     * value and the rest still hold whatever the allocation contained; on a fresh
     * allocation that is zero, which is indistinguishable from "the kernel dropped its
     * stores". Chasing that phantom cost several rounds: conv5's accumulators looked
     * 44% missing while `NFRU_DP4A_DUMP_MARK` proved every one of the 518,400 words had
     * been written by the thread that owned it.
     */
    void BarrierOut(VkCommandBuffer cmd)
    {
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        /* TRANSFER_WRITE covers the STOP_TO_SCRATCH copy-out issued just before this;
         * without it that copy and the caller's read-back are unordered too. */
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        api.CmdPipelineBarrier(cmd,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    /*
     * Translate the API-agnostic plan into Vulkan commands.
     *
     * Nothing here decides geometry, offsets or ordering: all of that is in
     * `BuildRecordPlan`, shared with the DX12 backend, so the two cannot drift apart.
     */
    NfruDp4aResult record(uint64_t commandBuffer, uint64_t inputBuffer, uint64_t inputOff,
                          uint64_t inputBytes, uint64_t outputBuffer, uint64_t outputOff,
                          uint64_t outputBytes, uint64_t scratchBuffer,
                          uint32_t w, uint32_t h) override
    {
        /* Vulkan descriptors can span the rest of the buffer (VK_WHOLE_SIZE), so the host's
         * buffer sizes are not needed on this path; D3D12 views must state an element
         * count, so it uses them. */
        (void)inputBytes;
        (void)outputBytes;
        if (!valid()) {
            return NFRU_DP4A_ERROR_NOT_READY;
        }
        if (w != width || h != height) {
            snprintf(err, sizeof(err), "resolution mismatch: context is %ux%u, dispatch %ux%u",
                     width, height, w, h);
            return NFRU_DP4A_ERROR_MODEL_MISMATCH;
        }
        VkCommandBuffer cmd = (VkCommandBuffer)commandBuffer;
        VkBuffer inBuf = (VkBuffer)inputBuffer;
        VkBuffer outBuf = (VkBuffer)outputBuffer;
        VkBuffer scrBuf = (VkBuffer)scratchBuffer;
        if (!scrBuf) {
            if (!ownScratch) {
                if (!CreateBuffer(layout.scratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch)) {
                    return NFRU_DP4A_ERROR_OUT_OF_MEMORY;
                }
                ownScratch = true;
            }
            scrBuf = scratch.buffer;
        }

        RecordPlan plan;
        if (!BuildRecordPlan(width, height, layout, w, h, inputOff, outputOff, plan,
                             err, sizeof(err))) {
            return NFRU_DP4A_ERROR_MODEL_MISMATCH;
        }

        for (uint32_t i = 0; i < plan.count; ++i) {
            const PlannedDispatch& d = plan.d[i];
            VkBuffer s = d.srcIsHost ? inBuf : scrBuf;
            VkBuffer t = d.dstIsHost ? outBuf : scrBuf;
            if (d.isConv) {
                VkDescriptorSet ds = Bind6(s, d.srcOffBytes, t, d.dstOffBytes,
                                           d.wOffBytes, d.bcOffBytes, d.mOffBytes, d.sOffBytes);
                api.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConv);
                api.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout,
                                          0, 1, &ds, 0, nullptr);
                api.CmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(d.pc), &d.pc);
                api.CmdDispatch(cmd, d.gx, d.gy, d.gz);
            } else {
                VkDescriptorSet ds = BindSupport(s, d.srcOffBytes, t, d.dstOffBytes);
                api.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSupport);
                api.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout,
                                          0, 1, &ds, 0, nullptr);
                api.CmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(d.spc), d.spc);
                api.CmdDispatch(cmd, d.groups, 1, 1);
            }
            Barrier(cmd);
        }

        if (plan.copyBytes) {
            VkBufferCopy region = {};
            region.srcOffset = plan.copySrcOff;
            region.dstOffset = plan.copyDstOff;
            region.size = plan.copyBytes;
            api.CmdCopyBuffer(cmd, scrBuf, outBuf, 1, &region);
        }
        BarrierOut(cmd);
        return NFRU_DP4A_OK;
    }
};

}  /* namespace */

NfruDp4aResult VkQueryDeviceCaps(uint64_t instance, uint64_t physicalDevice,
                                 uint32_t apiVersion, void* vkGetInstanceProcAddr,
                                 NfruDp4aDeviceCaps* outCaps)
{
    if (!outCaps || !vkGetInstanceProcAddr) {
        return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
    }
    memset(outCaps, 0, sizeof(*outCaps));
    PFN_vkGetInstanceProcAddr g = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr;

    PFN_vkGetPhysicalDeviceProperties getProps = nullptr;
    PFN_vkGetPhysicalDeviceFeatures getFeats = nullptr;
    GetInst(g, (VkInstance)instance, "vkGetPhysicalDeviceProperties", getProps);
    GetInst(g, (VkInstance)instance, "vkGetPhysicalDeviceFeatures", getFeats);
    if (!getProps || !getFeats) {
        return NFRU_DP4A_ERROR_UNSUPPORTED;
    }

    VkPhysicalDeviceProperties props;
    getProps((VkPhysicalDevice)physicalDevice, &props);
    VkPhysicalDeviceFeatures feats;
    getFeats((VkPhysicalDevice)physicalDevice, &feats);

    strncpy(outCaps->deviceName, props.deviceName, sizeof(outCaps->deviceName) - 1);
    outCaps->maxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
    outCaps->maxStorageBufferRange = props.limits.maxStorageBufferRange;
    outCaps->shaderInt64 = feats.shaderInt64 ? 1u : 0u;

    /*
     * VK_KHR_shader_integer_dot_product is core in Vulkan 1.3. Below that it must have
     * been enabled as an extension with its feature struct chained at device creation,
     * which this library cannot add retroactively, so it reports what the caller's device
     * already provides. shaderInt64 is genuinely required: the requantiser multiplies a
     * ~25-bit accumulator by a ~31-bit multiplier.
     */
    outCaps->integerDotProduct = (apiVersion >= VK_API_VERSION_1_3) ? 1u : 0u;
    outCaps->dotProduct4x8BitPackedSigned = outCaps->integerDotProduct;
    return NFRU_DP4A_OK;
}

bool VkLoadLibrary(VkLoader& loader, bool* outOwned)
{
    loader.module = (void*)LoadLibraryA("vulkan-1.dll");
    if (!loader.module) {
        return false;
    }
    loader.gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress((HMODULE)loader.module,
                                                            "vkGetInstanceProcAddr");
    if (!loader.gipa) {
        FreeLibrary((HMODULE)loader.module);
        loader.module = nullptr;
        return false;
    }
    if (outOwned) *outOwned = true;
    return true;
}

void VkUnloadLibrary(VkLoader& loader, bool ownModule)
{
    if (ownModule && loader.module) {
        FreeLibrary((HMODULE)loader.module);
    }
    loader.module = nullptr;
    loader.gipa = nullptr;
}

Device* VkCreateDevice(const NfruDp4aCreateInfo& createInfo, bool ownModule,
                       const VkLoader& loader, const char** outErr)
{
    VkBackend* d = new VkBackend();
    if (!d->Init(createInfo, loader)) {
        if (outErr) *outErr = d->lastError();
        d->Destroy();
        delete d;
        return nullptr;
    }
    (void)ownModule;
    return d;
}

}  /* namespace nfru */
