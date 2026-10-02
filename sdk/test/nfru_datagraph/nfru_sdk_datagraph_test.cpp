/*
 * nfru_sdk_datagraph_test.cpp -- SDK-level end-to-end verification of the NFRU data graph.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * WHAT THIS PROVES, AND WHAT IT DOES NOT
 * =============================================================================
 * The dp4a module was already verified bit-exact against a CPU reference on real
 * hardware, for both APIs (see tools/regress.ps1). That establishes THE MODULE IS RIGHT.
 *
 * What that does NOT establish is that the SDK drives it right. The SDK reaches the
 * module through a different road than the standalone harness does:
 *
 *     fpCreateBackendContext  -> CreateResourceDX12/VK (tensors!)
 *     fpCreateDataGraphPipeline
 *     fpScheduleGpuJob + fpExecuteGpuJobs (FFX_GPU_JOB_DATA_GRAPH)
 *     -> the dp4a record call
 *
 * Every one of those is code written for this integration. So this test goes through the
 * SDK's backend INTERFACE -- the same entry points the FI effect uses -- feeds it the
 * golden input the standalone harness is verified with, and compares the result against
 * the same reference bytes.
 *
 * This is deliberately the backend interface and not ffxCreateContext: the latter would
 * drag in the whole frame-interpolation effect (textures, swapchain, optical flow) and
 * fail for reasons that have nothing to do with inference.
 *
 * Usage:
 *     nfru_sdk_test.exe <input.bin> <reference.bin> <width> <height>
 */

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_types.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#if defined(NFRU_TEST_DX12)

#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#else

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#define VK_NO_PROTOTYPES
// Windows.h first: vulkan.h pulls in the platform surface types from it on Windows, and
// this harness loads vulkan-1.dll by hand (LoadLibraryA / GetProcAddress) rather than
// linking an import library, so that it needs no Vulkan SDK installed.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <vulkan/vulkan.h>

// Needed only to initialise the backend's Vulkan function table; see the note in main.
#include <ffx_api/ffx_api.h>
#include <ffx_api/vk/ffx_api_vk.h>

#endif

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static std::vector<unsigned char> ReadFile(const char* path, bool& ok)
{
    ok = false;
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || f == nullptr)
    {
        fprintf(stderr, "cannot open %s\n", path);
        return {};
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> data((size_t)(n > 0 ? n : 0));
    if (!data.empty() && fread(data.data(), 1, data.size(), f) != data.size())
    {
        fclose(f);
        fprintf(stderr, "short read on %s\n", path);
        return {};
    }
    fclose(f);
    ok = true;
    return data;
}

// Progress tracing. stdout is fully buffered when redirected, so without the fflush an
// access violation swallows everything printed before it and hides the crash site.
#define STEP(msg) do { printf("  [step] %s\n", msg); fflush(stdout); } while (0)
static void Fail(const char* what)
{
    fprintf(stderr, "FAIL: %s\n", what);
    exit(2);
}

// ===========================================================================
#if defined(NFRU_TEST_DX12)
// ===========================================================================
// D3D12 host
// ===========================================================================
struct Host
{
    ID3D12Device*              device       = nullptr;
    ID3D12CommandQueue*        queue        = nullptr;
    ID3D12CommandAllocator*    allocator    = nullptr;
    ID3D12GraphicsCommandList* commandList  = nullptr;
    ID3D12Fence*               fence        = nullptr;
    HANDLE                     fenceEvent   = nullptr;
    UINT64                     fenceValue   = 0;
};

static bool HostInit(Host& host)
{
    IDXGIFactory6* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;

    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.device))))
        {
            wprintf(L"adapter: %s\n", desc.Description);
            break;
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (host.device == nullptr) return false;

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(host.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&host.queue)))) return false;
    if (FAILED(host.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&host.allocator)))) return false;
    if (FAILED(host.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, host.allocator, nullptr, IID_PPV_ARGS(&host.commandList)))) return false;
    host.commandList->Close();
    if (FAILED(host.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&host.fence)))) return false;
    host.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    return host.fenceEvent != nullptr;
}

static void HostSubmitAndWait(Host& host)
{
    host.commandList->Close();
    ID3D12CommandList* lists[] = { host.commandList };
    host.queue->ExecuteCommandLists(1, lists);
    ++host.fenceValue;
    host.queue->Signal(host.fence, host.fenceValue);
    if (host.fence->GetCompletedValue() < host.fenceValue)
    {
        host.fence->SetEventOnCompletion(host.fenceValue, host.fenceEvent);
        WaitForSingleObject(host.fenceEvent, INFINITE);
    }
    host.allocator->Reset();
    host.commandList->Reset(host.allocator, nullptr);
}

static FfxDevice      MakeDevice(Host& host) { return ffxGetDeviceDX12(host.device); }
static size_t         ScratchSize()          { return ffxGetScratchMemorySizeDX12(1); }
static FfxErrorCode   MakeInterface(FfxInterface* i, FfxDevice d, void* s, size_t n) { return ffxGetInterfaceDX12(i, d, s, n, 1); }
static const char*    BackendName()          { return "dx12"; }

// ===========================================================================
#else
// ===========================================================================
// Vulkan host
// ===========================================================================
// One loader for everything: Vulkan permits resolving device-level entry points through
// vkGetInstanceProcAddr, which is exactly how the SDK's own backend does it.
// ===========================================================================
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr_t)(VkInstance, const char*);

// vkCreateInstance is deliberately NOT in this list.
//
// It is a GLOBAL command: it has to be fetched from vkGetInstanceProcAddr with a NULL
// instance. Fetching it with a valid instance is not the same query, and a loader is free
// to answer it with NULL -- which is exactly what happened here and made the harness die
// with "missing vkCreateInstance" after the instance had already been created.
#define VK_FUNCS(X)                                     \
    X(vkEnumeratePhysicalDevices)                       \
    X(vkGetPhysicalDeviceQueueFamilyProperties)         \
    X(vkGetPhysicalDeviceProperties)                    \
    X(vkGetDeviceProcAddr)                              \
    X(vkCreateDevice)                                   \
    X(vkGetDeviceQueue)                                 \
    X(vkDeviceWaitIdle)                                 \
    X(vkDestroyDevice)                                  \
    X(vkDestroyInstance)                                \
    X(vkCreateCommandPool)                              \
    X(vkDestroyCommandPool)                             \
    X(vkAllocateCommandBuffers)                         \
    X(vkFreeCommandBuffers)                             \
    X(vkBeginCommandBuffer)                             \
    X(vkEndCommandBuffer)                               \
    X(vkResetCommandBuffer)                             \
    X(vkQueueSubmit)                                    \
    X(vkQueueWaitIdle)

struct VkFns
{
    // Global, so it is loaded separately with a NULL instance (see VK_FUNCS below).
    PFN_vkCreateInstance vkCreateInstance = nullptr;

#define DECL(name) PFN_##name name = nullptr;
    VK_FUNCS(DECL)
#undef DECL
};

struct Host
{
    HMODULE                   loader   = nullptr;
    PFN_vkGetInstanceProcAddr_t gipa   = nullptr;
    VkFns                     fn       = {};
    VkInstance                instance = VK_NULL_HANDLE;
    VkPhysicalDevice          phys     = VK_NULL_HANDLE;
    VkDevice                  device   = VK_NULL_HANDLE;
    VkQueue                   queue    = VK_NULL_HANDLE;
    uint32_t                  family   = 0;
    VkCommandPool             pool     = VK_NULL_HANDLE;
    VkCommandBuffer           cmd      = VK_NULL_HANDLE;
};

static bool HostInit(Host& host)
{
    host.loader = LoadLibraryA("vulkan-1.dll");
    if (host.loader == nullptr) { fprintf(stderr, "vulkan-1.dll not found\n"); return false; }

    host.gipa = (PFN_vkGetInstanceProcAddr_t)GetProcAddress(host.loader, "vkGetInstanceProcAddr");
    if (host.gipa == nullptr) return false;

    VkApplicationInfo app = {};
    app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion       = VK_API_VERSION_1_3;
    app.pApplicationName = "nfru sdk datagraph test";

    VkInstanceCreateInfo ci = {};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    // Instance-level functions can only come from a null instance.
    auto load0 = [&](const char* n) { return host.gipa(VK_NULL_HANDLE, n); };
    host.fn.vkCreateInstance = (PFN_vkCreateInstance)load0("vkCreateInstance");
    if (host.fn.vkCreateInstance == nullptr) return false;
    if (host.fn.vkCreateInstance(&ci, nullptr, &host.instance) != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed\n"); return false; }

#define LOAD(name) host.fn.name = (PFN_##name)host.gipa(host.instance, #name); if (host.fn.name == nullptr) { fprintf(stderr, "missing " #name "\n"); return false; }
    VK_FUNCS(LOAD)
#undef LOAD

    uint32_t count = 0;
    host.fn.vkEnumeratePhysicalDevices(host.instance, &count, nullptr);
    if (count == 0) { fprintf(stderr, "no Vulkan physical devices\n"); return false; }
    std::vector<VkPhysicalDevice> devices(count);
    host.fn.vkEnumeratePhysicalDevices(host.instance, &count, devices.data());

    // Pick the first device with a compute-capable queue family.
    for (VkPhysicalDevice pd : devices)
    {
        uint32_t qCount = 0;
        host.fn.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qCount);
        host.fn.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qs.data());
        for (uint32_t i = 0; i < qCount; ++i)
        {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                host.phys   = pd;
                host.family = i;
                break;
            }
        }
        if (host.phys != VK_NULL_HANDLE) break;
    }
    if (host.phys == VK_NULL_HANDLE) { fprintf(stderr, "no compute queue family\n"); return false; }

    VkPhysicalDeviceProperties props = {};
    host.fn.vkGetPhysicalDeviceProperties(host.phys, &props);
    printf("device: %s\n", props.deviceName);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = host.family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    if (host.fn.vkCreateDevice(host.phys, &dci, nullptr, &host.device) != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed\n"); return false; }
    host.fn.vkGetDeviceQueue(host.device, host.family, 0, &host.queue);

    VkCommandPoolCreateInfo pci = {};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = host.family;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (host.fn.vkCreateCommandPool(host.device, &pci, nullptr, &host.pool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo ai = {};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = host.pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (host.fn.vkAllocateCommandBuffers(host.device, &ai, &host.cmd) != VK_SUCCESS) return false;

    return true;
}

static FfxErrorCode MakeInterface(FfxInterface* i, FfxDevice d, void* s, size_t n) { return ffxGetInterfaceVK(i, d, s, n, 1); }
static const char*  BackendName() { return "vk"; }

#endif

// ===========================================================================
// Common driver: golden input in, reference bytes out, through the SDK interface.
// ===========================================================================
int main(int argc, char** argv)
{
    // Unbuffered: an access violation must not be able to swallow the trace.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (argc < 5)
    {
        fprintf(stderr, "usage: %s <input.bin> <reference.bin> <width> <height>\n", argv[0]);
        return 1;
    }
    const char*  inPath = argv[1];
    const char*  refPath = argv[2];
    const uint32_t W = (uint32_t)atoi(argv[3]);
    const uint32_t H = (uint32_t)atoi(argv[4]);

    bool ok = false;
    std::vector<unsigned char> input = ReadFile(inPath, ok);
    if (!ok) return 1;
    std::vector<unsigned char> reference = ReadFile(refPath, ok);
    if (!ok) return 1;

    // Sanity: NFRU v1 is 16 int8 channels in, 4 logits out, NHWC.
    const size_t expectedIn  = (size_t)W * H * 16u;
    const size_t expectedOut = (size_t)W * H * 4u;
    if (input.size() != expectedIn)
    {
        fprintf(stderr, "input is %zu bytes, expected %zu (%ux%u x16)\n", input.size(), expectedIn, W, H);
        return 1;
    }
    if (reference.size() != expectedOut)
    {
        fprintf(stderr, "reference is %zu bytes, expected %zu (%ux%u x4)\n", reference.size(), expectedOut, W, H);
        return 1;
    }

    Host host = {};
    if (!HostInit(host)) Fail("host init");

#if !defined(NFRU_TEST_DX12)
    VkDeviceContext devCtx = {};
    devCtx.vkDevice              = host.device;
    devCtx.vkPhysicalDevice      = host.phys;
    devCtx.vkInstance            = host.instance;
    devCtx.vkGetInstanceProcAddr = host.gipa;
    // NOT optional. The SDK's Vulkan backend resolves every device-level entry point
    // through this pointer (vk_wrapper.cpp stores it, ffx_vk.cpp asserts it is non-null
    // and dereferences it), so leaving it null is an immediate access violation.
    devCtx.vkDeviceProcAddr      = host.fn.vkGetDeviceProcAddr;
    // -----------------------------------------------------------------------
    // The SDK's Vulkan backend resolves every entry point through a process-wide
    // function table, and it must be populated first.
    //
    // InitVulkanWrapper() is what populates it, and it is NOT in the SDK's export
    // table. The only exported way to reach it is ffxCreateContext with a VK backend
    // desc: ffx-api/src/backends.cpp calls it immediately, before it does anything
    // else. Skip this and ffxGetScratchMemorySizeVK calls through a null pointer --
    // which is exactly how this harness first died, with no output at all.
    //
    // The wrapper is process-wide, so a throwaway context is enough; the interface
    // this test actually drives is built explicitly below.
    // -----------------------------------------------------------------------
    ffxCreateBackendVKDesc backendDesc = {};
    backendDesc.header.type           = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backendDesc.vkDevice              = host.device;
    backendDesc.vkPhysicalDevice      = host.phys;
    backendDesc.vkInstance            = host.instance;
    backendDesc.vkDeviceProcAddr      = host.fn.vkGetDeviceProcAddr;
    backendDesc.vkGetInstanceProcAddr = host.gipa;

    ffxAllocationCallbacks wrapperAlloc = {};
    wrapperAlloc.alloc   = [](void*, uint64_t size) -> void* { return malloc((size_t)size); };
    wrapperAlloc.dealloc = [](void*, void* mem) { free(mem); };

    STEP("calling ffxCreateContext (wrapper init)");
    ffxContext wrapperContext = nullptr;
    const ffxReturnCode_t rc = ffxCreateContext(&wrapperContext, &backendDesc.header, &wrapperAlloc);
    printf("  [step] ffxCreateContext returned rc=%d (0 = FFX_API_RETURN_OK), ctx=%p", (int)rc, (void*)wrapperContext); fflush(stdout);

    FfxDevice ffxDevice = ffxGetDeviceVK(&devCtx);
    STEP("calling ffxGetScratchMemorySizeVK");
    const size_t scratchSize = ffxGetScratchMemorySizeVK(devCtx, 1);
    STEP("scratch size returned");
#else
    FfxDevice ffxDevice = MakeDevice(host);
    const size_t scratchSize = ScratchSize();
#endif

    printf("backend      : %s\n", BackendName());
    printf("scratch      : %zu bytes\n", scratchSize);

    STEP("scratch size known; allocating");
    std::vector<unsigned char> scratch(scratchSize);
    STEP("calling ffxGetInterface");
    FfxInterface iface = {};
    if (MakeInterface(&iface, ffxDevice, scratch.data(), scratchSize) != FFX_OK) Fail("ffxGetInterface");

    STEP("calling fpCreateBackendContext");
    FfxUInt32 contextId = 0;
    if (iface.fpCreateBackendContext(&iface, ARM_EFFECT_FRAMEINTERPOLATION, nullptr, &contextId) != FFX_OK)
        Fail("fpCreateBackendContext");

    // -----------------------------------------------------------------------
    // Tensors. This is where the two backends differ most, and where a mistake
    // would be invisible until the numbers come out wrong.
    // -----------------------------------------------------------------------
    STEP("creating input tensor (FFX_RESOURCE_TYPE_TENSOR)");
    FfxResourceInternal inTensor = {}, outTensor = {}, readback = {};

    FfxCreateResourceDescription inDesc = {};
    inDesc.heapType                          = FFX_HEAP_TYPE_DEFAULT;
    inDesc.resourceDescription.type          = FFX_RESOURCE_TYPE_TENSOR;
    inDesc.resourceDescription.shapeSize     = (uint32_t)expectedIn;
    inDesc.resourceDescription.usage         = FFX_RESOURCE_USAGE_READ_ONLY;
    inDesc.initialState                      = FFX_RESOURCE_STATE_COMMON;
    inDesc.name                              = "NFRU input tensor";
    inDesc.id                                = 1;
    inDesc.initData                          = FfxResourceInitData::FfxResourceInitBuffer(expectedIn, input.data());
    if (iface.fpCreateResource(&iface, &inDesc, contextId, &inTensor) != FFX_OK) Fail("create input tensor");

    STEP("input tensor created; creating output tensor");
    FfxCreateResourceDescription outDesc = {};
    outDesc.heapType                         = FFX_HEAP_TYPE_DEFAULT;
    outDesc.resourceDescription.type         = FFX_RESOURCE_TYPE_TENSOR;
    outDesc.resourceDescription.shapeSize    = (uint32_t)expectedOut;
    outDesc.resourceDescription.usage        = FFX_RESOURCE_USAGE_UAV;
    outDesc.initialState                     = FFX_RESOURCE_STATE_COMMON;
    outDesc.name                             = "NFRU output tensor";
    outDesc.id                               = 2;
    outDesc.initData                         = FfxResourceInitData::FfxResourceInitValue(expectedOut, 0);
    if (iface.fpCreateResource(&iface, &outDesc, contextId, &outTensor) != FFX_OK) Fail("create output tensor");

    // A plain buffer to copy the result into, so it can be mapped and compared.
    STEP("output tensor created; creating readback buffer");
    FfxCreateResourceDescription readDesc = {};
    readDesc.heapType                        = FFX_HEAP_TYPE_READBACK;
    readDesc.resourceDescription.type        = FFX_RESOURCE_TYPE_BUFFER;
    readDesc.resourceDescription.width       = (uint32_t)expectedOut;
    readDesc.resourceDescription.usage       = FFX_RESOURCE_USAGE_READ_ONLY;
    readDesc.initialState                    = FFX_RESOURCE_STATE_COMMON;
    readDesc.name                            = "NFRU readback buffer";
    readDesc.id                              = 3;
    readDesc.initData                        = FfxResourceInitData::FfxResourceInitValue(expectedOut, 0);
    if (iface.fpCreateResource(&iface, &readDesc, contextId, &readback) != FFX_OK) Fail("create readback buffer");

    STEP("resources created; running init-data upload jobs");
    // Run the init-data upload jobs the resource creation queued.
    {
#if defined(NFRU_TEST_DX12)
        iface.fpExecuteGpuJobs(&iface, (FfxCommandList)host.commandList, contextId);
        HostSubmitAndWait(host);
#else
        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        host.fn.vkBeginCommandBuffer(host.cmd, &bi);
        iface.fpExecuteGpuJobs(&iface, (FfxCommandList)host.cmd, contextId);
        host.fn.vkEndCommandBuffer(host.cmd);
        VkSubmitInfo si = {};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &host.cmd;
        host.fn.vkQueueSubmit(host.queue, 1, &si, VK_NULL_HANDLE);
        host.fn.vkQueueWaitIdle(host.queue);
        host.fn.vkResetCommandBuffer(host.cmd, 0);
#endif
    }

    // -----------------------------------------------------------------------
    // The data-graph pipeline. This is the entry point this whole integration added.
    // -----------------------------------------------------------------------
    STEP("upload jobs done; creating the data-graph pipeline");
    FfxPipelineDescription pipeDesc = {};
    // name is a char array here, not a pointer.
    strncpy(pipeDesc.name, "NFRU v1 int8 dp4a", FFX_RESOURCE_NAME_SIZE - 1);
    pipeDesc.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

    FfxPipelineState pipeline = {};
    FfxErrorCode createResult = iface.fpCreateDataGraphPipeline(
        &iface, ARM_EFFECT_FRAMEINTERPOLATION, FFX_FRAMEINTERPOLATION_PASS_NFRU_INTERPOLATION,
        /*permutationOptions*/ 0, &pipeDesc, contextId, W, H, &pipeline);
    if (createResult != FFX_OK)
    {
        fprintf(stderr, "FAIL: fpCreateDataGraphPipeline returned %d\n", createResult);
        return 2;
    }
    printf("pipeline     : %s  (srv tensors %u, uav tensors %u)\n",
           pipeline.name, pipeline.srvTensorCount, pipeline.uavTensorCount);
    for (uint32_t i = 0; i < pipeline.srvTensorCount; ++i)
        printf("  srv[%u] slot %u name '%s'\n", i, pipeline.srvTensorBindings[i].slotIndex, pipeline.srvTensorBindings[i].name);
    for (uint32_t i = 0; i < pipeline.uavTensorCount; ++i)
        printf("  uav[%u] slot %u name '%s'\n", i, pipeline.uavTensorBindings[i].slotIndex, pipeline.uavTensorBindings[i].name);

    // -----------------------------------------------------------------------
    // Dispatch it.
    // -----------------------------------------------------------------------
    STEP("pipeline created; scheduling the data-graph job");
    FfxGpuJobDescription job = {};
    job.jobType = FFX_GPU_JOB_DATA_GRAPH;
    job.dataGraphJobDescription.pipeline    = pipeline;
    job.dataGraphJobDescription.srvTensors[0].resource = inTensor;
    job.dataGraphJobDescription.uavTensors[0].resource = outTensor;

    iface.fpScheduleGpuJob(&iface, &job);

    // Copy the output tensor into the readback buffer, in the same submission.
    FfxGpuJobDescription copyJob = {};
    copyJob.jobType                    = FFX_GPU_JOB_COPY;
    copyJob.copyJobDescriptor.src      = outTensor;
    copyJob.copyJobDescriptor.dst      = readback;
    copyJob.copyJobDescriptor.srcOffset = 0;
    copyJob.copyJobDescriptor.dstOffset = 0;
    copyJob.copyJobDescriptor.size      = (uint32_t)expectedOut;
    iface.fpScheduleGpuJob(&iface, &copyJob);

    {
#if defined(NFRU_TEST_DX12)
        iface.fpExecuteGpuJobs(&iface, (FfxCommandList)host.commandList, contextId);
        HostSubmitAndWait(host);
#else
        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        host.fn.vkBeginCommandBuffer(host.cmd, &bi);
        iface.fpExecuteGpuJobs(&iface, (FfxCommandList)host.cmd, contextId);
        host.fn.vkEndCommandBuffer(host.cmd);
        VkSubmitInfo si = {};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &host.cmd;
        host.fn.vkQueueSubmit(host.queue, 1, &si, VK_NULL_HANDLE);
        host.fn.vkQueueWaitIdle(host.queue);
#endif
    }

    // -----------------------------------------------------------------------
    // Compare.
    // -----------------------------------------------------------------------
    STEP("jobs executed; mapping the readback buffer");
    void* mapped = nullptr;
    if (iface.fpMapResource(&iface, readback, &mapped) != FFX_OK || mapped == nullptr)
        Fail("fpMapResource on the readback buffer");

    const unsigned char* got = (const unsigned char*)mapped;
    size_t mismatches = 0;
    size_t firstBad = (size_t)-1;
    for (size_t i = 0; i < expectedOut; ++i)
    {
        if (got[i] != reference[i]) { if (firstBad == (size_t)-1) firstBad = i; ++mismatches; }
    }

    iface.fpUnmapResource(&iface, readback);

    printf("result       : mismatches: %zu / %zu\n", mismatches, expectedOut);
    if (mismatches != 0)
    {
        printf("first difference at byte %zu: got %d, expected %d\n", firstBad, (int)got[firstBad], (int)reference[firstBad]);
        printf("FAIL\n");
        return 1;
    }
    printf("PASS %s SDK data graph matches the CPU reference byte for byte\n", BackendName());
    return 0;
}
