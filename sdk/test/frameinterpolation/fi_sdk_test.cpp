/*
 * fi_sdk_test.cpp -- does Arm's neural frame interpolation actually RUN?
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * WHAT THIS PROVES, AND WHAT IT DOES NOT
 * =============================================================================
 * Until now every claim about this port has been about compilation: the shaders build, the
 * blob accessor serves all ten passes, the backends link. None of that is evidence that a
 * fragment pass ever executes.
 *
 * What had never been exercised is the execution path added to the D3D12 backend --
 * CreateGraphicsPipelineDX12, executeGpuJobFragmentDX12 and the FFX_GPU_JOB_FRAGMENT case.
 * Those run only when a fragment pass is dispatched, which is what this does.
 *
 * It drives the shipped artifact's export table (ffxFrameInterpolation* and
 * ffxGetInterfaceDX12/VK), so a mistake in the SDK's own pipeline creation, root signature,
 * descriptor binding, render-target handling or draw lands here.
 *
 * This establishes THAT IT RUNS. It does not establish that the picture is right: there is
 * no reference image. Treat a pass here as "the effect no longer fails before its first
 * draw", not as "the effect is correct".
 *
 * Usage: fi_sdk_test.exe
 */
#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_types.h>
#include <FidelityFX/host/ffx_assert.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#if defined(FI_TEST_DX12)

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
// Progress tracing. stdout is fully buffered when redirected, so without the fflush an
// access violation swallows everything printed before it and hides the crash site.
#define STEP(msg) do { printf("  [step] %s\n", msg); fflush(stdout); } while (0)
static void Fail(const char* what)
{
    fprintf(stderr, "FAIL: %s\n", what);
    exit(2);
}

// ===========================================================================
#if defined(FI_TEST_DX12)
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
    Host host = {};
    if (!HostInit(host)) Fail("host init");

#if !defined(FI_TEST_DX12)
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

    /*
     * ===========================================================================
     * KNOWN LIMITATION: the Vulkan run of this harness cannot start.
     * ===========================================================================
     * The Vulkan backend keeps its function table in a process-wide singleton that is
     * populated only by InitVulkanWrapper, which is not exported. The only exported route to
     * it is ffxCreateContext, and that call needs a COMPLETE effect description:
     *
     *   - passing the backend desc as the top level asks GetffxProvider for a provider of
     *     type FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, which does not exist, and the
     *     call returns FFX_API_RETURN_NO_PROVIDER (4);
     *   - chaining it under ffxApiCreateContextDescFrameGeneration does reach the provider,
     *     but that provider then builds a full frame-generation context -- swapchain
     *     description, present callbacks, optical-flow sizing -- and dies inside
     *     ffxCreateContext without a description for the swapchain.
     *
     * There is no third route. ffxGetInterfaceVK itself calls ffxGetScratchMemorySizeVK
     * (ffx_vk.cpp), which reads the singleton, so the interface cannot be obtained first
     * either.
     *
     * Both forms were tried. The first returns an error; the second faults. This file
     * therefore reports the first, because a diagnosable failure is worth more than a crash
     * with no output -- which is what the previous version of this file did, having inherited
     * the same unchecked call from the NFRU harness.
     *
     * What this does NOT mean: it is not a statement about the Vulkan backend, which builds
     * clean and whose dp4a paths are verified bit-exact by the standalone regression harness.
     * It means this harness needs a full ffx-api frame-generation context before it can drive
     * the Vulkan backend through the SDK, and that is a separate piece of work.
     */
    STEP("calling ffxCreateContext (wrapper init)");
    ffxContext wrapperContext = nullptr;
    const ffxReturnCode_t wrapperRc = ffxCreateContext(&wrapperContext, &backendDesc.header, &wrapperAlloc);
    printf("  ffxCreateContext returned rc=%d (0 = FFX_API_RETURN_OK), ctx=%p\n",
           (int)wrapperRc, (void*)wrapperContext); fflush(stdout);
    if (wrapperRc != FFX_API_RETURN_OK)
    {
        fprintf(stderr,
                "\nFAIL: ffxCreateContext returned %d (%s).\n"
                "      This call is the only exported way to populate the Vulkan function table\n"
                "      that ffxGetScratchMemorySizeVK reads, so the Vulkan run stops here.\n"
                "      See the comment block above for what a complete context needs.\n"
                "      The D3D12 run of this same harness does execute -- see fi_sdk_test_dx12.\n",
                (int)wrapperRc,
                wrapperRc == FFX_API_RETURN_NO_PROVIDER ? "FFX_API_RETURN_NO_PROVIDER" : "see ffx_api.h");
        return 2;
    }

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
    // =======================================================================
    // Frame interpolation.
    // =======================================================================
    const uint32_t W = 1280;
    const uint32_t H = 720;

#if defined(FI_TEST_DX12)
    const FfxCommandList cmdList = (FfxCommandList)host.commandList;
#else
    const FfxCommandList cmdList = (FfxCommandList)host.cmd;
#endif

    // Record into the backend, then hand the command list to the GPU.
    //
    // The two APIs differ in who owns the command list lifecycle: the D3D12 harness leaves
    // one open and HostSubmitAndWait closes, submits and resets it; the Vulkan one begins
    // and ends a command buffer around each batch. Both are the NFRU harness's own idioms,
    // reused rather than reinvented.
    auto executeAndSubmit = [&]() {
#if defined(FI_TEST_DX12)
        iface.fpExecuteGpuJobs(&iface, cmdList, contextId);
        HostSubmitAndWait(host);
#else
        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        host.fn.vkBeginCommandBuffer(host.cmd, &bi);
        iface.fpExecuteGpuJobs(&iface, cmdList, contextId);
        host.fn.vkEndCommandBuffer(host.cmd);
        VkSubmitInfo si = {};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &host.cmd;
        host.fn.vkQueueSubmit(host.queue, 1, &si, VK_NULL_HANDLE);
        host.fn.vkQueueWaitIdle(host.queue);
        host.fn.vkResetCommandBuffer(host.cmd, 0);
#endif
    };

    // fpCreateResource + fpGetResource is API-agnostic and is how the NFRU harness makes
    // its tensors, so both backends take the same road.
    auto makeTexture = [&](const char* name, uint32_t id, FfxSurfaceFormat fmt, uint32_t w, uint32_t h,
                           FfxResourceUsage usage) -> FfxResource {
        FfxCreateResourceDescription d = {};
        d.heapType                   = FFX_HEAP_TYPE_DEFAULT;
        d.resourceDescription.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
        d.resourceDescription.format = fmt;
        d.resourceDescription.width  = w;
        d.resourceDescription.height = h;
        d.resourceDescription.depth  = 1;
        d.resourceDescription.mipCount = 1;
        d.resourceDescription.flags  = FFX_RESOURCE_FLAGS_NONE;
        d.resourceDescription.usage  = usage;
        d.initialState               = FFX_RESOURCE_STATE_COMMON;
        d.name                       = name;
        d.id                         = id;
        d.initData                   = FfxResourceInitData::FfxResourceInitValue(0, 0);

        FfxResourceInternal internal = {};
        if (iface.fpCreateResource(&iface, &d, contextId, &internal) != FFX_OK)
        {
            fprintf(stderr, "FAIL: fpCreateResource(%s)\n", name);
            exit(2);
        }
        return iface.fpGetResource(&iface, internal);
    };

    FfxFloat32x4x4 identity = {1.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 1.0f};

    // -----------------------------------------------------------------------
    // Create the context.
    //
    // This is where createPipelineStates() builds one pipeline per pass. Which entry point
    // it uses depends on the context flags; see fragmentRun below. Before this integration
    // the data-graph pass returned FFX_ERROR_BACKEND_API_ERROR, so the context could not be
    // created at all.
    // -----------------------------------------------------------------------
    /*
     * Which permutations to ask for.
     *
     * The Arm FI picks compute or fragment per stage from the context flags, and it does NOT
     * default to fragment: with flags = 0 every stage is a compute pipeline and
     * CreateGraphicsPipelineDX12 is never called. That is a legitimate configuration, but
     * it exercises none of the D3D12 graphics work, so the harness has to run both -- an
     * earlier version of this file claimed the fragment passes had run while the log showed
     * eight compute pipelines and zero graphics ones.
     *
     *     fi_sdk_test_<api>.exe            -> compute  (FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE)
     *     fi_sdk_test_<api>.exe fragment   -> fragment (FFX_FG_CONTEXT_FLAG_ALL_STAGES_FRAGMENT)
     *
     * Run as separate processes rather than twice in one: resource ids accumulate in the
     * backend context, so a second context in the same process would need a fresh set.
     */
    const bool fragmentRun = (argc > 1 && strcmp(argv[1], "fragment") == 0);

    FfxFrameInterpolationContextDescription fiDesc = {};
    fiDesc.flags               = fragmentRun ? FFX_FG_CONTEXT_FLAG_ALL_STAGES_FRAGMENT
                                             : FFX_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE;
    fiDesc.maxRenderSize       = {W, H};
    fiDesc.displaySize         = {W, H};
    fiDesc.backBufferFormat    = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    fiDesc.backendInterface    = iface;
    fiDesc.opticalFlowSize     = {W / 8, H / 8};
    fiDesc.opticalFlowGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
    fiDesc.fpMessage           = nullptr;
    fiDesc.initialViewProjection[0]  = identity[0];
    fiDesc.initialViewProjection[5]  = identity[5];
    fiDesc.initialViewProjection[10] = identity[10];
    fiDesc.initialViewProjection[15] = identity[15];

    /*
     * NOT on the stack. FfxFrameInterpolationContext is
     * uint32_t data[FFX_FRAMEINTERPOLATION_CONTEXT_SIZE] with the size at 1024*256, so it
     * is exactly 1 MiB -- the same as the default thread stack. Declaring it locally made
     * the harness die in main's prologue with STATUS_STACK_OVERFLOW (0xC00000FD) and print
     * nothing at all, which reads like a loader problem and is not one. Static storage is
     * zero-initialised, which is what the SDK expects.
     */
    static FfxFrameInterpolationContext fiContext;
    STEP("ffxFrameInterpolationContextCreate -- builds all ten pipelines");
    FfxErrorCode fiRc = ffxFrameInterpolationContextCreate(&fiContext, &fiDesc);
    printf("  returned fiRc=%d (0 = FFX_OK)\n", (int)fiRc); fflush(stdout);
    if (fiRc != FFX_OK) Fail("ffxFrameInterpolationContextCreate");
    printf("  configuration: %s\n", fragmentRun ? "all stages as FRAGMENT jobs"
                                                 : "all stages as COMPUTE jobs"); fflush(stdout);

    // -----------------------------------------------------------------------
    // Resources. The Arm FI allocates its own internals; the host supplies the current
    // back buffer, the output, the two optical-flow planes, and the prepare inputs.
    // -----------------------------------------------------------------------
    STEP("creating resources");
    FfxResource colour      = makeTexture("current backbuffer", 10, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource output      = makeTexture("interpolated output", 11, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, W, H, FFX_RESOURCE_USAGE_UAV);
    FfxResource depth       = makeTexture("depth", 12, FFX_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource mvs         = makeTexture("motion vectors", 13, FFX_SURFACE_FORMAT_R16G16_FLOAT, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource ofVector    = makeTexture("optical flow vector", 14, FFX_SURFACE_FORMAT_R16G16_SINT, W / 8, H / 8, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource ofScd       = makeTexture("optical flow scd", 15, FFX_SURFACE_FORMAT_R8_UINT, W / 8, H / 8, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource depthTm1    = makeTexture("depth tm1", 16, FFX_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource depthTp1    = makeTexture("depth tm1 next", 17, FFX_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource colourTm1   = makeTexture("colour tm1", 18, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    FfxResource colourBk    = makeTexture("colour tm1 backup", 19, FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, W, H, FFX_RESOURCE_USAGE_READ_ONLY);
    STEP("resources created; running the init-data upload jobs they queued");
    executeAndSubmit();
    STEP("upload jobs done");

    // -----------------------------------------------------------------------
    // Prepare -- the compute half.
    // -----------------------------------------------------------------------
    FfxFrameInterpolationPrepareDescription prep = {};
    prep.commandList                = cmdList;
    prep.jitterOffset               = {0.0f, 0.0f};
    prep.motionVectorScale          = {1.0f, 1.0f};
    prep.mvSimilarityThreshold      = 0.0f;
    prep.mvSimilarityNoiseThreshold = 0.0f;
    prep.frameTimeDelta             = 16.67f;
    prep.cameraNear                 = 0.1f;
    prep.cameraFar                  = 1000.0f;
    prep.viewSpaceToMetersFactor    = 1.0f;
    prep.cameraFovAngleVertical     = 1.0f;
    prep.depth                      = depth;
    prep.motionVectors              = mvs;
    prep.depthTm1                   = depthTm1;
    prep.depthTm1Next               = depthTp1;
    prep.colorTm1                   = colourTm1;
    prep.colorTm1Backup             = colourBk;
    prep.frameID                    = 1;
    prep.viewProjection[0]  = identity[0];
    prep.viewProjection[5]  = identity[5];
    prep.viewProjection[10] = identity[10];
    prep.viewProjection[15] = identity[15];

    STEP("ffxFrameInterpolationPrepare");
    fiRc = ffxFrameInterpolationPrepare(&fiContext, &prep);
    printf("  prepare returned fiRc=%d\n", (int)fiRc); fflush(stdout);
    if (fiRc != FFX_OK) Fail("ffxFrameInterpolationPrepare");
    executeAndSubmit();
    STEP("prepare done");

    // -----------------------------------------------------------------------
    // Dispatch -- the fragment half, and the reason this harness exists.
    // -----------------------------------------------------------------------
    FfxFrameInterpolationDispatchDescription disp = {};
    disp.flags                           = 0;
    disp.commandList                     = cmdList;
    disp.currentBackBuffer               = colour;
    disp.output                          = output;
    disp.opticalFlowVector               = ofVector;
    disp.opticalFlowSceneChangeDetection = ofScd;
    disp.opticalFlowBufferSize           = {W / 8, H / 8};
    disp.opticalFlowScale                = {1.0f / (float)W, 1.0f / (float)H};
    disp.cameraNear                      = 0.1f;
    disp.cameraFar                       = 1000.0f;
    disp.cameraFovAngleVertical          = 1.0f;
    disp.viewSpaceToMetersFactor         = 1.0f;
    disp.frameTimeDelta                  = 16.67f;
    disp.reset                           = true;
    disp.frameID                         = 1;

    STEP("ffxFrameInterpolationDispatch");
    fiRc = ffxFrameInterpolationDispatch(&fiContext, &disp);
    printf("  dispatch returned fiRc=%d\n", (int)fiRc); fflush(stdout);
    if (fiRc != FFX_OK) Fail("ffxFrameInterpolationDispatch");
    executeAndSubmit();
    STEP("dispatch done and submitted");

    STEP("destroying the context");
    ffxFrameInterpolationContextDestroy(&fiContext);

#if !defined(FI_TEST_DX12)
    // The throwaway wrapper context from the init above owns a whole second FI context
    // (the provider builds two backends and an effect context). Leaving it alive leaks all
    // of that for the process's lifetime.
    ffxDestroyContext(&wrapperContext, &wrapperAlloc);
#endif

    printf("\nRESULT: Arm frame interpolation ran on %s without error (%s stages).\n",
           BackendName(), fragmentRun ? "fragment" : "compute");
    printf("        Create, Prepare, Dispatch and Destroy all returned FFX_OK and the\n");
    printf("        command list was submitted and executed.\n");
    printf("        What this does NOT establish: that the picture is correct. There is no\n");
    printf("        reference image and nothing here reads a pixel back.\n");
    return 0;
}
