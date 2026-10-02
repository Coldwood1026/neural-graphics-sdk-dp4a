/*
 * nfru_dp4a.h -- Arm NFRU v1 neural frame rate upscaling: int8 DP4A inference backend
 *                (public C API).
 *
 * Replaces the VK_ARM_data_graph / VkDataGraphPipelineARM execution path, which runs the
 * network's TOSA graph on Arm's data-graph engine and therefore exists only on Arm
 * hardware. This backend runs the mathematically identical graph on any Vulkan 1.3
 * device or any D3D12 device with shader model 6.4 (DP4A), and produces bit-identical
 * int8 results on both -- the two backends are cross-checked against each other.
 *
 * Design, following the proven NSS desktop backend:
 *   - pure inference. The host owns VkInstance/VkDevice/VkQueue/VkCommandBuffer (or
 *     ID3D12Device/ID3D12CommandQueue/ID3D12GraphicsCommandList). This library builds
 *     its own pipelines and buffers and records dispatches into the host's command
 *     buffer. It never submits.
 *   - zero copy at the boundary: the host passes its own buffers for input and output;
 *     nothing is read back to the CPU per frame.
 *   - the weights are baked into the binary, so there is nothing to load at runtime.
 *
 * Numerical contract, matched bit for bit against the training-time quantisation
 * observers in `fru_v1_int8.pt`:
 *   accumulation : sum(q_a * q_w) via OpSDot / Dot4AddI8Packed, int32, 4 channels per
 *                  instruction
 *   out-of-range : taps read 0x80808080 == four int8 of -128 == the input zero point,
 *                  so padding contributes z_a*sum(w) by itself
 *   bias         : bc = bias/acc_scale - z_a*sum(w), which cancels that padding term
 *                  exactly. Getting this identity wrong shifts every output by up to
 *                  128*sum(w) and saturates the layer.
 *   requantise   : r = (int64(v) * M + (1 << (S-1))) >> S ; r += out_zp ; clamp int8.
 *                  int64 is required: v is ~25 bits and M up to 31.
 *   ReLU         : implicit. Layers with out_zp == -128 become non-negative from the
 *                  clamp alone and must NOT be max(x,0)'d again. The output head uses
 *                  out_zp == 44 and is deliberately not ReLU'd.
 *
 * Usage (Vulkan):
 *     NfruDp4aContext *ctx = NULL;
 *     NfruDp4aCreateInfo ci = { ... };
 *     nfruDp4aCreateContext(&ci, &ctx);
 *     NfruDp4aDispatchInfo di = { ... };
 *     nfruDp4aRecord(ctx, cmd, &di);        // 19 dispatches + barriers
 *     nfruDp4aDestroyContext(ctx);
 *
 * Usage (DX12) is identical except that `instance` carries the ID3D12Device and `queue`
 * the ID3D12CommandQueue, and `commandBuffer` is an ID3D12GraphicsCommandList.
 */
#ifndef NFRU_DP4A_H
#define NFRU_DP4A_H

#include <stddef.h>
#include <stdint.h>

/*
 * Three linkage modes:
 *
 *   NFRU_DP4A_INTERNAL  built into a larger DLL (the SDK backends do this). The
 *                       functions keep external linkage inside that DLL but carry no
 *                       import/export decoration, so the host's export table stays
 *                       exactly as it was -- which is what lets `dumpbin /exports` on
 *                       ngsdk_windows_x64.dll still report the upstream 62 entries.
 *   NFRU_DP4A_BUILD     building the standalone nfru_dp4a.dll  -> dllexport
 *   neither             consuming the standalone nfru_dp4a.dll -> dllimport
 *
 * Without the NFRU_DP4A_INTERNAL case, compiling these sources into another DLL would
 * leave them declared dllimport, and the linker would look for the very symbols the
 * translation units in front of it are defining.
 */
#if defined(NFRU_DP4A_INTERNAL)
#  define NFRU_DP4A_API
#  define NFRU_DP4A_CALL __cdecl
#elif defined(_WIN32)
#  ifdef NFRU_DP4A_BUILD
#    define NFRU_DP4A_API __declspec(dllexport)
#  else
#    define NFRU_DP4A_API __declspec(dllimport)
#  endif
#  define NFRU_DP4A_CALL __cdecl
#else
#  define NFRU_DP4A_API __attribute__((visibility("default")))
#  define NFRU_DP4A_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ version */
#define NFRU_DP4A_VERSION_MAJOR 1
#define NFRU_DP4A_VERSION_MINOR 0
#define NFRU_DP4A_VERSION_PATCH 0
#define NFRU_DP4A_MAKE_VERSION(maj, min, pat) (((maj) << 22) | ((min) << 12) | (pat))
#define NFRU_DP4A_VERSION NFRU_DP4A_MAKE_VERSION(1, 0, 0)

/* ------------------------------------------------------------------ results */
typedef enum NfruDp4aResult {
    NFRU_DP4A_OK = 0,
    NFRU_DP4A_ERROR_GENERIC = 1,
    NFRU_DP4A_ERROR_UNSUPPORTED = 2,      /* no DP4A or no shaderInt64/shader model 6.4 */
    NFRU_DP4A_ERROR_INVALID_ARGUMENT = 3,
    NFRU_DP4A_ERROR_OUT_OF_MEMORY = 4,
    NFRU_DP4A_ERROR_VULKAN_FAILED = 5,    /* graphics API call failed */
    NFRU_DP4A_ERROR_NOT_READY = 6,
    NFRU_DP4A_ERROR_MODEL_MISMATCH = 7,   /* resolution does not match the context */
    NFRU_DP4A_ERROR_IO_FAILED = 8
} NfruDp4aResult;

/* ------------------------------------------------------------------ device caps
 *
 * Public ABI: the host instantiates this on its stack. Never reorder or insert fields;
 * add new capabilities through a separate struct and query function. */
typedef struct NfruDp4aDeviceCaps {
    uint32_t integerDotProduct;            /* VK_KHR_shader_integer_dot_product */
    uint32_t dotProduct4x8BitPackedSigned; /* hardware DP4A available */
    uint32_t shaderInt64;                  /* required by the requantiser */
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxStorageBufferRange;
    char     deviceName[256];
} NfruDp4aDeviceCaps;

/* ------------------------------------------------------------------ create info */
typedef struct NfruDp4aCreateInfo {
    /* Vulkan: VkInstance / VkPhysicalDevice / VkDevice / VkQueue.
     * DX12:   instance carries the ID3D12Device and queue the ID3D12CommandQueue;
     *         physicalDevice may be 0. */
    uint64_t instance;
    uint64_t physicalDevice;
    uint64_t device;
    uint64_t queue;
    uint32_t queueFamilyIndex;
    uint32_t apiVersion;        /* VkPhysicalDeviceProperties::apiVersion */

    /* Optional pre-supplied upload command buffer. Reserved: this backend uploads
     * weights through host-visible memory and does not need one. */
    uint64_t uploadCommandBuffer;

    /* Working resolution. NFRU v1 is shape independent, so any size works; 270x480 is
     * the layout used for 1080p output. */
    uint32_t width;
    uint32_t height;

    /* Host Vulkan loader entry point (PFN_vkGetInstanceProcAddr). NULL means the
     * library loads vulkan-1.dll itself. Ignored by the DX12 backend. */
    void *vkGetInstanceProcAddr;

    void (*logCallback)(int level, const char *msg, void *userData);
    void *logUserData;
} NfruDp4aCreateInfo;

/* ------------------------------------------------------------------ buffers */
typedef struct NfruDp4aBuffer {
    uint64_t buffer;            /* VkBuffer or ID3D12Resource* */
    uint64_t offset;            /* byte offset, normally 0 */
    uint64_t size;              /* usable bytes */
} NfruDp4aBuffer;

/* ------------------------------------------------------------------ per-frame
 *
 * input  : int8 NHWC [H][W][16], stored four int8 per uint32 word, quantised as
 *          q = round(x * 255) - 128 (zero point -128, scale 1/255).
 * output : int8 NHWC [H][W][4]. These are PRE-SOFTMAX LOGITS:
 *              logits = 0.35356706380844116f * (code - 44)
 *          then softmax. (The published VGF's output zero point is 44 and its output
 *          scale is 0.35356706380844116.)
 */
typedef struct NfruDp4aDispatchInfo {
    NfruDp4aBuffer input;
    NfruDp4aBuffer output;

    uint32_t width;
    uint32_t height;

    /* Optional host-supplied scratch for the intermediate tensors. Pass {0,0,0} and the
     * library uses its own. Query the requirement with nfruDp4aGetScratchSize(). */
    NfruDp4aBuffer scratch;
} NfruDp4aDispatchInfo;

typedef struct NfruDp4aContext NfruDp4aContext;

/* ------------------------------------------------------------------ API */

NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aQueryDeviceCaps(uint64_t instance,
                        uint64_t physicalDevice,
                        uint32_t apiVersion,
                        void    *vkGetInstanceProcAddr,
                        NfruDp4aDeviceCaps *outCaps);

/* Creates pipelines, uploads the baked weights and allocates internal scratch. */
NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aCreateContext(const NfruDp4aCreateInfo *createInfo,
                      NfruDp4aContext **outContext);

NFRU_DP4A_API void NFRU_DP4A_CALL
nfruDp4aDestroyContext(NfruDp4aContext *context);

/* Records the whole graph (16 conv + 1 nearest-x2 resize + 2 concat = 19 dispatches)
 * into the host's command buffer, with the necessary barriers. Does not submit.
 *
 * Caller guarantees: the command buffer/list is recording; the input contents are ready
 * and visible; the output is not in use by another pass; and (DX12) both resources are
 * in a state a UAV write is legal from. */
NFRU_DP4A_API NfruDp4aResult NFRU_DP4A_CALL
nfruDp4aRecord(NfruDp4aContext *context,
               uint64_t        commandBuffer,
               const NfruDp4aDispatchInfo *dispatchInfo);

NFRU_DP4A_API uint64_t NFRU_DP4A_CALL nfruDp4aGetScratchSize(const NfruDp4aContext *context);
NFRU_DP4A_API uint64_t NFRU_DP4A_CALL nfruDp4aGetInternalMemoryUsage(const NfruDp4aContext *context);
NFRU_DP4A_API void NFRU_DP4A_CALL nfruDp4aFlushPendingUploads(NfruDp4aContext *context);

NFRU_DP4A_API uint32_t NFRU_DP4A_CALL nfruDp4aGetVersion(void);
NFRU_DP4A_API const char * NFRU_DP4A_CALL nfruDp4aGetResultString(NfruDp4aResult result);

/* Which backend this context selected: 0 = Vulkan, 1 = DX12.
 * Set NFRU_DP4A_BACKEND=dx12|vulkan before nfruDp4aCreateContext to force one, which is
 * how the two are cross-checked for identical output. */
NFRU_DP4A_API uint32_t NFRU_DP4A_CALL nfruDp4aGetBackend(NfruDp4aContext *context);
NFRU_DP4A_API const char * NFRU_DP4A_CALL nfruDp4aGetBackendName(NfruDp4aContext *context);

/* The most recent error text, for logging. Never null. */
NFRU_DP4A_API const char * NFRU_DP4A_CALL nfruDp4aGetLastError(NfruDp4aContext *context);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NFRU_DP4A_H */
