/*
 * nfru_device.h -- graphics-API abstraction shared by the Vulkan and DX12 dp4a backends.
 *
 * The graph, the push-constant layout and the dispatch sequence are identical on both
 * APIs; only buffer creation, descriptor binding and command recording differ. This
 * interface captures exactly that difference, so the graph is written once.
 *
 * Terminology: a "tensor" here is a plain int8 buffer in NHWC order with four channels
 * per uint32 word, which is the layout the dp4a kernel consumes. No numeric conversion
 * happens in this layer -- the data is integer end to end.
 */
#ifndef NFRU_DEVICE_H
#define NFRU_DEVICE_H

#include <stdint.h>
#include "nfru_dp4a.h"
#include "nfru_model_baked.h"

namespace nfru {

/*
 * Tensor slots. Slot 0 is the host's input buffer and slot 1 its output buffer, so the
 * graph starts and ends on host memory and needs no edge copies. Slots 2..15 are
 * intermediates carved out of the scratch buffer; a slot is reused when its previous
 * value is dead, which the graph definition encodes by listing destination slots.
 */
enum {
    kGraphInput = 0,
    kGraphOutput = 1,
    kTensorSlots = 16
};

/* The NFRU v1 graph in execution order: 16 convolutions, one nearest x2 resize and two
 * channel concatenations. */
enum GraphOpKind { kOpConv = 0, kOpResize = 1, kOpConcat = 2 };

struct GraphOp {
    uint32_t kind;            /* GraphOpKind */
    int32_t  layer;           /* kNfruLayers index, for kOpConv */
    int32_t  dst;             /* destination tensor slot */
    int32_t  src;             /* source slot, for kOpConv/kOpResize (else -1) */
    int32_t  srcs[4];         /* source slots, kOpConcat only; unused entries are -1 */
    uint32_t srcCount;
};

extern const GraphOp kGraphOps[];
extern const uint32_t kGraphOpCount;

/* Layer names in kNfruLayers order, for trace output only. */
extern const char* const kLayerNames[NFRU_MODEL_LAYER_COUNT];

/* Push constants for the convolution kernel, matching `fru_conv_rq.comp` exactly. */
struct ConvPushConstants {
    uint32_t in_w;
    uint32_t in_c4;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t out_c4;
    int32_t  pad_top;
    int32_t  pad_left;
    uint32_t stride_h;
    uint32_t stride_w;
    int32_t  out_zp;
    uint32_t kh;
    uint32_t kw;
    int32_t  has_lut;
    uint32_t in_h;
};
static_assert(sizeof(ConvPushConstants) == 14 * 4, "push constant layout changed");

/* Push constants for the support kernels (copy / resize2x / concat), also 16 words so
 * that the DX12 32-bit root constants and the Vulkan block agree word for word. */
enum SupportPc {
    kPcOp = 0,
    kPcCount = 1,
    kPcOutOff = 2,
    kPcDstW = 3,
    kPcDstH = 4,
    kPcDstC4 = 5,
    kPcSrcOff = 6,
    kPcSrcW = 7,
    kPcSrcH = 8,
    kPcSrcC4 = 9,
    kPcPlaneRows = 10,
    kPcPlaneWords = 11,
    kPcSrcC4Mul = 12,
    kPcInOff = 13,
    kPcWords = 16
};
enum SupportOp { kSupportCopy = 0, kSupportResize = 1, kSupportConcat = 2 };

/* Bindings used by the convolution dispatch. */
enum ConvBinding {
    kBindActIn = 0,
    kBindWeights = 1,
    kBindBias = 2,
    kBindMult = 3,
    kBindShift = 4,
    kBindActOut = 5,
    kBindCount = 6
};

/* Support kernels use two bindings so that the DX12 root signature stays simple. */
enum SupportBinding { kBindSupportSrc = 0, kBindSupportDst = 1, kBindSupportCount = 2 };

/* One tensor slot's location and shape inside the scratch buffer. */
struct TensorRef {
    uint64_t offset;
    uint64_t size;
    uint32_t w;
    uint32_t h;
    uint32_t c4;      /* uint32 words per pixel */
};

struct TensorLayout {
    TensorRef slot[kTensorSlots];
    uint64_t scratchBytes;
};

/*
 * Resolves every slot's offset and shape at a given resolution. Returns false when the
 * graph is inconsistent (for instance a concatenation whose inputs disagree in spatial
 * extent); that cannot happen at these resolutions but is checked, because a silent
 * mismatch would produce wrong numbers rather than a visible failure.
 */
bool ComputeTensorLayout(uint32_t width, uint32_t height, TensorLayout& out);

/* -----------------------------------------------------------------------------
 * Record planning
 * -----------------------------------------------------------------------------
 *
 * Everything that decides WHAT work is recorded -- push constants, dispatch extents,
 * tensor offsets, the order of dispatch passes -- is built here, once, and both backends
 * only turn it into API calls. The two backends must produce bit-identical output, and
 * geometry is where every fault in this port has lived; duplicating that logic per API
 * means fixing each bug twice and hoping the copies stay in step. This does not.
 *
 * A note on why the tensor shapes are tracked as a separate array rather than read from
 * `TensorLayout`: `ComputeTensorLayout` is last-writer-wins over the WHOLE graph, so
 * `layout.slot[6]` describes the 480x270x32 concatenation written at op 16 even when the
 * op being recorded is conv5a at op 5, whose input is conv5's 240x135x16 output. Using
 * the layout for shapes gives every reused slot the wrong extent, wrong row pitch and
 * wrong channel count. Offsets are stable and still come from the layout.
 */

struct PlannedDispatch {
    bool isConv;

    /* Convolution dispatches. */
    ConvPushConstants pc;
    const NfruBakedLayer* layer;
    uint32_t gx, gy, gz;
    uint64_t wOffBytes, bcOffBytes, mOffBytes, sOffBytes;

    /* Support dispatches. */
    uint32_t spc[kPcWords];
    uint32_t groups;

    /* Where the source and destination live. `srcIsHost` / `dstIsHost` select the host's
     * input and output buffers over the backend's scratch buffer. */
    bool srcIsHost, dstIsHost;
    uint64_t srcOffBytes, dstOffBytes;
};

struct RecordPlan {
    enum { kMaxDispatches = 48 };
    PlannedDispatch d[kMaxDispatches];
    uint32_t count;

    /* Optional copy-out used by NFRU_DP4A_STOP_TO_SCRATCH. `copyBytes == 0` means none.
     * Expressed as a scratch -> host-output copy over whole buffers. */
    uint64_t copySrcOff, copyDstOff, copyBytes;
};

/*
 * Builds the plan for one command buffer. `width`/`height` are the context resolution and
 * `w`/`h` the requested one; `inputOff`/`outputOff` are the host's byte offsets into its
 * input and output buffers. Returns false with `err` filled in for an invalid request.
 */
bool BuildRecordPlan(uint32_t width, uint32_t height, const TensorLayout& layout,
                     uint32_t w, uint32_t h, uint64_t inputOff, uint64_t outputOff,
                     RecordPlan& out, char* err, size_t errLen);

/* Selects the kernel's debug output path:
 *   0 -- normal packed int8 codes
 *   2 -- raw int32 accumulator per output channel (NFRU_DP4A_DUMP_ACC)
 *   3 -- per-thread marker instead of the accumulator (NFRU_DP4A_DUMP_MARK)
 * A zero word is otherwise ambiguous between "this thread never ran" and "the thread ran
 * and its store was lost", which have completely different causes. */
int DumpMode();

/*
 * Backend interface, implemented by nfru_vk.cpp and nfru_dx12.cpp.
 *
 * A backend owns the compute pipelines, the weight/bias/mult/shift buffers and (when the
 * host does not supply one) the scratch buffer. It never owns the host's input/output
 * buffers and never submits work: the host decides when the command buffer runs, so this
 * can share a batch with the surrounding passes.
 */
class Device {
public:
    virtual ~Device() {}

    virtual bool valid() const = 0;
    virtual const char* backendName() const = 0;
    virtual uint64_t scratchBytes() const = 0;
    virtual uint64_t internalBytes() const = 0;
    virtual const char* lastError() const = 0;

    /*
     * Records the whole graph into the host's command buffer. `scratchBuffer` of 0 means
     * "use the backend's own". All handles are raw API objects cast to uint64_t so this
     * header stays API-agnostic.
     *
     * `inputBytes` / `outputBytes` are the sizes of the host's buffers. Vulkan does not
     * need them (descriptors there can span the whole buffer), but D3D12 views must state
     * an element count, and guessing it from the tensor shape breaks the dump modes, whose
     * output is four times the usual size.
     */
    virtual NfruDp4aResult record(uint64_t commandBuffer,
                                  uint64_t inputBuffer, uint64_t inputOff,
                                  uint64_t inputBytes,
                                  uint64_t outputBuffer, uint64_t outputOff,
                                  uint64_t outputBytes,
                                  uint64_t scratchBuffer,
                                  uint32_t width, uint32_t height) = 0;
};

}  /* namespace nfru */

#endif /* NFRU_DEVICE_H */
