/*
 * nfru_graph.cpp -- the NFRU v1 execution graph, independent of the graphics API.
 *
 * Derived from `NFRUAutoEncoder.forward()`:
 *
 *     x = conv1(x); x = conv2(x); x = conv3(x)
 *     skip1 = skip1_conv(x); b = conv5(x)                 // conv5 is stride 2
 *     xa = conv5a(b); xb = conv5b(b)
 *     xc = conv5c_1(conv5c(b)); xd = conv5d_2(conv5d_1(conv5d(b)))
 *     x = conv5e(cat([xa, xb, xc, xd], dim=1))
 *     x = upsample_nearest_2x(x); x = conv6(x)
 *     x = conv7(cat([x, skip1], dim=1)); out = output_conv_mv(x)
 *
 * which is 16 convolutions, one nearest x2 resize and two channel concatenations: the
 * 19 dispatches the backend records. Tensor slots 0 and 1 are the host's input and
 * output buffers, so the graph needs no edge copies.
 */
#include "nfru_device.h"

namespace nfru {

/*
 * Dispatch list. `dst`, `src` and `srcs` index the tensor-slot table in
 * ComputeTensorLayout(). The convolution order matters only through these indices.
 */
const GraphOp kGraphOps[] = {
    /* kind        layer  dst  src  srcs                        count */
    { kOpConv,        0,   2,   0,  { -1, -1, -1, -1 },          0 },  /* conv1         -> 2  */
    { kOpConv,        1,   3,   2,  { -1, -1, -1, -1 },          0 },  /* conv2         -> 3  */
    { kOpConv,        2,   4,   3,  { -1, -1, -1, -1 },          0 },  /* conv3         -> 4  */
    { kOpConv,        3,   5,   4,  { -1, -1, -1, -1 },          0 },  /* skip1_conv    -> 5  */
    { kOpConv,        4,   6,   4,  { -1, -1, -1, -1 },          0 },  /* conv5 (stride2)-> 6 */
    { kOpConv,        5,   7,   6,  { -1, -1, -1, -1 },          0 },  /* conv5a        -> 7  */
    { kOpConv,        6,   8,   6,  { -1, -1, -1, -1 },          0 },  /* conv5b        -> 8  */
    { kOpConv,        7,   9,   6,  { -1, -1, -1, -1 },          0 },  /* conv5c        -> 9  */
    { kOpConv,        8,  10,   9,  { -1, -1, -1, -1 },          0 },  /* conv5c_1      -> 10 */
    { kOpConv,        9,  11,   6,  { -1, -1, -1, -1 },          0 },  /* conv5d        -> 11 */
    { kOpConv,       10,  12,  11,  { -1, -1, -1, -1 },          0 },  /* conv5d_1      -> 12 */
    { kOpConv,       11,  13,  12,  { -1, -1, -1, -1 },          0 },  /* conv5d_2      -> 13 */
    { kOpConcat,     -1,  14,  -1,  { 7, 8, 10, 13 },            4 },  /* cat xa xb xc xd -> 14 */
    { kOpConv,       12,  15,  14,  { -1, -1, -1, -1 },          0 },  /* conv5e        -> 15 */
    { kOpResize,     -1,   3,  15,  { -1, -1, -1, -1 },          0 },  /* upsample 2x   -> 3' */
    { kOpConv,       13,   4,   3,  { -1, -1, -1, -1 },          0 },  /* conv6         -> 4' */
    { kOpConcat,     -1,   6,  -1,  { 4, 5, -1, -1 },            2 },  /* cat(x, skip1) -> 6' */
    { kOpConv,       14,   7,   6,  { -1, -1, -1, -1 },          0 },  /* conv7         -> 7' */
    { kOpConv,       15,   1,   7,  { -1, -1, -1, -1 },          0 },  /* output_conv_mv-> out */
};
const uint32_t kGraphOpCount = sizeof(kGraphOps) / sizeof(kGraphOps[0]);

/* Names in kNfruLayers order. Used only by the trace output. */
const char* const kLayerNames[NFRU_MODEL_LAYER_COUNT] = {
    "conv1", "conv2", "conv3", "skip1_conv", "conv5",
    "conv5a", "conv5b", "conv5c", "conv5c_1", "conv5d", "conv5d_1", "conv5d_2",
    "conv5e", "conv6", "conv7", "output_conv_mv"
};

/*
 * Tensor shapes in execution order. The spatial size is derived at runtime: the stride-2
 * conv5 halves it and the resize doubles it.
 *
 * Slots are reused as scratch (slot 3 holds conv2's output early and the upsampled
 * tensor later), so a slot's authoritative shape is the one written by the LAST op
 * targeting it. Walking the ops in order and overwriting reproduces exactly that.
 */
bool ComputeTensorLayout(uint32_t width, uint32_t height, TensorLayout& out)
{
    int32_t ch[kTensorSlots];
    uint32_t h[kTensorSlots], w[kTensorSlots];
    for (int i = 0; i < kTensorSlots; ++i) {
        ch[i] = -1;
        h[i] = 0;
        w[i] = 0;
    }
    ch[kGraphInput] = NFRU_MODEL_INPUT_CHANNELS;
    h[kGraphInput] = height;
    w[kGraphInput] = width;

    for (uint32_t oi = 0; oi < kGraphOpCount; ++oi) {
        const GraphOp& op = kGraphOps[oi];
        if (op.kind == kOpConv) {
            const NfruBakedLayer& L = kNfruLayers[op.layer];
            const int32_t ih = (int32_t)h[op.src];
            const int32_t iw = (int32_t)w[op.src];
            const int32_t oh = (ih + (int32_t)L.padT + (int32_t)L.padB - (int32_t)L.kh)
                               / (int32_t)L.strideH + 1;
            const int32_t ow = (iw + (int32_t)L.padL + (int32_t)L.padR - (int32_t)L.kw)
                               / (int32_t)L.strideW + 1;
            if (oh <= 0 || ow <= 0) {
                return false;
            }
            h[op.dst] = (uint32_t)oh;
            w[op.dst] = (uint32_t)ow;
            ch[op.dst] = (int32_t)(L.outC4 * 4);
        } else if (op.kind == kOpResize) {
            h[op.dst] = h[op.src] * 2;
            w[op.dst] = w[op.src] * 2;
            ch[op.dst] = ch[op.src];
        } else {  /* kOpConcat */
            if (op.srcCount == 0 || op.srcCount > 4) {
                return false;
            }
            int32_t c = 0;
            const uint32_t hh = h[op.srcs[0]];
            const uint32_t ww = w[op.srcs[0]];
            for (uint32_t s = 0; s < op.srcCount; ++s) {
                if (ch[op.srcs[s]] < 0 || h[op.srcs[s]] != hh || w[op.srcs[s]] != ww) {
                    return false;
                }
                c += ch[op.srcs[s]];
            }
            ch[op.dst] = c;
            h[op.dst] = hh;
            w[op.dst] = ww;
        }
    }

    /* Pack with 256-byte alignment. The kernel only needs 4, but a round offset makes
     * the layout readable in a debugger and keeps the two backends trivially identical. */
    uint64_t offset = 0;
    for (int i = 0; i < kTensorSlots; ++i) {
        TensorRef& t = out.slot[i];
        if (ch[i] < 0) {
            t.offset = 0;
            t.size = 0;
            t.w = t.h = 0;
            t.c4 = 0;
            continue;
        }
        offset = (offset + 255) & ~(uint64_t)255;
        t.offset = offset;
        t.size = (uint64_t)h[i] * w[i] * (uint32_t)ch[i];
        t.w = w[i];
        t.h = h[i];
        t.c4 = (uint32_t)ch[i] / 4;
        offset += t.size;
    }
    out.scratchBytes = (offset + 255) & ~(uint64_t)255;
    return true;
}

}  /* namespace nfru */
