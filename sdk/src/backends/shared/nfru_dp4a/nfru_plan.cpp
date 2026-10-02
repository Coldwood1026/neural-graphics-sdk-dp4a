/*
 * nfru_plan.cpp -- the API-agnostic record plan for the NFRU v1 dp4a graph.
 *
 * This is the single place that decides what gets recorded: push constants, dispatch
 * extents, tensor offsets and the order of passes. Both `nfru_vk.cpp` and `nfru_dx12.cpp`
 * only translate the resulting list into API calls, which is what makes "Vulkan and DX12
 * are bit-identical" a property of the code rather than a claim about two files that were
 * edited in parallel.
 *
 * The kernels themselves are `nfru_conv_rq.comp` / `.hlsl` and `nfru_support.comp` /
 * `.hlsl`; their semantics (int8 NHWC, four channels per uint32, dp4a accumulation,
 * TOSA SINGLE_ROUND requantisation, the input zero point folded into the bias, and ReLU
 * implied by an output zero point of -128) are documented in the convolution source.
 *
 * SPDX-License-Identifier: MIT
 */
#include "nfru_device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace nfru {

int DumpMode()
{
    if (getenv("NFRU_DP4A_DUMP_MARK")) return 3;
    if (getenv("NFRU_DP4A_DUMP_ACC")) return 2;
    return 0;
}

namespace {

/* Appends one dispatch, failing the whole plan if the fixed array is too small: silently
 * dropping a dispatch is indistinguishable from a correct but incomplete graph. */
bool Push(RecordPlan& plan, const PlannedDispatch& d)
{
    if (plan.count >= RecordPlan::kMaxDispatches) return false;
    plan.d[plan.count++] = d;
    return true;
}

uint32_t DivUp(uint64_t n, uint32_t by) { return (uint32_t)((n + by - 1) / by); }

}  /* namespace */

bool BuildRecordPlan(uint32_t width, uint32_t height, const TensorLayout& layout,
                     uint32_t w, uint32_t h, uint64_t inputOff, uint64_t outputOff,
                     RecordPlan& out, char* err, size_t errLen)
{
    memset(&out, 0, sizeof(out));

    /* The shape of each slot AS OF the dispatch being recorded. See nfru_device.h. */
    TensorRef cur[kTensorSlots];
    memset(cur, 0, sizeof(cur));
    cur[kGraphInput].offset = 0;
    cur[kGraphInput].size = (uint64_t)w * h * 16u;
    cur[kGraphInput].w = w;
    cur[kGraphInput].h = h;
    cur[kGraphInput].c4 = 4;

    /*
     * NFRU_DP4A_SUPPORT_SELFTEST=<0|1|2> runs ONE support kernel against the host input
     * buffer instead of the graph. 0 = copy, 1 = nearest x2 resize, 2 = concat of the
     * input with itself. It exercises the support kernels' index arithmetic in isolation,
     * which the end-to-end graph cannot do because it never exposes an intermediate.
     */
    if (const char* selfEnv = getenv("NFRU_DP4A_SUPPORT_SELFTEST")) {
        const int mode = atoi(selfEnv);
        const uint32_t inW = w, inH = h;      /* host tensor: [1][h][w][16] */
        const uint32_t inC4 = 4;
        const uint32_t inWords = inW * inH * inC4;
        const uint32_t outW = (mode == 1) ? inW * 2 : inW;
        const uint32_t outH = (mode == 1) ? inH * 2 : inH;
        const uint32_t outC4 = (mode == 2) ? inC4 * 2 : inC4;

        PlannedDispatch d = {};
        d.isConv = false;
        d.srcIsHost = true;
        d.dstIsHost = true;
        d.srcOffBytes = 0;
        d.dstOffBytes = outputOff;
        uint32_t* pc = d.spc;
        uint64_t count = 0;
        if (mode == 0) {
            pc[kPcOp] = kSupportCopy;
            pc[kPcSrcOff] = 0;
            pc[kPcOutOff] = (uint32_t)(outputOff / 4);
            count = inWords;
        } else if (mode == 1) {
            pc[kPcOp] = kSupportResize;
            pc[kPcSrcOff] = 0;
            pc[kPcSrcW] = inW;
            pc[kPcSrcH] = inH;
            pc[kPcSrcC4] = inC4;
            pc[kPcOutOff] = (uint32_t)(outputOff / 4);
            pc[kPcDstW] = outW;
            pc[kPcDstH] = outH;
            pc[kPcDstC4] = outC4;
            count = (uint64_t)outW * outH * outC4;
        } else {
            pc[kPcOp] = kSupportConcat;
            pc[kPcInOff] = 0;
            pc[kPcPlaneRows] = inH;
            pc[kPcPlaneWords] = inW * inC4;
            pc[kPcSrcC4Mul] = inC4;
            pc[kPcOutOff] = (uint32_t)(outputOff / 4);
            pc[kPcDstW] = outW;
            pc[kPcDstH] = outH;
            pc[kPcDstC4] = outC4;
            count = 2ull * inWords;
        }
        pc[kPcCount] = (uint32_t)count;
        d.groups = DivUp(count, 256);
        if (getenv("NFRU_DP4A_TRACE_SUPPORT")) {
            printf("[support] selftest mode=%d count=%llu groups=%llu | op=%u dstW=%u dstH=%u "
                   "dstC4=%u srcW=%u srcH=%u srcC4=%u srcOff=%u outOff=%u planeRows=%u "
                   "planeWords=%u mul=%u inOff=%u\n",
                   mode, (unsigned long long)count, (unsigned long long)d.groups,
                   pc[kPcOp], pc[kPcDstW], pc[kPcDstH], pc[kPcDstC4],
                   pc[kPcSrcW], pc[kPcSrcH], pc[kPcSrcC4], pc[kPcSrcOff], pc[kPcOutOff],
                   pc[kPcPlaneRows], pc[kPcPlaneWords], pc[kPcSrcC4Mul], pc[kPcInOff]);
            fflush(stdout);
        }
        return Push(out, d);
    }

    /*
     * NFRU_DP4A_LAYER_ONLY=<conv-op-index> runs exactly ONE convolution with the host input
     * buffer as its activation input and the host output buffer as its result, sized from
     * that layer's own geometry. The layer's geometry is derived from its kernel, stride
     * and padding, so the graph's inter-layer layout is never consulted and the context
     * does not have to be created at a resolution at which the whole graph closes.
     */
    if (const char* layerOnlyEnv = getenv("NFRU_DP4A_LAYER_ONLY")) {
        const int opIndex = atoi(layerOnlyEnv);
        if (opIndex < 0 || opIndex >= (int)kGraphOpCount ||
            kGraphOps[opIndex].kind != kOpConv) {
            snprintf(err, errLen, "NFRU_DP4A_LAYER_ONLY=%d is not a conv op", opIndex);
            return false;
        }
        const GraphOp& op = kGraphOps[opIndex];
        const NfruBakedLayer& L = kNfruLayers[op.layer];
        const uint32_t inH = h, inW = w;
        const int32_t oH = ((int32_t)inH + (int32_t)L.padT + (int32_t)L.padB
                            - (int32_t)L.kh) / (int32_t)L.strideH + 1;
        const int32_t oW = ((int32_t)inW + (int32_t)L.padL + (int32_t)L.padR
                            - (int32_t)L.kw) / (int32_t)L.strideW + 1;
        if (oH <= 0 || oW <= 0) {
            snprintf(err, errLen, "layer-only %dx%d yields no output", oW, oH);
            return false;
        }

        PlannedDispatch d = {};
        d.isConv = true;
        d.layer = &L;
        d.pc.in_w = inW;
        d.pc.in_c4 = L.inC4;
        d.pc.out_h = (uint32_t)oH;
        d.pc.out_w = (uint32_t)oW;
        d.pc.out_c4 = L.outC4;
        d.pc.pad_top = (int32_t)L.padT;
        d.pc.pad_left = (int32_t)L.padL;
        d.pc.stride_h = L.strideH;
        d.pc.stride_w = L.strideW;
        d.pc.out_zp = L.outZeroPoint;
        d.pc.kh = L.kh;
        d.pc.kw = L.kw;
        d.pc.has_lut = DumpMode();
        d.pc.in_h = inH;
        d.gx = DivUp((uint32_t)oW, 8);
        d.gy = DivUp((uint32_t)oH, 8);
        d.gz = (L.outC4 + 3) / 4;
        d.wOffBytes = (uint64_t)L.wOff * 4u;
        d.bcOffBytes = (uint64_t)L.bcOff * 4u;
        d.mOffBytes = (uint64_t)L.mOff * 4u;
        d.sOffBytes = (uint64_t)L.sOff * 4u;
        d.srcIsHost = true;
        d.dstIsHost = true;
        d.srcOffBytes = inputOff;
        d.dstOffBytes = outputOff;
        printf("[layer-only] op%d %s : in %ux%u -> out %ux%u c4=%u (bytes %llu)\n",
               opIndex, kLayerNames[op.layer], inW, inH, (uint32_t)oW, (uint32_t)oH,
               L.outC4, (unsigned long long)oW * oH * L.outC4 * 4);
        fflush(stdout);
        return Push(out, d);
    }

    /*
     * NFRU_DP4A_STOP_AFTER=<n> records only the first n dispatches. The last one is
     * retargeted to the host output buffer, which makes a single intermediate comparable
     * against the CPU reference; with NFRU_DP4A_STOP_TO_SCRATCH=1 it instead keeps writing
     * into scratch and the tensor is copied out afterwards, exactly as the graph leaves
     * it.
     */
    const char* stopEnv = getenv("NFRU_DP4A_STOP_AFTER");
    const int stopAfter = stopEnv ? atoi(stopEnv) : 0;
    const bool stopToScratch = getenv("NFRU_DP4A_STOP_TO_SCRATCH") != nullptr;
    /*
     * NFRU_DP4A_INPUT_FOR_OP=<n> makes dispatch n read the host input buffer instead of its
     * real producer's tensor, so a crafted impulse input maps the kernel's (iy, ix)
     * sampling straight onto the output.
     */
    const char* inputForOpEnv = getenv("NFRU_DP4A_INPUT_FOR_OP");
    const int inputForOp = inputForOpEnv ? atoi(inputForOpEnv) : -1;
    int32_t lastStopDst = -1;

    for (uint32_t oi = 0; oi < kGraphOpCount; ++oi) {
        if (stopAfter > 0 && (int)oi >= stopAfter) break;
        const bool lastInStop = (stopAfter > 0 && (int)oi == stopAfter - 1);
        if (lastInStop) lastStopDst = kGraphOps[oi].dst;

        const GraphOp& op = kGraphOps[oi];
        const bool forceHostIn = (inputForOp >= 0 && (int)oi == inputForOp);
        const int32_t dst = (lastInStop && !stopToScratch) ? kGraphOutput : op.dst;
        const bool srcIsHost =
            forceHostIn || ((op.kind == kOpConcat) ? false : (op.src == kGraphInput));
        const bool dstIsHost = (dst == kGraphOutput);

        if (op.kind == kOpConv) {
            const NfruBakedLayer& L = kNfruLayers[op.layer];
            const TensorRef& st = cur[op.src];
            const int32_t outH = ((int32_t)st.h + (int32_t)L.padT + (int32_t)L.padB
                                  - (int32_t)L.kh) / (int32_t)L.strideH + 1;
            const int32_t outW = ((int32_t)st.w + (int32_t)L.padL + (int32_t)L.padR
                                  - (int32_t)L.kw) / (int32_t)L.strideW + 1;
            if (outH <= 0 || outW <= 0) {
                snprintf(err, errLen, "op%u %s produced no output from %ux%u", oi,
                         kLayerNames[op.layer], st.w, st.h);
                return false;
            }

            PlannedDispatch d = {};
            d.isConv = true;
            d.layer = &L;
            d.pc.in_w = st.w;
            d.pc.in_c4 = L.inC4;
            d.pc.out_h = (uint32_t)outH;
            d.pc.out_w = (uint32_t)outW;
            d.pc.out_c4 = L.outC4;
            d.pc.pad_top = (int32_t)L.padT;
            d.pc.pad_left = (int32_t)L.padL;
            d.pc.stride_h = L.strideH;
            d.pc.stride_w = L.strideW;
            d.pc.out_zp = L.outZeroPoint;
            d.pc.kh = L.kh;
            d.pc.kw = L.kw;
            d.pc.has_lut = DumpMode();
            d.pc.in_h = st.h;
            /*
             * The dispatch grid covers the layer's TRUE output, not the destination slot's
             * recorded extent. Slots are reused and the layout is last-writer-wins, so the
             * slot can be smaller than the tensor being written -- and then part of the
             * output would never be dispatched at all.
             */
            d.gx = DivUp((uint32_t)outW, 8);
            d.gy = DivUp((uint32_t)outH, 8);
            d.gz = (L.outC4 + 3) / 4;
            d.wOffBytes = (uint64_t)L.wOff * 4u;
            d.bcOffBytes = (uint64_t)L.bcOff * 4u;
            d.mOffBytes = (uint64_t)L.mOff * 4u;
            d.sOffBytes = (uint64_t)L.sOff * 4u;
            d.srcIsHost = srcIsHost;
            d.dstIsHost = dstIsHost;
            d.srcOffBytes = srcIsHost ? inputOff : layout.slot[op.src].offset;
            d.dstOffBytes = dstIsHost ? outputOff : layout.slot[dst].offset;

            if (getenv("NFRU_DP4A_TRACE_PC")) {
                const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&d.pc);
                printf("[pc] op%-2u %-15s :", oi, kLayerNames[op.layer]);
                for (int i = 0; i < 14; ++i) printf(" %u", w32[i]);
                printf("\n[pc]   as fields: in_w=%u in_c4=%u out_h=%u out_w=%u out_c4=%u "
                       "padT=%d padL=%d sh=%u sw=%u outzp=%d kh=%u kw=%u in_h=%u\n",
                       d.pc.in_w, d.pc.in_c4, d.pc.out_h, d.pc.out_w, d.pc.out_c4,
                       d.pc.pad_top, d.pc.pad_left, d.pc.stride_h, d.pc.stride_w,
                       d.pc.out_zp, d.pc.kh, d.pc.kw, d.pc.in_h);
                fflush(stdout);
            }
            if (getenv("NFRU_DP4A_TRACE_GEOM")) {
                printf("[geom] op%-2u %-15s in %ux%u c4=%u h=%u | out %ux%u c4=%u "
                       "| pad %d,%d s %u,%u k %ux%u | so=%llu do=%llu "
                       "| wOff=%u wWords=%u bcOff=%u M=%d S=%d ozp=%d\n",
                       oi, kLayerNames[op.layer], st.w, st.h, d.pc.in_c4, d.pc.in_h,
                       (uint32_t)outW, (uint32_t)outH, d.pc.out_c4, d.pc.pad_top,
                       d.pc.pad_left, d.pc.stride_h, d.pc.stride_w, d.pc.kh, d.pc.kw,
                       (unsigned long long)d.srcOffBytes, (unsigned long long)d.dstOffBytes,
                       L.wOff, L.wWords, L.bcOff, (int)kNfruMult[L.mOff],
                       (int)kNfruShift[L.sOff], (int)L.outZeroPoint);
                fflush(stdout);
            }
            if (!Push(out, d)) {
                snprintf(err, errLen, "more than %d dispatches", (int)RecordPlan::kMaxDispatches);
                return false;
            }
            /* What this op actually produced, for the ops that consume it. */
            cur[dst].w = (uint32_t)outW;
            cur[dst].h = (uint32_t)outH;
            cur[dst].c4 = L.outC4;
            cur[dst].size = (uint64_t)outW * (uint64_t)outH * (uint64_t)L.outC4 * 4u;
            continue;
        }

        /*
         * Support kernels. Offsets come from `layout` (a slot's offset never depends on
         * which op writes it); shapes come from `cur`. A resize's destination shape is
         * derived from its source rather than looked up, so nothing here depends on
         * last-writer-wins state.
         */
        PlannedDispatch d = {};
        d.isConv = false;
        d.srcIsHost = (op.kind == kOpResize) && (op.src == kGraphInput);
        d.dstIsHost = dstIsHost;
        /* Support kernels address their source through the PUSH CONSTANTS (kPcSrcOff /
         * kPcInOff) and their destination through kPcOutOff, all of which are absolute
         * word offsets inside whatever buffer is bound. So the descriptors themselves are
         * bound at the base of each buffer -- offsetting them here as well as in the push
         * constant would count the tensor's position twice. The one exception is the host
         * input buffer, whose own byte offset the descriptor carries because the push
         * constant for it is zeroed. */
        d.srcOffBytes = d.srcIsHost ? inputOff : 0;
        d.dstOffBytes = 0;

        /* A resize is a single pass; a concatenation is one pass per source, because each
         * source has its own offset, pitch and channel count in the scratch buffer. */
        uint32_t passes = 1;
        if (op.kind == kOpResize) {
            const TensorRef& st = cur[op.src];
            d.spc[kPcOp] = kSupportResize;
            d.spc[kPcOutOff] = (uint32_t)(layout.slot[op.dst].offset / 4);
            d.spc[kPcDstW] = st.w * 2u;
            d.spc[kPcDstH] = st.h * 2u;
            d.spc[kPcDstC4] = st.c4;
            d.spc[kPcSrcOff] = d.srcIsHost ? 0
                                           : (uint32_t)(layout.slot[op.src].offset / 4);
            d.spc[kPcSrcW] = st.w;
            d.spc[kPcSrcH] = st.h;
            d.spc[kPcSrcC4] = st.c4;
            d.spc[kPcCount] = (uint32_t)((uint64_t)d.spc[kPcDstW] * d.spc[kPcDstH] *
                                         d.spc[kPcDstC4]);
            d.groups = DivUp(d.spc[kPcCount], 256);
            cur[op.dst].w = d.spc[kPcDstW];
            cur[op.dst].h = d.spc[kPcDstH];
            cur[op.dst].c4 = d.spc[kPcDstC4];
            cur[op.dst].size = (uint64_t)d.spc[kPcCount] * 4u;
            if (d.dstIsHost) d.spc[kPcOutOff] = (uint32_t)(outputOff / 4);
        } else {
            const TensorRef& s0 = cur[op.srcs[0]];
            uint32_t totalC4 = 0;
            for (uint32_t k = 0; k < op.srcCount; ++k) totalC4 += cur[op.srcs[k]].c4;
            d.spc[kPcOp] = kSupportConcat;
            d.spc[kPcDstW] = s0.w;
            d.spc[kPcDstH] = s0.h;
            d.spc[kPcDstC4] = totalC4;
            d.spc[kPcSrcC4Mul] = s0.c4;
            /* The concatenation's own shape: the first source's spatial extent with the
             * channel word counts summed. */
            cur[op.dst].w = s0.w;
            cur[op.dst].h = s0.h;
            cur[op.dst].c4 = totalC4;
            cur[op.dst].size = (uint64_t)s0.w * s0.h * totalC4 * 4u;
            passes = op.srcCount;
        }

        if (getenv("NFRU_DP4A_TRACE_SUPPORT")) {
            printf("[sup] op%-2u %-6s | op=%u dst %ux%u c4=%u outOff=%u passes=%u\n",
                   oi, (op.kind == kOpResize) ? "resize" : "concat", d.spc[kPcOp],
                   d.spc[kPcDstW], d.spc[kPcDstH], d.spc[kPcDstC4], d.spc[kPcOutOff], passes);
            fflush(stdout);
        }

        for (uint32_t k = 0; k < passes; ++k) {
            PlannedDispatch pass = d;
            if (op.kind == kOpConcat) {
                const TensorRef& sk = cur[op.srcs[k]];
                uint32_t channelBase = 0;
                for (uint32_t j = 0; j < k; ++j) channelBase += cur[op.srcs[j]].c4;
                const bool srcHost = (op.srcs[k] == kGraphInput);
                pass.srcIsHost = srcHost;
                pass.srcOffBytes = srcHost ? inputOff : 0;
                pass.spc[kPcInOff] = srcHost
                                         ? 0
                                         : (uint32_t)(layout.slot[op.srcs[k]].offset / 4);
                pass.spc[kPcPlaneRows] = sk.h;
                pass.spc[kPcPlaneWords] = sk.w * sk.c4;
                pass.spc[kPcSrcC4Mul] = sk.c4;
                pass.spc[kPcCount] = (uint32_t)((uint64_t)sk.w * sk.h * sk.c4);
                const uint32_t outOffWords = dstIsHost ? (uint32_t)(outputOff / 4)
                                                       : (uint32_t)(layout.slot[op.dst].offset / 4);
                pass.spc[kPcOutOff] = outOffWords + channelBase;
                pass.groups = DivUp(pass.spc[kPcCount], 256);
            }
            if (!Push(out, pass)) {
                snprintf(err, errLen, "more than %d dispatches", (int)RecordPlan::kMaxDispatches);
                return false;
            }
        }
    }

    /*
     * Copy-out for STOP_TO_SCRATCH. The byte count is what the last recorded op actually
     * wrote, from `cur`, not `layout.slot[].size`: slots are reused and the layout holds
     * the shape of whatever writes the slot LAST, which is frequently a different op.
     * Copying the slot's final size reads past the tensor and pulls in neighbouring
     * scratch, making the comparison fail for reasons unrelated to the kernels.
     */
    if (lastStopDst >= 0 && stopToScratch && lastStopDst != kGraphOutput) {
        out.copyBytes = cur[lastStopDst].size;
        out.copySrcOff = layout.slot[lastStopDst].offset;
        out.copyDstOff = outputOff;
        printf("[stop] op%d -> slot %d (%ux%u c4=%u), copying %llu bytes from offset %llu\n",
               stopAfter - 1, lastStopDst, cur[lastStopDst].w, cur[lastStopDst].h,
               cur[lastStopDst].c4, (unsigned long long)out.copyBytes,
               (unsigned long long)out.copySrcOff);
        fflush(stdout);
    }
    return true;
}

}  /* namespace nfru */
