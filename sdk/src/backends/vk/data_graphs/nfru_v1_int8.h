/*
 * nfru_v1_int8.h -- NFRU v1 int8 data-graph model descriptor.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * THIS FILE REPLACES THE HEADER THAT `sdk/tools/binary_store/Model_Parser.exe`
 * USED TO GENERATE FROM `data_graphs/nfru/nfru_v1_int8.vgf`.
 * =============================================================================
 *
 * Why it is hand-written now
 * --------------------------
 * The old pipeline was: `compile_data_graphs()` runs Model_Parser on the `.vgf`,
 * which emits this header as a serialised `FfxDataGraphBlob` -- the raw VGF bytes
 * plus the tensor tables extracted from it. That blob had exactly one consumer,
 * `CreateDataGraphPipelineVK`, which fed it to
 * vkCreateDataGraphPipelinesARM.
 *
 * That execution path is gone, replaced by a portable dp4a compute backend
 * (`sdk/src/backends/vk/nfru_dp4a*`) which carries its own baked int8 weights and
 * therefore never reads a VGF. Removing the ARM path therefore also removes the
 * only reason to parse one, and with it the Model_Parser build dependency --
 * which is convenient, because Model_Parser crashes with 0xC0000409 on the
 * repository's own `.vgf` files.
 *
 * Everything the SDK still needs from the blob is *identity and wiring*: which
 * model this is, and which tensors it consumes and produces. Those are stated
 * below from the authoritative sources rather than transcribed from a tool:
 *
 *   * counts: the `.vgf` binaries declare exactly `graph_partition_0_arg_0` and
 *     `graph_partition_0_res_0` for NFRU (one argument, one result);
 *   * names: `ffx_frameinterpolation.cpp` carries the data-graph names as
 *     explicit duplicates in its resource-binding table, commented
 *     "data graph duplicates":
 *
 *         {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_IN_TENSOR,         "Resource_0_input"}
 *         {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUT_PARAMS_TENSOR, "Resource_1_output"}
 *
 *     `patchResourceIdentifier()` joins pipeline tensor bindings to resource
 *     identifiers BY NAME, so these strings are load-bearing: if they do not match
 *     the effect's table, binding resolution fails.
 *
 * `Resource_<n>_input` / `Resource_<n>_output` is a numbering scheme, not a VGF
 * name: the VGF calls its nodes `graph_partition_0_arg_0` / `_res_0`. The index is
 * 1-based over all tensors of the graph, arguments first -- which is why NSS's two
 * results are `Resource_1_output` and `Resource_2_output` and not `_0_`/`_1_`.
 */
#ifndef NFRU_V1_INT8_H
#define NFRU_V1_INT8_H

#include <FidelityFX/host/ffx_types.h>

/*
 * Tensor names, in the order the effect code fills them into
 * FfxDataGraphJobDescription::srvTensors / ::uavTensors. The order defines the
 * index the backend sees, so it must not be permuted casually.
 */
static const char* g_nfru_v1_int8_InputTensorNames[] = {
    "Resource_0_input",
};
static const uint32_t g_nfru_v1_int8_InputTensorSets[]     = { 0 };
static const uint32_t g_nfru_v1_int8_InputTensorBindings[] = { 0 };

static const char* g_nfru_v1_int8_OutputTensorNames[] = {
    "Resource_1_output",
};
static const uint32_t g_nfru_v1_int8_OutputTensorSets[]     = { 0 };
static const uint32_t g_nfru_v1_int8_OutputTensorBindings[] = { 0 };

/*
 * Model identity. `graphEntryPoint` is what the dp4a backend switches on to pick
 * its baked weight set; it is the one field here that carries real information.
 *
 * `graphData` is deliberately null and `constantNums` zero: there is no ARM graph
 * to upload and no pipeline constants to marshal. The fields that used to describe
 * tensor formats and dimensions are left empty for the same reason -- the only
 * code that read them built VkTensorDescriptionARM structures, and that code is
 * gone. A backend that needs a shape derives it from the dispatch resolution.
 */
#define NFRU_V1_INT8_ENTRY_POINT "nfru_v1_int8"

static const FfxDataGraphBlob g_nfru_v1_int8_Info = {
    /* constantNums        */ 0,
    /* constantIds         */ nullptr,
    /* constantFormats     */ nullptr,
    /* constantShapeSize   */ nullptr,
    /* constantShapes      */ nullptr,
    /* constantSparsityDimensions */ nullptr,
    /* constantDataSize    */ nullptr,
    /* constantDatas       */ nullptr,

    /* graphEntryPoint     */ NFRU_V1_INT8_ENTRY_POINT,
    /* graphDataSize       */ 0,
    /* graphData           */ nullptr,

    /* inputTensorNums     */ 1,
    /* inputTensorNames    */ g_nfru_v1_int8_InputTensorNames,
    /* inputTensorSets     */ g_nfru_v1_int8_InputTensorSets,
    /* inputTensorBindings */ g_nfru_v1_int8_InputTensorBindings,
    /* inputTensorFormats  */ nullptr,
    /* inputTensorDimSize  */ nullptr,
    /* inputTensorDims     */ nullptr,

    /* outputTensorNums    */ 1,
    /* outputTensorNames   */ g_nfru_v1_int8_OutputTensorNames,
    /* outputTensorSets    */ g_nfru_v1_int8_OutputTensorSets,
    /* outputTensorBindings*/ g_nfru_v1_int8_OutputTensorBindings,
    /* outputTensorFormats */ nullptr,
    /* outputTensorDimSize */ nullptr,
    /* outputTensorDims    */ nullptr,
};

#endif /* NFRU_V1_INT8_H */
