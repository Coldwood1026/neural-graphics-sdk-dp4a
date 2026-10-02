/*
 * nss_v1_0_1_high_int8.h -- NSS v1.0.1 "high" (quality) int8 data-graph model descriptor.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Replaces the header Model_Parser generated from
 * `data_graphs/nss/nss_v1_0_1_high_int8.vgf`. See `nfru_v1_int8.h` for the full
 * reasoning; the short version is that the only consumer of this blob was
 * vkCreateDataGraphPipelinesARM, that path is replaced by a portable dp4a backend
 * which carries its own baked weights, and Model_Parser crashes on this
 * repository's own `.vgf` files.
 * =============================================================================
 *
 * Counts come from the VGF itself, which declares `graph_partition_0_arg_0` and
 * `graph_partition_0_res_0` / `_res_1`: one argument, two results.
 *
 * Names come from `ffx_nss.cpp`, which carries them as explicit data-graph
 * duplicates in its binding tables, together with the channel layout:
 *
 *     // nss VGF: Resource_1_output = KPN coefficients (36ch, 1/4 res),
 *     //          Resource_2_output = temporal feedback (4ch)
 *     {FFX_NSS_RESOURCE_IDENTIFIER_KPN_TENSOR,      "Resource_1_output"}
 *     {FFX_NSS_RESOURCE_IDENTIFIER_FEEDBACK_TENSOR, "Resource_2_output"}
 *
 * `patchResourceIdentifier()` matches these by string, so they must stay
 * byte-identical to the effect's table.
 */
#ifndef NSS_V1_0_1_HIGH_INT8_H
#define NSS_V1_0_1_HIGH_INT8_H

#include <FidelityFX/host/ffx_types.h>

/* Order matches how the effect fills FfxDataGraphJobDescription::*Tensors. */
static const char* g_nss_v1_0_1_high_int8_InputTensorNames[] = {
    "Resource_0_input",
};
static const uint32_t g_nss_v1_0_1_high_int8_InputTensorSets[]     = { 0 };
static const uint32_t g_nss_v1_0_1_high_int8_InputTensorBindings[] = { 0 };

static const char* g_nss_v1_0_1_high_int8_OutputTensorNames[] = {
    "Resource_1_output",   /* KPN coefficients, 36ch, quarter resolution */
    "Resource_2_output",   /* temporal feedback, 4ch                     */
};
static const uint32_t g_nss_v1_0_1_high_int8_OutputTensorSets[]     = { 0, 0 };
static const uint32_t g_nss_v1_0_1_high_int8_OutputTensorBindings[] = { 0, 1 };

#define NSS_V1_0_1_HIGH_INT8_ENTRY_POINT "nss_v1_0_1_high_int8"

static const FfxDataGraphBlob g_nss_v1_0_1_high_int8_Info = {
    /* constantNums        */ 0,
    /* constantIds         */ nullptr,
    /* constantFormats     */ nullptr,
    /* constantShapeSize   */ nullptr,
    /* constantShapes      */ nullptr,
    /* constantSparsityDimensions */ nullptr,
    /* constantDataSize    */ nullptr,
    /* constantDatas       */ nullptr,

    /* graphEntryPoint     */ NSS_V1_0_1_HIGH_INT8_ENTRY_POINT,
    /* graphDataSize       */ 0,
    /* graphData           */ nullptr,

    /* inputTensorNums     */ 1,
    /* inputTensorNames    */ g_nss_v1_0_1_high_int8_InputTensorNames,
    /* inputTensorSets     */ g_nss_v1_0_1_high_int8_InputTensorSets,
    /* inputTensorBindings */ g_nss_v1_0_1_high_int8_InputTensorBindings,
    /* inputTensorFormats  */ nullptr,
    /* inputTensorDimSize  */ nullptr,
    /* inputTensorDims     */ nullptr,

    /* outputTensorNums    */ 2,
    /* outputTensorNames   */ g_nss_v1_0_1_high_int8_OutputTensorNames,
    /* outputTensorSets    */ g_nss_v1_0_1_high_int8_OutputTensorSets,
    /* outputTensorBindings*/ g_nss_v1_0_1_high_int8_OutputTensorBindings,
    /* outputTensorFormats */ nullptr,
    /* outputTensorDimSize */ nullptr,
    /* outputTensorDims    */ nullptr,
};

#endif /* NSS_V1_0_1_HIGH_INT8_H */
