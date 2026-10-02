/*
 * nss_v1_0_1_mid_low_int8.h -- NSS v1.0.1 "mid/low" (sparse) int8 data-graph descriptor.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * Replaces the header Model_Parser generated from
 * `data_graphs/nss/nss_v1_0_1_mid_low_int8.vgf`. See `nfru_v1_int8.h` for the full
 * reasoning.
 * =============================================================================
 *
 * This graph has the same interface as the high-quality one -- one argument, two
 * results (the VGF declares `graph_partition_0_arg_0`, `_res_0`, `_res_1`) -- but a
 * different internal structure and its own weight set. `ffx_nss.cpp` selects
 * between the two on FFX_NSS_SHADER_QUALITY_MODE_QUALITY: quality mode uses the
 * high graph, everything else uses this one.
 */
#ifndef NSS_V1_0_1_MID_LOW_INT8_H
#define NSS_V1_0_1_MID_LOW_INT8_H

#include <FidelityFX/host/ffx_types.h>

static const char* g_nss_v1_0_1_mid_low_int8_InputTensorNames[] = {
    "Resource_0_input",
};
static const uint32_t g_nss_v1_0_1_mid_low_int8_InputTensorSets[]     = { 0 };
static const uint32_t g_nss_v1_0_1_mid_low_int8_InputTensorBindings[] = { 0 };

static const char* g_nss_v1_0_1_mid_low_int8_OutputTensorNames[] = {
    "Resource_1_output",   /* KPN coefficients, 36ch, quarter resolution */
    "Resource_2_output",   /* temporal feedback, 4ch                     */
};
static const uint32_t g_nss_v1_0_1_mid_low_int8_OutputTensorSets[]     = { 0, 0 };
static const uint32_t g_nss_v1_0_1_mid_low_int8_OutputTensorBindings[] = { 0, 1 };

#define NSS_V1_0_1_MID_LOW_INT8_ENTRY_POINT "nss_v1_0_1_mid_low_int8"

static const FfxDataGraphBlob g_nss_v1_0_1_mid_low_int8_Info = {
    /* constantNums        */ 0,
    /* constantIds         */ nullptr,
    /* constantFormats     */ nullptr,
    /* constantShapeSize   */ nullptr,
    /* constantShapes      */ nullptr,
    /* constantSparsityDimensions */ nullptr,
    /* constantDataSize    */ nullptr,
    /* constantDatas       */ nullptr,

    /* graphEntryPoint     */ NSS_V1_0_1_MID_LOW_INT8_ENTRY_POINT,
    /* graphDataSize       */ 0,
    /* graphData           */ nullptr,

    /* inputTensorNums     */ 1,
    /* inputTensorNames    */ g_nss_v1_0_1_mid_low_int8_InputTensorNames,
    /* inputTensorSets     */ g_nss_v1_0_1_mid_low_int8_InputTensorSets,
    /* inputTensorBindings */ g_nss_v1_0_1_mid_low_int8_InputTensorBindings,
    /* inputTensorFormats  */ nullptr,
    /* inputTensorDimSize  */ nullptr,
    /* inputTensorDims     */ nullptr,

    /* outputTensorNums    */ 2,
    /* outputTensorNames   */ g_nss_v1_0_1_mid_low_int8_OutputTensorNames,
    /* outputTensorSets    */ g_nss_v1_0_1_mid_low_int8_OutputTensorSets,
    /* outputTensorBindings*/ g_nss_v1_0_1_mid_low_int8_OutputTensorBindings,
    /* outputTensorFormats */ nullptr,
    /* outputTensorDimSize */ nullptr,
    /* outputTensorDims    */ nullptr,
};

#endif /* NSS_V1_0_1_MID_LOW_INT8_H */
