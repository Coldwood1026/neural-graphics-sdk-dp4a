/*
 * nss_dp4a_dx12.cpp -- D3D12 host for the NSS int8 dp4a graph.
 *
 * SPDX-License-Identifier: MIT
 * =============================================================================
 * WHAT THIS IS, AND HOW IT DIFFERS FROM THE VULKAN HOST
 * =============================================================================
 * This is not a translation of nss_dp4a.cpp. That file fuses planning and emission:
 * inside one per-layer loop body it writes descriptors, binds the pipeline, computes the
 * push constants, computes the dispatch grid and issues vkCmdDispatch, with roughly 40
 * distinct Vulkan entry points woven through context creation, resource management and
 * the record path. There is no plan object to emit twice, which is precisely why the
 * NFRU backend was ported with a shared BuildRecordPlan() and this one could not be.
 *
 * What it *does* have is nss::Model, and Model::dispatches() returns a vector<Dispatch>
 * that is already API-agnostic: kernel kind, input/output buffer indices, input/output
 * shapes, the dispatch grid, a 20-word push-constant block, and the weight/bias/mult/
 * shift/LUT pointers for the layer. The geometry comes from the generated plan header
 * (nss_model_plan_<quality>.h), which is pure data.
 *
 * So this file is an emitter over Model::dispatches(), sharing Model -- and therefore the
 * shape derivation and the baked weights -- with the Vulkan host rather than duplicating
 * either. What is duplicated is the ~200 lines of binding/grid arithmetic that
 * nss_dp4a.cpp keeps inline; see Record() for the two places that had to be copied
 * deliberately and the comments saying why.
 *
 * =============================================================================
 * TWO THINGS D3D12 MAKES EASIER THAN VULKAN HERE
 * =============================================================================
 * 1. Weight upload needs a queue. On Vulkan the module had to be taught to record its
 *    copies into a caller-supplied command buffer, because the SDK's Vulkan backend
 *    holds no VkQueue and vkCreateCommandPool demands a queue family the device was
 *    created with. D3D12 has no such constraint: an application may create additional
 *    queues at will, so this context simply makes its own compute queue for the one-shot
 *    upload and never needs the host's.
 *
 * 2. Binding is by descriptor heap offset rather than by descriptor set. One
 *    shader-visible CBV_SRV_UAV heap holds eight descriptors per dispatch (six SRVs,
 *    two UAVs), built once at init and selected per dispatch by offset, so recording a
 *    frame does no descriptor writes at all.
 *
 * =============================================================================
 * WHAT THIS DOES NOT DO
 * =============================================================================
 * No Tensor Core path. The module's conv_tc kernel is built for
 * VK_KHR_cooperative_matrix with NVIDIA's 16x16x32 sint8 shape; D3D12's counterpart is
 * SM 6.9 WaveMatrix, which is a different feature on different hardware and would need
 * its own kernel plus its own selection logic. The module already routes every layer to
 * DP4A when cooperative matrix is unavailable, so a DP4A-only D3D12 host is a supported
 * configuration rather than a degraded one -- it is just not the fast one where the
 * hardware could do better.
 */

#include "nss_dp4a.h"
#include "nss_dp4a_dx12.h"
#include "nss_dp4a_model.h"
#include "nss_shaders_dxil.h"

#include <d3d12.h>
#include <dxgi.h>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* Binding layout, and it must match conv_rq.hlsl / resize2x.hlsl / concat_copy.hlsl.
 * The Vulkan side uses descriptor set 0 bindings 0..7; these numbers are not copied
 * across, because a binding number only means anything relative to its own API. */
constexpr uint32_t kSrvCount = 6;      // t0 actIn, t1 wPk, t2 biasCr, t3 mult, t4 shiftB, t5 lut
constexpr uint32_t kUavCount = 2;      // u0 actOut, u1 dbgBuf
constexpr uint32_t kDescriptorsPerDispatch = kSrvCount + kUavCount;

/* Root parameter indices. */
constexpr uint32_t kRootSrvTable = 0;
constexpr uint32_t kRootUavTable = 1;
constexpr uint32_t kRootConstants = 2;
constexpr uint32_t kPushWords = 20;    // the largest push block any kernel takes

/* Carried over verbatim from nss_dp4a.cpp. These are arithmetic contracts with the
 * kernels, not tuning knobs, so they cannot be chosen independently here. */
constexpr uint32_t CONV_OCG    = 4u;
constexpr uint32_t CONV_TILE_X = 1u;

inline uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1u) / a * a; }

struct Buf
{
    ID3D12Resource*      res   = nullptr;
    uint64_t             size  = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct Dx12
{
    ID3D12Device*        device   = nullptr;
    nss::Model           model;
    uint32_t             quality  = 0;

    ID3D12RootSignature* rootSig   = nullptr;
    ID3D12PipelineState* psoConv    = nullptr;
    ID3D12PipelineState* psoConv2x2 = nullptr;
    ID3D12PipelineState* psoResize  = nullptr;
    ID3D12PipelineState* psoConcat  = nullptr;

    ID3D12DescriptorHeap* heap      = nullptr;
    uint32_t              heapInc   = 0;
    D3D12_GPU_DESCRIPTOR_HANDLE heapGpuBase = {};

    std::vector<Buf>     scratch;          // one per nss::Model::buffers() entry
    std::vector<Buf>     layerConsts;      // 5 per dispatch: w, bc, mult, shift, lut
    Buf                  tail;             // kTensorTailSlack, shared, never written
    uint64_t             internalBytes = 0;

    /* Owned solely for the one-shot upload at init. See the header note. */
    ID3D12CommandQueue*       uploadQueue  = nullptr;
    ID3D12CommandAllocator*   uploadAlloc  = nullptr;
    ID3D12GraphicsCommandList* uploadList  = nullptr;
    ID3D12Fence*              uploadFence  = nullptr;
    HANDLE                    uploadEvent  = nullptr;
    uint64_t                  uploadValue  = 0;
};

/* ------------------------------------------------------------------ buffers */

bool CreateBuffer(Dx12* c, uint64_t bytes, Buf* out)
{
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment        = 0;
    desc.Width            = bytes;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.Format           = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    /* Unordered access is not optional: every one of these is bound as a raw UAV by at
     * least one kernel. Without the flag, creating the view fails much later and the
     * error names a typeless format rather than the missing flag. */
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(c->device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(&out->res))))
    {
        return false;
    }
    out->size  = bytes;
    out->state = D3D12_RESOURCE_STATE_COMMON;
    return true;
}

/* Record a byte-for-byte upload of `bytes` from `data` into an already-created buffer,
 * onto the upload command list. Flushed by UploadFlush() once, after all of them. */
void UploadRange(Dx12* c, Buf* dst, const void* data, uint64_t bytes)
{
    ID3D12Resource* staging = nullptr;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (FAILED(c->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(&staging))))
    {
        return;
    }

    void* mapped = nullptr;
    if (SUCCEEDED(staging->Map(0, nullptr, &mapped)) && mapped != nullptr)
    {
        std::memcpy(mapped, data, (size_t)bytes);
        staging->Unmap(0, nullptr);
    }

    /* COMMON -> COPY_DEST -> COMMON. COMMON is a legal predecessor for a copy, but being
     * explicit keeps the tracking below honest. */
    D3D12_RESOURCE_BARRIER toCopy = {};
    toCopy.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition.pResource   = dst->res;
    toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toCopy.Transition.StateBefore = dst->state;
    toCopy.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    c->uploadList->ResourceBarrier(1, &toCopy);

    c->uploadList->CopyBufferRegion(dst->res, 0, staging, 0, bytes);

    D3D12_RESOURCE_BARRIER toCommon = toCopy;
    toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    toCommon.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
    c->uploadList->ResourceBarrier(1, &toCommon);
    dst->state = D3D12_RESOURCE_STATE_COMMON;

    /* The barriers above already ordered the copy, so releasing the staging resource is
     * safe only after the GPU is done. Deferring to a destructor-like list would be more
     * correct in general; here every upload happens at init and FlushUploads() waits, so
     * a simple release after the wait is enough -- see UploadFlush(). */
    staging->Release();
}

bool UploadFlush(Dx12* c)
{
    if (FAILED(c->uploadList->Close())) return false;
    ID3D12CommandList* lists[] = { c->uploadList };
    c->uploadQueue->ExecuteCommandLists(1, lists);
    ++c->uploadValue;
    if (FAILED(c->uploadQueue->Signal(c->uploadFence, c->uploadValue))) return false;
    if (c->uploadFence->GetCompletedValue() < c->uploadValue)
    {
        if (FAILED(c->uploadFence->SetEventOnCompletion(c->uploadValue, c->uploadEvent))) return false;
        WaitForSingleObject(c->uploadEvent, INFINITE);
    }
    return true;
}

/* ------------------------------------------------------------------ pipelines */

bool CreateRootSignature(Dx12* c)
{
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors                    = kSrvCount;
    ranges[0].BaseShaderRegister                = 0;
    ranges[0].RegisterSpace                     = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;

    ranges[1].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors                    = kUavCount;
    ranges[1].BaseShaderRegister                = 0;
    ranges[1].RegisterSpace                     = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[3] = {};
    params[kRootSrvTable].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[kRootSrvTable].DescriptorTable.NumDescriptorRanges = 1;
    params[kRootSrvTable].DescriptorTable.pDescriptorRanges    = &ranges[0];
    params[kRootSrvTable].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;

    params[kRootUavTable].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[kRootUavTable].DescriptorTable.NumDescriptorRanges = 1;
    params[kRootUavTable].DescriptorTable.pDescriptorRanges    = &ranges[1];
    params[kRootUavTable].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;

    /* Root constants rather than a CBV: the kernels take at most 80 bytes, they change
     * per dispatch, and a root constant costs no descriptor and no allocation. */
    params[kRootConstants].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[kRootConstants].Constants.ShaderRegister  = 0;
    params[kRootConstants].Constants.RegisterSpace   = 0;
    params[kRootConstants].Constants.Num32BitValues  = kPushWords;
    params[kRootConstants].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 3;
    rsd.pParameters   = params;
    rsd.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* blob = nullptr;
    ID3DBlob* err  = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)))
    {
        if (err) err->Release();
        return false;
    }
    const HRESULT hr = c->device->CreateRootSignature(0, blob->GetBufferPointer(),
                                                      blob->GetBufferSize(),
                                                      IID_PPV_ARGS(&c->rootSig));
    blob->Release();
    if (err) err->Release();
    return SUCCEEDED(hr);
}

bool CreatePipeline(Dx12* c, const unsigned char* dxil, unsigned int dxilSize, ID3D12PipelineState** out)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    desc.pRootSignature = c->rootSig;
    desc.CS.pShaderBytecode = dxil;
    desc.CS.BytecodeLength  = dxilSize;
    return SUCCEEDED(c->device->CreateComputePipelineState(&desc, IID_PPV_ARGS(out)));
}

/* ------------------------------------------------------------------ init */

bool BuildResources(Dx12* c)
{
    const std::vector<nss::Shape>& bufs  = c->model.buffers();
    const std::vector<nss::Dispatch>& dsp = c->model.dispatches();

    c->scratch.assign(bufs.size(), Buf{});
    for (size_t i = 0; i < bufs.size(); ++i)
    {
        const nss::Shape& s = bufs[i];
        if (s.h == 0) continue;

        /* Bordered, plus the trailing slack the Tensor Core kernel would need. The DP4A
         * paths do not read past the interior, but the slack is kept because the buffer
         * is shared with -- and the size is part of -- the module's contract with the
         * kernels, and a difference here would be invisible until a border case. */
        const uint64_t bytes = (uint64_t)(s.h + 2u) * (s.w + 2u) * s.c + nss::kTensorTailSlack;
        if (!CreateBuffer(c, bytes, &c->scratch[i])) return false;
        c->internalBytes += bytes;

        /* Pre-fill with the activation zero point. This is load-bearing, not tidiness:
         * the kernels treat the border as z_a = -128 = 0x80808080, and the whole
         * "out-of-range taps contribute nothing" equivalence rests on those bytes
         * actually holding it. */
        std::vector<uint32_t> fill((size_t)(bytes / 4u), 0x80808080u);
        UploadRange(c, &c->scratch[i], fill.data(), (uint64_t)fill.size() * 4u);
    }

    /* One trailing-slack buffer, shared as the "read a little past the end" landing zone
     * for the tail-group branches. It is never written, so a single instance suffices. */
    if (!CreateBuffer(c, nss::kTensorTailSlack, &c->tail)) return false;
    c->internalBytes += nss::kTensorTailSlack;
    std::vector<uint8_t> zeros((size_t)nss::kTensorTailSlack, 0);
    UploadRange(c, &c->tail, zeros.data(), nss::kTensorTailSlack);

    /* Five constant buffers per dispatch: weights, bias+correction, multiplier, shift,
     * LUT. */
    c->layerConsts.assign(dsp.size() * 5u, Buf{});

    for (size_t i = 0; i < dsp.size(); ++i)
    {
        const nss::Dispatch& d = dsp[i];
        if (d.kind != 0) continue;   // only conv carries weights and quantisation tables

        const uint64_t wBytes = (uint64_t)d.wWords * 4u;
        const uint64_t qBytes = (uint64_t)d.multCount * 4u;

        const uint64_t sizes[5] = { wBytes, qBytes, qBytes, qBytes, qBytes };
        const void*    srcs[5]  = { d.w, d.bc, d.mult, d.shift, d.lut };

        for (uint32_t k = 0; k < 5; ++k)
        {
            if (sizes[k] == 0 || srcs[k] == nullptr) continue;
            Buf& b = c->layerConsts[i * 5u + k];
            const uint64_t aligned = (sizes[k] + 255u) & ~255ull;   // 256B-aligned raw views
            if (!CreateBuffer(c, aligned, &b)) return false;
            c->internalBytes += aligned;
            UploadRange(c, &b, srcs[k], sizes[k]);
        }
    }

    return true;
}

bool BuildDescriptors(Dx12* c)
{
    const std::vector<nss::Dispatch>& dsp = c->model.dispatches();

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = (UINT)(dsp.size() * kDescriptorsPerDispatch) + 8u;
    hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(c->device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&c->heap)))) return false;

    c->heapInc     = c->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    c->heapGpuBase = c->heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE cpuBase = c->heap->GetCPUDescriptorHandleForHeapStart();

    for (size_t i = 0; i < dsp.size(); ++i)
    {
        const nss::Dispatch& d = dsp[i];
        D3D12_CPU_DESCRIPTOR_HANDLE base = cpuBase;
        base.ptr += (SIZE_T)(i * kDescriptorsPerDispatch) * c->heapInc;

        /* Slot order matches kSrvCount/kUavCount above: 0..5 SRVs, 6..7 UAVs. */
        const Buf* srv[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
        const Buf* uav[2] = { nullptr, nullptr };

        if (d.kind == 0u)
        {
            srv[0] = (d.inBuf < 0) ? nullptr : &c->scratch[d.inBuf];     // actIn, patched below
            srv[1] = &c->layerConsts[i * 5u + 0];
            srv[2] = &c->layerConsts[i * 5u + 1];
            srv[3] = &c->layerConsts[i * 5u + 2];
            srv[4] = &c->layerConsts[i * 5u + 3];
            srv[5] = &c->layerConsts[i * 5u + 4];
        }
        else
        {
            srv[0] = (d.inBuf < 0) ? nullptr : &c->scratch[d.inBuf];
            srv[1] = &c->tail;   // unused by resize/concat, but the slot must be valid
            srv[5] = &c->tail;
        }
        uav[0] = &c->scratch[d.outBuf];
        uav[1] = &c->tail;       // debug sink; the kernels only touch it when dbg != 0

        for (uint32_t k = 0; k < kSrvCount; ++k)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Format                  = DXGI_FORMAT_R32_TYPELESS;
            sd.ViewDimension           = D3D12_SRV_DIMENSION_BUFFER;
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Buffer.NumElements      = (UINT)((srv[k] ? srv[k]->size : 4u) / 4u);
            sd.Buffer.Flags            = D3D12_BUFFER_SRV_FLAG_RAW;

            D3D12_CPU_DESCRIPTOR_HANDLE h = base;
            h.ptr += (SIZE_T)k * c->heapInc;
            /* A null slot cannot be left uninitialised: a descriptor table must be fully
             * populated or the GPU reads whatever was there. Where a kernel does not use a
             * binding, point it at a valid 4-byte buffer. */
            if (srv[k] == nullptr) srv[k] = &c->tail;
            sd.Buffer.NumElements = (UINT)(srv[k]->size / 4u);
            c->device->CreateShaderResourceView(srv[k]->res, &sd, h);
        }

        for (uint32_t k = 0; k < kUavCount; ++k)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format             = DXGI_FORMAT_R32_TYPELESS;
            ud.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
            ud.Buffer.NumElements = (UINT)(uav[k]->size / 4u);
            ud.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;

            D3D12_CPU_DESCRIPTOR_HANDLE h = base;
            h.ptr += (SIZE_T)(kSrvCount + k) * c->heapInc;
            c->device->CreateUnorderedAccessView(uav[k]->res, nullptr, &ud, h);
        }
    }
    return true;
}

/* ------------------------------------------------------------------ record */

/* Move every scratch buffer into UNORDERED_ACCESS once, at init. After that the only
 * synchronisation a frame needs is a UAV barrier between dispatches, which is what the
 * kernels actually require: each layer reads what the previous one wrote, and nothing
 * here is ever a copy source or a shader resource in a fixed-function sense. */
void TransitionScratchToUav(Dx12* c, ID3D12GraphicsCommandList* cmd)
{
    for (Buf& b : c->scratch)
    {
        if (b.res == nullptr || b.state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) continue;
        D3D12_RESOURCE_BARRIER bar = {};
        bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = b.res;
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bar.Transition.StateBefore = b.state;
        bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cmd->ResourceBarrier(1, &bar);
        b.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
}

void RawSrv(ID3D12Device* dev, ID3D12Resource* res, uint64_t size, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format                  = DXGI_FORMAT_R32_TYPELESS;
    sd.ViewDimension           = D3D12_SRV_DIMENSION_BUFFER;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Buffer.NumElements      = (UINT)(size / 4u);
    sd.Buffer.Flags            = D3D12_BUFFER_SRV_FLAG_RAW;
    dev->CreateShaderResourceView(res, &sd, h);
}

void RawUav(ID3D12Device* dev, ID3D12Resource* res, uint64_t size, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format             = DXGI_FORMAT_R32_TYPELESS;
    ud.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = (UINT)(size / 4u);
    ud.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(res, nullptr, &ud, h);
}

/*
 * Rebuild one dispatch's eight descriptors with the host's buffers substituted.
 *
 * Three dispatches per frame touch host memory: the one that reads the graph input
 * (inBuf < 0) and the two that write the graph outputs. Their descriptors cannot be
 * built at init because the host buffers are not known until dispatch time. Rewriting
 * three whole eight-descriptor blocks per frame is cheaper than maintaining a second
 * descriptor table, and it keeps every table in the heap contiguous.
 *
 * The caller passes the scratch views that would otherwise be there for the slots it is
 * not overriding.
 */
void PatchHostDescriptors(Dx12* c, size_t di,
                          ID3D12Resource* hostIn, uint64_t hostInSize,
                          ID3D12Resource* hostOut, uint64_t hostOutSize)
{
    const nss::Dispatch& d = c->model.dispatches()[di];
    const std::vector<nss::Shape>& bufs = c->model.buffers();

    D3D12_CPU_DESCRIPTOR_HANDLE base = c->heap->GetCPUDescriptorHandleForHeapStart();
    base.ptr += (SIZE_T)(di * kDescriptorsPerDispatch) * c->heapInc;

    auto at = [&](uint32_t k)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = base;
        h.ptr += (SIZE_T)k * c->heapInc;
        return h;
    };

    if (hostIn != nullptr)
    {
        RawSrv(c->device, hostIn, hostInSize, at(0));
    }
    if (hostOut != nullptr)
    {
        RawUav(c->device, hostOut, hostOutSize, at(kSrvCount));
    }
    (void)d;
    (void)bufs;
}

bool RecordDispatches(Dx12* c, ID3D12GraphicsCommandList* cmd, const NssDp4aDispatchInfo* info)
{
    const std::vector<nss::Dispatch>& dsp = c->model.dispatches();
    const std::vector<nss::Shape>&    bufs = c->model.buffers();
    const std::vector<int>&           outs = c->model.outputBufs();

    const uint64_t hostIn     = info->input.buffer;
    const uint64_t hostOutKpn = info->outputKpn.buffer;
    const uint64_t hostOutTmp = info->outputTemporal.buffer;

    /* Which internal buffer index maps to which host output. */
    int outKpnBuf = -1, outTmpBuf = -1;
    if (outs.size() >= 1) outKpnBuf = outs[0];
    if (outs.size() >= 2) outTmpBuf = outs[1];

    ID3D12DescriptorHeap* heaps[] = { c->heap };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(c->rootSig);

    TransitionScratchToUav(c, cmd);

    for (size_t di = 0; di < dsp.size(); ++di)
    {
        const nss::Dispatch& d = dsp[di];

        /* ---- host-bound slots, if this dispatch has any ---- */
        ID3D12Resource* patchIn   = nullptr; uint64_t patchInSize  = 0;
        ID3D12Resource* patchOut  = nullptr; uint64_t patchOutSize = 0;

        if (d.inBuf < 0 && hostIn != 0)
        {
            patchIn     = reinterpret_cast<ID3D12Resource*>(hostIn);
            patchInSize = info->input.size;
        }
        if (d.outBuf == outKpnBuf && hostOutKpn != 0)
        {
            patchOut     = reinterpret_cast<ID3D12Resource*>(hostOutKpn);
            patchOutSize = info->outputKpn.size;
        }
        else if (d.outBuf == outTmpBuf && hostOutTmp != 0)
        {
            patchOut     = reinterpret_cast<ID3D12Resource*>(hostOutTmp);
            patchOutSize = info->outputTemporal.size;
        }
        if (patchIn != nullptr || patchOut != nullptr)
        {
            PatchHostDescriptors(c, di, patchIn, patchInSize, patchOut, patchOutSize);
        }

        /* ---- bind ---- */
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = c->heapGpuBase;
        gpu.ptr += (UINT64)(di * kDescriptorsPerDispatch) * c->heapInc;
        cmd->SetComputeRootDescriptorTable(kRootSrvTable, gpu);

        D3D12_GPU_DESCRIPTOR_HANDLE gpuUav = gpu;
        gpuUav.ptr += (UINT64)kSrvCount * c->heapInc;
        cmd->SetComputeRootDescriptorTable(kRootUavTable, gpuUav);

        /* ---- push constants ---- */
        /*
         * The 20-word block comes straight from the baked plan; only the three words a
         * kernel cannot know at bake time are overridden, and only for conv (the
         * resize/concat blocks are five words and have no such fields).
         *
         * Copied deliberately from nss_dp4a.cpp, because it is the same contract:
         *
         *   [14] inH        interior height of whatever actIn points at. When actIn is the
         *                   host's buffer the kernel cannot derive it, so it is passed.
         *   [15] inBorded   0 when reading the host's tight buffer -- the kernel then
         *                   substitutes 0x80808080 for out-of-range taps instead of
         *                   reading a border, which is bit-identical (see conv_rq.hlsl).
         *   [16] outBorded  0 when writing the host's tight buffer, which is what saves
         *                   the output copy-back.
         */
        uint32_t push[kPushWords];
        std::memcpy(push, d.push, sizeof(uint32_t) * kPushWords);
        uint32_t pushSize = d.pushSize;

        if (d.kind == 0)
        {
            push[14] = (d.inBuf < 0) ? c->model.height() : bufs[d.inBuf].h;
            push[15] = (d.inBuf < 0) ? 0u : 1u;
            push[16] = (patchOut != nullptr) ? 0u : 1u;
            /* kPushWords is 20 and the conv block is 17; the trailing three stay zero
             * because the Tensor Core kernel that would use them is not built here. */
            pushSize = 17u;
        }
        cmd->SetComputeRoot32BitConstants(kRootConstants, pushSize, push, 0);

        /* ---- grid ---- */
        /*
         * Also copied from nss_dp4a.cpp: the plan's gx/gy/gz are in pixels or in oc4
         * groups, not in workgroups, and the folding has to agree with the kernel's
         * __init__/OCG layout exactly.
         */
        uint32_t dx, dy, dz;
        if (d.kind == 0u)
        {
            const uint32_t gz = (d.gz + CONV_OCG - 1u) / CONV_OCG;
            const uint32_t gx = (d.gx + CONV_TILE_X - 1u) / CONV_TILE_X;
            dx = (gx + 7u) / 8u;
            dy = (d.gy + 7u) / 8u;
            dz = gz;
        }
        else
        {
            dx = (d.gx + 7u) / 8u;
            dy = (d.gy + 7u) / 8u;
            dz = d.gz;
        }

        ID3D12PipelineState* pso = (d.kind == 0u) ? c->psoConv
                                 : (d.kind == 1u) ? c->psoResize
                                                  : c->psoConcat;
        cmd->SetPipelineState(pso);
        cmd->Dispatch(dx, dy, dz);

        /* Every layer reads what the previous one wrote, so a UAV barrier between them is
         * required and sufficient -- there is no render-target or copy state involved. */
        D3D12_RESOURCE_BARRIER ub = {};
        ub.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        ub.UAV.pResource = c->scratch[d.outBuf].res;
        cmd->ResourceBarrier(1, &ub);
    }

    return true;
}

}  // namespace

/* ==========================================================================
 * Public entry points, for the SDK's D3D12 backend.
 * ========================================================================== */

extern "C" NssDx12* NssDx12Create(const NssDx12CreateInfo* createInfo, const char** outError)
{
    if (createInfo == nullptr || createInfo->device == 0) return nullptr;

    Dx12* c = new Dx12();
    c->device  = reinterpret_cast<ID3D12Device*>(createInfo->device);
    c->quality = createInfo->quality;

    std::string err;
    if (!c->model.load(createInfo->quality == NSS_DP4A_QUALITY_HIGH ? 0 : 1,
                       createInfo->width, createInfo->height, &err))
    {
        if (outError) *outError = "nss_dp4a dx12: model load failed";
        delete c;
        return nullptr;
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(c->device->CreateCommandQueue(&qd, IID_PPV_ARGS(&c->uploadQueue))) ||
        FAILED(c->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&c->uploadAlloc))) ||
        FAILED(c->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, c->uploadAlloc, nullptr, IID_PPV_ARGS(&c->uploadList))) ||
        FAILED(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->uploadFence))))
    {
        if (outError) *outError = "nss_dp4a dx12: upload queue/allocator/list/fence creation failed";
        NssDx12Destroy((NssDx12*)c);
        return nullptr;
    }
    c->uploadEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    bool ok = CreateRootSignature(c)
           && CreatePipeline(c, nss_model::kDxil_conv_rq,   nss_model::kDxil_conv_rq_size,   &c->psoConv)
           && CreatePipeline(c, nss_model::kDxil_resize2x,  nss_model::kDxil_resize2x_size,  &c->psoResize)
           && CreatePipeline(c, nss_model::kDxil_concat_copy, nss_model::kDxil_concat_copy_size, &c->psoConcat)
           && BuildResources(c)
           && UploadFlush(c)
           && BuildDescriptors(c);

    if (!ok)
    {
        if (outError) *outError = "nss_dp4a dx12: pipeline/resource/descriptor setup failed";
        NssDx12Destroy((NssDx12*)c);
        return nullptr;
    }
    return (NssDx12*)c;
}

extern "C" void NssDx12Destroy(NssDx12* ctx)
{
    Dx12* c = (Dx12*)ctx;
    if (c == nullptr) return;

    auto rel = [](auto*& p) { if (p) { p->Release(); p = nullptr; } };
    for (Buf& b : c->scratch)     rel(b.res);
    for (Buf& b : c->layerConsts) rel(b.res);
    rel(c->tail.res);
    rel(c->psoConv); rel(c->psoConv2x2); rel(c->psoResize); rel(c->psoConcat);
    rel(c->rootSig); rel(c->heap);
    rel(c->uploadList); rel(c->uploadAlloc); rel(c->uploadQueue); rel(c->uploadFence);
    if (c->uploadEvent) CloseHandle(c->uploadEvent);
    delete c;
}

extern "C" bool NssDx12Record(NssDx12* ctx, uint64_t commandList, const NssDp4aDispatchInfo* info)
{
    Dx12* c = (Dx12*)ctx;
    if (c == nullptr || info == nullptr) return false;
    return RecordDispatches(c, reinterpret_cast<ID3D12GraphicsCommandList*>(commandList), info);
}

extern "C" uint64_t NssDx12InternalBytes(NssDx12* ctx)
{
    Dx12* c = (Dx12*)ctx;
    return c ? c->internalBytes : 0;
}
