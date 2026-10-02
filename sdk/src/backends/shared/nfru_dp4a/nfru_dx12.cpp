/*
 * nfru_dx12.cpp -- Direct3D 12 dp4a backend for the NFRU v1 inference graph.
 *
 * The counterpart of `nfru_vk.cpp`: same graph, same push constants, same integer
 * arithmetic, different API. Both must produce bit-identical int8 output, which is the
 * reason the graph and the kernel semantics live in one place and only the binding code
 * differs.
 *
 * The host owns the ID3D12Device and ID3D12CommandQueue, passed through
 * NfruDp4aCreateInfo::instance and ::queue, and the ID3D12GraphicsCommandList passed to
 * nfruDp4aRecord. This file creates only its own root signature, pipelines, weight
 * buffers and scratch; it never submits and never changes the state of the host's
 * resources.
 *
 * The kernel is `nfru_conv_rq.hlsl`, whose dot4add_i8packed intrinsic compiles to
 * dx.op.dot4AddPacked (Dot4AddI8Packed), i.e. the hardware DP4A instruction on any
 * shader-model 6.4 GPU. This needs no vendor extension at all -- unlike
 * VK_ARM_data_graph, DP4A is part of the base shader model -- which is exactly why this
 * backend can replace the Arm-only path.
 *
 * Descriptor strategy: one shader-visible heap holding a ring of descriptor groups.
 * A group is written and bound per dispatch; the ring is large enough that a descriptor
 * a pending command list still references is never overwritten.
 *
 * Byte offsets into tensors inside the scratch buffer are expressed through
 * D3D12_BUFFER_SRV/UAV `FirstElement`, which is in 4-byte units. Every tensor offset
 * this backend produces is 4-byte aligned by construction (the layout is packed in
 * uint32 words).
 */
#include "nfru_dx12.h"
#include "nfru_shaders_dxil.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <string.h>
#include <stdio.h>

namespace nfru {

namespace {

enum { kRootConstants = 0, kRootTable = 1, kRootParams = 2 };

const uint32_t kDescriptorRingGroups = 64;
const uint32_t kConvDescriptors = 6;      /* actIn, w, bc, mult, shift, actOut */

struct DxBuf {
    ID3D12Resource* res = nullptr;
    uint64_t size = 0;
};

class Dx12Device final : public Device {
public:
    ID3D12Device* dev = nullptr;
    ID3D12RootSignature* rootSig = nullptr;
    ID3D12PipelineState* psoConv = nullptr;
    ID3D12PipelineState* psoSupport = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE heapCpu = {};
    D3D12_GPU_DESCRIPTOR_HANDLE heapGpu = {};
    uint32_t descSize = 0;
    uint32_t ringIndex = 0;

    DxBuf weights{}, bias{}, mult{}, shift{}, scratch{};
    bool ownScratch = false;
    /* Scratch is used as a UAV by every dispatch and as a COPY SOURCE by the
     * STOP_TO_SCRATCH copy-out, and D3D12 requires the state to match the use. */
    D3D12_RESOURCE_STATES scratchState = D3D12_RESOURCE_STATE_COMMON;

    uint32_t width = 0, height = 0;
    TensorLayout layout{};
    char err[256] = {};

    const char* lastError() const override { return err; }
    bool valid() const override { return psoConv && psoSupport && heap && rootSig; }
    const char* backendName() const override { return "DX12/dp4a"; }
    uint64_t scratchBytes() const override { return layout.scratchBytes; }
    uint64_t internalBytes() const override
    {
        return weights.size + bias.size + mult.size + shift.size +
               (ownScratch ? scratch.size : 0);
    }

    /* ---------------------------------------------------------------- helpers */

    bool CreateUpload(uint64_t bytes, const void* data, DxBuf& out)
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                D3D12_RESOURCE_STATE_GENERIC_READ,
                                                nullptr, IID_PPV_ARGS(&out.res)))) {
            snprintf(err, sizeof(err), "CreateCommittedResource(upload, %llu bytes) failed",
                     (unsigned long long)bytes);
            return false;
        }
        void* mapped = nullptr;
        D3D12_RANGE readRange = { 0, 0 };
        if (FAILED(out.res->Map(0, &readRange, &mapped))) {
            snprintf(err, sizeof(err), "ID3D12Resource::Map failed during weight upload");
            return false;
        }
        memcpy(mapped, data, (size_t)bytes);
        out.res->Unmap(0, nullptr);
        out.size = bytes;
        return true;
    }

    /*
     * `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` is mandatory for any resource the
     * shader stores into. Without it the runtime rejects the resource's transition to
     * UNORDERED_ACCESS -- and it does NOT report that where the barrier is recorded: the
     * command list's Close() fails later with E_INVALIDARG, which points at nothing in
     * particular. Scratch is written by every dispatch in the graph, so this flag is not
     * optional here.
     */
    bool CreateDefault(uint64_t bytes, DxBuf& out)
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&out.res)))) {
            snprintf(err, sizeof(err), "CreateCommittedResource(default, %llu bytes) failed",
                     (unsigned long long)bytes);
            return false;
        }
        out.size = bytes;
        return true;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(uint32_t index) const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = heapCpu;
        h.ptr += (SIZE_T)index * descSize;
        return h;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(uint32_t index) const
    {
        D3D12_GPU_DESCRIPTOR_HANDLE h = heapGpu;
        h.ptr += (UINT64)index * descSize;
        return h;
    }

    /*
     * Views over the tensors, at a 4-byte-aligned word offset.
     *
     * `D3D12_BUFFER_SRV_FLAG_RAW` / `D3D12_BUFFER_UAV_FLAG_RAW` are REQUIRED: these are
     * raw (ByteAddressBuffer / RWByteAddressBuffer) views, and R32_TYPELESS is not a
     * format a *typed* buffer view may use. Omitting the flag does not fail the create
     * call or pipeline creation -- the debug layer reports it and D3D12 eventually removes
     * the device with DXGI_ERROR_INVALID_CALL, with nothing pointing at the view. Both
     * kernels load and store through raw views, so this is the only correct form.
     */
    void MakeSrv(uint32_t descIndex, ID3D12Resource* res, uint64_t offBytes, uint64_t bytes)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = (UINT64)(offBytes / 4);
        d.Buffer.NumElements = (UINT)((bytes - offBytes) / 4);
        d.Buffer.StructureByteStride = 0;
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        dev->CreateShaderResourceView(res, &d, CpuHandle(descIndex));
    }

    void MakeUav(uint32_t descIndex, ID3D12Resource* res, uint64_t offBytes, uint64_t bytes)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = (UINT64)(offBytes / 4);
        d.Buffer.NumElements = (UINT)((bytes - offBytes) / 4);
        d.Buffer.StructureByteStride = 0;
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        dev->CreateUnorderedAccessView(res, nullptr, &d, CpuHandle(descIndex));
    }

    /* --------------------------------------------------------------- creation */

    bool Init(const NfruDp4aCreateInfo& ci)
    {
        dev = reinterpret_cast<ID3D12Device*>(ci.device);
        if (!dev) {
            snprintf(err, sizeof(err), "NfruDp4aCreateInfo::device is not an ID3D12Device");
            return false;
        }
        width = ci.width;
        height = ci.height;

        D3D12_DESCRIPTOR_RANGE ranges[2] = {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = kConvDescriptors - 1;   /* t0..t4 */
        ranges[0].BaseShaderRegister = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;                      /* u0 */
        ranges[1].BaseShaderRegister = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = kConvDescriptors - 1;

        D3D12_ROOT_PARAMETER params[kRootParams] = {};
        params[kRootConstants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[kRootConstants].Constants.ShaderRegister = 0;
        params[kRootConstants].Constants.RegisterSpace = 0;
        params[kRootConstants].Constants.Num32BitValues = 16;
        params[kRootConstants].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[kRootTable].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[kRootTable].DescriptorTable.NumDescriptorRanges = 2;
        params[kRootTable].DescriptorTable.pDescriptorRanges = ranges;
        params[kRootTable].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsd = {};
        rsd.NumParameters = kRootParams;
        rsd.pParameters = params;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ID3DBlob* blob = nullptr;
        ID3DBlob* errBlob = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                               &blob, &errBlob))) {
            snprintf(err, sizeof(err), "D3D12SerializeRootSignature failed: %s",
                     errBlob ? (const char*)errBlob->GetBufferPointer() : "(no detail)");
            if (errBlob) errBlob->Release();
            return false;
        }
        const HRESULT hr = dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                                    blob->GetBufferSize(),
                                                    IID_PPV_ARGS(&rootSig));
        blob->Release();
        if (FAILED(hr)) {
            snprintf(err, sizeof(err), "CreateRootSignature failed (0x%08x)", (unsigned)hr);
            return false;
        }

        psoConv = MakePso(kNfruDxilConv, sizeof(kNfruDxilConv));
        if (!psoConv) return false;
        psoSupport = MakePso(kNfruDxilSupport, sizeof(kNfruDxilSupport));
        if (!psoSupport) return false;

        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = kDescriptorRingGroups * kConvDescriptors;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) {
            snprintf(err, sizeof(err), "CreateDescriptorHeap failed");
            return false;
        }
        heapCpu = heap->GetCPUDescriptorHandleForHeapStart();
        heapGpu = heap->GetGPUDescriptorHandleForHeapStart();
        descSize = dev->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        if (!CreateUpload(sizeof(kNfruWeights), kNfruWeights, weights)) return false;
        if (!CreateUpload(sizeof(kNfruBias), kNfruBias, bias)) return false;
        if (!CreateUpload(sizeof(kNfruMult), kNfruMult, mult)) return false;
        if (!CreateUpload(sizeof(kNfruShift), kNfruShift, shift)) return false;

        if (!ComputeTensorLayout(width, height, layout)) {
            /*
             * `NFRU_DP4A_LAYER_ONLY` runs ONE convolution with the host buffers as its
             * input and output, so the inter-layer layout is never consulted -- the layer's
             * geometry comes from its own kernel, stride and padding. It also cannot be
             * consistent at a reduced resolution: the x2 resize and the two
             * concatenations only close at the model's native size. Refusing to create the
             * context there made every 240x135 layer untestable on both backends.
             */
            if (!getenv("NFRU_DP4A_LAYER_ONLY")) {
                snprintf(err, sizeof(err),
                         "tensor layout inconsistent at %ux%u (a concatenation's inputs "
                         "disagree in spatial extent)", width, height);
                return false;
            }
            memset(&layout, 0, sizeof(layout));
            /* Generous, and only ever allocated -- never read -- in this mode. */
            layout.scratchBytes = ((uint64_t)width * height * 64u + 255u) & ~(uint64_t)255;
        }
        return true;
    }

    ID3D12PipelineState* MakePso(const uint32_t* dxil, size_t bytes)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rootSig;
        pd.CS.pShaderBytecode = dxil;
        pd.CS.BytecodeLength = bytes;
        ID3D12PipelineState* pso = nullptr;
        const HRESULT hr = dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
        if (FAILED(hr)) {
            snprintf(err, sizeof(err),
                     "CreateComputePipelineState failed (0x%08x): the device does not "
                     "support shader model 6.4 / DP4A", (unsigned)hr);
            return nullptr;
        }
        return pso;
    }

    void Destroy()
    {
        if (psoConv) { psoConv->Release(); psoConv = nullptr; }
        if (psoSupport) { psoSupport->Release(); psoSupport = nullptr; }
        if (rootSig) { rootSig->Release(); rootSig = nullptr; }
        if (heap) { heap->Release(); heap = nullptr; }
        auto rel = [](DxBuf& b) { if (b.res) { b.res->Release(); b.res = nullptr; } };
        rel(weights); rel(bias); rel(mult); rel(shift);
        if (ownScratch) rel(scratch);
    }

    /* -------------------------------------------------------------- recording */

    /*
     * A UAV barrier between dispatches.
     *
     * D3D12 does not order unordered-access writes against later reads; without this the
     * next dispatch can read a tensor before the previous one's stores are visible. On
     * Vulkan the equivalent is the compute->compute memory barrier the other backend
     * issues. Omitting it here does not fail loudly -- it produces a graph that is usually
     * right and occasionally not, which is the worst possible failure mode to debug.
     */
    void UavBarrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res)
    {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.UAV.pResource = res;
        cmd->ResourceBarrier(1, &b);
    }

    void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        if (before == after) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.Transition.pResource = res;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        cmd->ResourceBarrier(1, &b);
    }

    NfruDp4aResult record(uint64_t commandList, uint64_t inputResource, uint64_t inputOff,
                          uint64_t inputBytes, uint64_t outputResource, uint64_t outputOff,
                          uint64_t outputBytes, uint64_t scratchResource,
                          uint32_t w, uint32_t h) override
    {
        if (!valid()) {
            return NFRU_DP4A_ERROR_NOT_READY;
        }
        if (w != width || h != height) {
            snprintf(err, sizeof(err), "resolution mismatch: context is %ux%u, dispatch %ux%u",
                     width, height, w, h);
            return NFRU_DP4A_ERROR_MODEL_MISMATCH;
        }
        ID3D12GraphicsCommandList* cmd =
            reinterpret_cast<ID3D12GraphicsCommandList*>(commandList);
        if (!cmd) {
            snprintf(err, sizeof(err), "command list handle is null");
            return NFRU_DP4A_ERROR_INVALID_ARGUMENT;
        }
        ID3D12Resource* inRes = reinterpret_cast<ID3D12Resource*>(inputResource);
        ID3D12Resource* outRes = reinterpret_cast<ID3D12Resource*>(outputResource);

        if (!scratch.res) {
            if (!CreateDefault(layout.scratchBytes, scratch)) {
                return NFRU_DP4A_ERROR_OUT_OF_MEMORY;
            }
            ownScratch = true;
        }
        (void)scratchResource;

        /* Same plan as the Vulkan backend -- see nfru_plan.cpp. Nothing about geometry,
         * offsets or ordering is decided here. */
        RecordPlan plan;
        if (!BuildRecordPlan(width, height, layout, w, h, inputOff, outputOff, plan,
                             err, sizeof(err))) {
            return NFRU_DP4A_ERROR_MODEL_MISMATCH;
        }

        ID3D12DescriptorHeap* heaps[1] = { heap };
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(rootSig);

        /*
         * Scratch is written as a UAV from here on. D3D12 would promote it implicitly from
         * COMMON, but an implicit promotion cannot be undone by a later explicit
         * transition, and the copy-out below needs it in COPY_SOURCE -- so the state is
         * made explicit once and tracked from then on.
         */
        if (scratchState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            Transition(cmd, scratch.res, scratchState,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            scratchState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        for (uint32_t i = 0; i < plan.count; ++i) {
            const PlannedDispatch& d = plan.d[i];

            const uint32_t group = ringIndex;
            ringIndex = (ringIndex + 1) % kDescriptorRingGroups;
            const uint32_t base = group * kConvDescriptors;

            ID3D12Resource* sres = d.srcIsHost ? inRes : scratch.res;
            ID3D12Resource* dres = d.dstIsHost ? outRes : scratch.res;
            const uint64_t ssize = d.srcIsHost ? inputBytes : scratch.size;
            const uint64_t dsize = d.dstIsHost ? outputBytes : scratch.size;

            if (d.isConv) {
                /* t0..t4 are the five SRVs, u0 is the one UAV. The root table lays the SRV
                 * range down at table slot 0 and the UAV range at slot 5
                 * (OffsetInDescriptorsFromTableStart = kConvDescriptors - 1), so the
                 * output descriptor MUST be base + 5. Writing it to base + 1 puts a UAV
                 * where the shader expects t1 -- the multiplier pool -- and the kernel's
                 * stores then land in the wrong buffer entirely. */
                MakeSrv(base + 0, sres, d.srcOffBytes, ssize);
                MakeSrv(base + 1, weights.res, d.wOffBytes, weights.size);
                MakeSrv(base + 2, bias.res, d.bcOffBytes, bias.size);
                MakeSrv(base + 3, mult.res, d.mOffBytes, mult.size);
                MakeSrv(base + 4, shift.res, d.sOffBytes, shift.size);
                MakeUav(base + 5, dres, d.dstOffBytes, dsize);

                cmd->SetPipelineState(psoConv);
                cmd->SetComputeRoot32BitConstants(kRootConstants, 14, &d.pc, 0);
                cmd->SetComputeRootDescriptorTable(kRootTable, GpuHandle(base));
                cmd->Dispatch(d.gx, d.gy, d.gz);
            } else {
                /* Support kernels read their source offsets from the push constants, so
                 * both views span the whole bound buffer. */
                MakeSrv(base + 0, sres, d.srcOffBytes, ssize);
                MakeUav(base + 5, dres, d.dstOffBytes, dsize);

                cmd->SetPipelineState(psoSupport);
                cmd->SetComputeRoot32BitConstants(kRootConstants, kPcWords, d.spc, 0);
                cmd->SetComputeRootDescriptorTable(kRootTable, GpuHandle(base));
                cmd->Dispatch(d.groups, 1, 1);
            }
            UavBarrier(cmd, scratch.res);
            if (d.dstIsHost) UavBarrier(cmd, outRes);
        }

        if (plan.copyBytes) {
            /*
             * A copy reads a resource in COPY_SOURCE state. Copying straight out of
             * UNORDERED_ACCESS is not a valid D3D12 use, and it does not fail -- it
             * returns stale or partially written data, which is exactly the kind of
             * result that looks like a kernel bug.
             */
            Transition(cmd, scratch.res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            scratchState = D3D12_RESOURCE_STATE_COPY_SOURCE;
            cmd->CopyBufferRegion(outRes, plan.copyDstOff, scratch.res, plan.copySrcOff,
                                  plan.copyBytes);
            Transition(cmd, scratch.res, D3D12_RESOURCE_STATE_COPY_SOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            scratchState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        return NFRU_DP4A_OK;
    }
};

}  /* namespace */

Device* Dx12CreateDevice(const NfruDp4aCreateInfo& createInfo, const char** outErr)
{
    Dx12Device* d = new Dx12Device();
    if (!d->Init(createInfo)) {
        if (outErr) *outErr = d->lastError();
        d->Destroy();
        delete d;
        return nullptr;
    }
    return d;
}

}  /* namespace nfru */
