/*
 * nfru_dx12.h -- Direct3D 12 dp4a backend.
 */
#ifndef NFRU_DX12_H
#define NFRU_DX12_H

#include "nfru_dp4a.h"
#include "nfru_device.h"

namespace nfru {

/*
 * Builds the DX12 backend from the host's ID3D12Device and ID3D12CommandQueue, passed
 * through NfruDp4aCreateInfo::instance (device) and ::queue (command queue) as
 * uint64_t. Returns null and fills outErr on failure.
 */
Device* Dx12CreateDevice(const NfruDp4aCreateInfo& createInfo, const char** outErr);

}  /* namespace nfru */

#endif /* NFRU_DX12_H */
