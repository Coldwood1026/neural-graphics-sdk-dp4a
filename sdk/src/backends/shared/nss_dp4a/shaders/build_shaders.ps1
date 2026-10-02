<#
.SYNOPSIS
    Rebuild the NSS D3D12 shader binaries and the header they are embedded into.

.DESCRIPTION
    The Vulkan half of this module does not use this script. Its SPIR-V is baked by the
    upstream `tools/bake_model.py` into `generated/nss_spirv_embed.h`, from GLSL sources
    that live in the separate nss_kernel repository. This script exists for the D3D12
    half, whose kernels are HLSL ports of those GLSL sources and whose DXIL is embedded
    into `generated/nss_shaders_dxil.h`.

    Layout:

        shaders/glsl/*.comp     the reference kernels (upstream)
        shaders/*.hlsl          the D3D12 ports of those kernels
        shaders/dxil/*.dxil     the compiled output
        generated/nss_shaders_dxil.h    the embedded output

    Unlike the NFRU module, every input here reproduces its checked-in output. `dxc` on
    the checked-in HLSL with the flags below gives back the checked-in .dxil byte for
    byte. That is worth stating explicitly because in the NFRU module the opposite is
    true for the SPIR-V half, and a reader who has seen that note should not assume it
    applies here.

.PARAMETER Dxc
    Path to dxc.exe. Defaults to the one in the newest Windows SDK found.
#>
[CmdletBinding()]
param(
    [string]$Dxc
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = Split-Path -Parent $here          # .../nss_dp4a/shaders -> .../nss_dp4a

if (-not $Dxc) {
    $cand = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxc.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
    if ($cand) { $Dxc = $cand.FullName }
}
if (-not $Dxc) { throw "dxc.exe not found; pass -Dxc <path>" }
Write-Host "dxc: $Dxc"

function Invoke-Step([string]$what, [scriptblock]$body) {
    Write-Host "  $what" -NoNewline
    & $body
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED (exit $LASTEXITCODE)"; throw "$what failed" }
    Write-Host "  ok"
}

# The kernel -> (file, entry point, profile, extra defines) table.
#
# conv_rq needs cs_6_4: dot4add_i8packed is a Shader Model 6.4 intrinsic. The other two
# are plain integer/buffer work and compile at cs_6_0, which keeps the requirement for
# those kernels as low as it can be.
#
# conv_rq_2x2 is the same HLSL built with -D CONV_2X2=1: one thread produces a 2x2 block
# of output pixels, which is worth it only on the layers with very few output channels.
# It is built and embedded, but the D3D12 host currently dispatches every convolution
# through the base kernel -- the variant is bit-identical by construction, so this is a
# missed optimisation rather than a correctness gap. See the module README.
$kernels = @(
    @{ out = "conv_rq.dxil";     src = "conv_rq.hlsl";     entry = "CSConv";       profile = "cs_6_4"; defines = @() },
    @{ out = "conv_rq_2x2.dxil"; src = "conv_rq.hlsl";     entry = "CSConv";       profile = "cs_6_4"; defines = @("CONV_2X2=1") },
    @{ out = "resize2x.dxil";    src = "resize2x.hlsl";    entry = "CSResize2x";   profile = "cs_6_0"; defines = @() },
    @{ out = "concat_copy.dxil"; src = "concat_copy.hlsl"; entry = "CSConcatCopy"; profile = "cs_6_0"; defines = @() }
)

Write-Host "DXIL:"
foreach ($k in $kernels) {
    $defs = @()
    foreach ($d in $k.defines) { $defs += "-D"; $defs += $d }
    $label = "$($k.out) ($($k.profile) $($k.entry)$(if($k.defines.Count){" -D $($k.defines -join ' ')"}))"
    Invoke-Step $label {
        & $Dxc -T $k.profile -E $k.entry @defs -Fo "$here\dxil\$($k.out)" "$here\$($k.src)"
    }
    # Disassembly alongside the binary: it is the only way to confirm after the fact that
    # a kernel still contains the instruction that matters, e.g. that the fused form of
    # dot4add_i8packed survived.
    & $Dxc -T $k.profile -E $k.entry @defs -Fc "$here\dxil\$([IO.Path]::GetFileNameWithoutExtension($k.out)).asm" "$here\$($k.src)" | Out-Null
}

Write-Host "embedding:"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }
Invoke-Step "generated/nss_shaders_dxil.h" {
    & $py "$root\tools\embed_dxil.py" --out "$root\generated\nss_shaders_dxil.h" --shaders "$here\dxil"
}

Write-Host ""
Write-Host "Done. Rebuild the SDK, then confirm the DXIL->D3D12 path on hardware."
