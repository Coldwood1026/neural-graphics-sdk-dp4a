<#
.SYNOPSIS
    Rebuild the NFRU dp4a shader binaries and the headers they are embedded into.

.DESCRIPTION
    There are three checked-in artefacts and this script produces all three:

        shaders/dxil/nfru_conv_rq.dxil  shade  }  embedded into
        shaders/dxil/nfru_support.dxil  shade  }  nfru_shaders_dxil.h
        shaders/nfru_conv_rq.spv                 }  embedded into
        shaders/nfru_support.spv                 }  nfru_shaders_spv.h

    They are checked in so that a build of the SDK needs no shader compiler at all: the
    SDK's own shader pipeline (FidelityFX_SC / glslang / dxc) is never invoked for this
    module. This script exists for when the kernels are edited.

    READ THIS BEFORE REGENERATING THE SPIR-V
    ----------------------------------------
    The two halves are NOT equally reproducible, and the difference is not cosmetic:

      DXIL (D3D12)   IS reproducible. dxc on the checked-in .hlsl files with the flags
                     below reproduces the checked-in .dxil byte for byte. Verified.

      SPIR-V (Vulkan) IS NOT reproducible from anything currently in this tree. The
                     checked-in .spv blobs are the ones that were validated bit-exact
                     against the CPU reference on hardware, but recompiling either the
                     .comp sources or the .hlsl sources produces a DIFFERENT, LARGER
                     blob (glslang version and/or a since-edited source). See
                     "KNOWN GAP" in README.md.

    So: regenerating the DXIL is safe and verifiable. Regenerating the SPIR-V will change
    the shipped binary, and the hardware regression must be re-run and must still show
    zero mismatching bytes before that change is trusted. Do not regenerate the SPIR-V
    casually.

.PARAMETER SpvOnly
    Rebuild only the SPIR-V.

.PARAMETER DxilOnly
    Rebuild only the DXIL.

.PARAMETER Dxc
    Path to dxc.exe. Defaults to the one in the newest Windows SDK found.

.PARAMETER Glslang
    Path to glslangValidator.exe. Defaults to the SDK's vendored copy.
#>
[CmdletBinding()]
param(
    [switch]$SpvOnly,
    [switch]$DxilOnly,
    [string]$Dxc,
    [string]$Glslang
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = Split-Path -Parent $here          # .../nfru_dp4a/shaders -> .../nfru_dp4a

# ---------------------------------------------------------------- tool discovery
if (-not $Dxc) {
    $cand = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxc.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
    if ($cand) { $Dxc = $cand.FullName }
}
if (-not $Glslang) {
    # The SDK vendors glslang for its own shader pipeline.
    $cand = Join-Path $root '..\..\..\tools\binary_store\glslangValidator.exe'
    if (Test-Path $cand) { $Glslang = (Resolve-Path $cand).Path }
}

Write-Host "dxc     : $(if ($Dxc) { $Dxc } else { '(not found)' })"
Write-Host "glslang : $(if ($Glslang) { $Glslang } else { '(not found)' })"

function Invoke-Step([string]$what, [scriptblock]$body) {
    Write-Host "  $what" -NoNewline
    & $body
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED (exit $LASTEXITCODE)"; throw "$what failed" }
    Write-Host "  ok"
}

# ---------------------------------------------------------------- DXIL  (D3D12)
if (-not $SpvOnly) {
    if (-not $Dxc) { throw "dxc.exe not found; pass -Dxc <path>" }
    Write-Host "DXIL:"
    # cs_6_4 for the convolution (it uses dot4add_i8packed, which needs SM 6.4) and
    # cs_6_0 for the resize/concat support kernel.
    Invoke-Step "nfru_conv_rq.dxil (cs_6_4 CSConv)" {
        & $Dxc -T cs_6_4 -E CSConv -Fo "$here\dxil\nfru_conv_rq.dxil" "$here\nfru_conv_rq.hlsl"
    }
    Invoke-Step "nfru_support.dxil (cs_6_0 CSSupport)" {
        & $Dxc -T cs_6_0 -E CSSupport -Fo "$here\dxil\nfru_support.dxil" "$here\nfru_support.hlsl"
    }
}

# ---------------------------------------------------------------- SPIR-V (Vulkan)
if (-not $DxilOnly) {
    Write-Host "SPIR-V:"
    Write-Warning @"
Regenerating the SPIR-V will NOT reproduce the checked-in blobs; it produces larger ones.
The checked-in blobs are the validated ones. Re-run the hardware regression afterwards
(see tools/regress.ps1) and require zero mismatching bytes before trusting the change.
"@

    # Preferred route: dxc with SPIR-V codegen (the dxc shipped in the Vulkan SDK has it;
    # the Windows SDK's does not and fails with "SPIR-V CodeGen not available").
    $dxcHasSpirv = $false
    if ($Dxc) {
        $probe = "$env:TEMP\nfru_spirv_probe.spv"
        & $Dxc -T cs_6_4 -E CSConv -spirv -Fo $probe "$here\nfru_conv_rq.hlsl" 2>$null
        $dxcHasSpirv = ($LASTEXITCODE -eq 0 -and (Test-Path $probe))
        Remove-Item -Force $probe -ErrorAction SilentlyContinue
    }

    if ($dxcHasSpirv) {
        # One source of truth: the same HLSL that produces the DXIL.
        Invoke-Step "nfru_conv_rq.spv (dxc -spirv from HLSL)" {
            & $Dxc -T cs_6_4 -E CSConv -spirv -fvk-use-gl-layout -fspv-target-env=vulkan1.3 `
                   -Fo "$here\nfru_conv_rq.spv" "$here\nfru_conv_rq.hlsl"
        }
        Invoke-Step "nfru_support.spv (dxc -spirv from HLSL)" {
            & $Dxc -T cs_6_0 -E CSSupport -spirv -fvk-use-gl-layout -fspv-target-env=vulkan1.3 `
                   -Fo "$here\nfru_support.spv" "$here\nfru_support.hlsl"
        }
    }
    elseif ($Glslang) {
        # Fallback: the GLSL sources. Equivalent kernels, different blob.
        Invoke-Step "nfru_conv_rq.spv (glslang from .comp)" {
            & $Glslang -V --target-env vulkan1.3 -o "$here\nfru_conv_rq.spv" "$here\nfru_conv_rq.comp"
        }
        Invoke-Step "nfru_support.spv (glslang from .comp)" {
            & $Glslang -V --target-env vulkan1.3 -o "$here\nfru_support.spv" "$here\nfru_support.comp"
        }
    }
    else {
        throw "No SPIR-V capable compiler found. Pass -Dxc <dxc with SPIR-V codegen> or -Glslang <glslangValidator>."
    }
}

# ---------------------------------------------------------------- re-embed
Write-Host "embedding:"
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }
Invoke-Step "nfru_shaders_dxil.h" {
    & $py "$root\tools\embed_dxil.py" --conv "$here\dxil\nfru_conv_rq.dxil" `
        --support "$here\dxil\nfru_support.dxil" --out "$root\nfru_shaders_dxil.h"
}
Invoke-Step "nfru_shaders_spv.h" {
    & $py "$root\tools\embed_spv.py" --conv "$here\nfru_conv_rq.spv" `
        --support "$here\nfru_support.spv" --out "$root\nfru_shaders_spv.h"
}

Write-Host ""
Write-Host "Done. Rebuild the SDK, then re-run tools/regress.ps1 before trusting the result."
