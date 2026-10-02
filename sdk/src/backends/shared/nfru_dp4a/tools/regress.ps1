# =============================================================================
# NFRU dp4a -- full hardware regression, both backends
# =============================================================================
# Runs every check that has actually caught a bug in this port:
#
#   1. the whole 19-dispatch graph against the CPU reference, byte for byte
#   2. each of the 18 recorded stages against the CPU stage ladder
#   3. each of the 16 convolutions alone against its raw int32 accumulator
#      reference (this is the check that localises a fault to one layer)
#   4. Vulkan's dumped output vs DX12's, byte for byte, so "both backends are
#      bit-identical" is measured rather than assumed
#
# Usage:  pwsh -File tools/regress.ps1            # both backends
#         pwsh -File tools/regress.ps1 -Backend dx12
param(
    [string]$Backend = "both",
    [string]$Packed  = "port/nfru_qat"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Absolute paths: PowerShell does not resolve a relative native-executable path that uses
# forward slashes, and it fails silently enough to look like the harness producing no
# output at all.
$vkExe = Join-Path $root "fru-port\bin\Release\nfru_test.exe"
$dxExe = Join-Path $root "fru-port\bin\Release\nfru_test_dx12.exe"
$inFile = "$Packed/golden_input_codes.bin"
$refOut = "$Packed/reference_output_codes.bin"

# Every debug switch has to be cleared or a stale one silently changes what is measured.
Remove-Item Env:NFRU_DP4A_TRACE_SUPPORT, Env:NFRU_DP4A_TRACE_PC, Env:NFRU_DP4A_TRACE_GEOM,
            Env:NFRU_DP4A_TRACE_BIND, Env:NFRU_DP4A_STOP_AFTER, Env:NFRU_DP4A_STOP_TO_SCRATCH,
            Env:NFRU_DP4A_DUMP_ACC, Env:NFRU_DP4A_DUMP_MARK, Env:NFRU_DP4A_LAYER_ONLY,
            Env:NFRU_DP4A_DUMP, Env:NFRU_DP4A_INPUT_FOR_OP, Env:NFRU_DP4A_RUN_TWICE `
            -ErrorAction SilentlyContinue

$failures = @()

function Report([string]$backend, [string]$check, [bool]$ok, [string]$detail) {
    $tag = if ($ok) { "PASS" } else { "FAIL" }
    Write-Host ("{0,-6} {1,-7} {2,-34} {3}" -f $tag, $backend, $check, $detail)
    if (-not $ok) { $script:failures += "$backend/$check" }
}

function RunGraph([string]$backend, [string]$exe) {
    $env:NFRU_DP4A_BACKEND = $backend
    $o = & $exe $inFile $refOut 480 270 2>&1
    $m = ($o | Select-String "mismatches")
    $text = if ($m) { $m.ToString().Trim() } else { "no output" }
    Report $backend "whole graph (518400 bytes)" ($text -match ":\s*0\s*/") $text
}

function RunLadder([string]$backend, [string]$exe) {
    $env:NFRU_DP4A_BACKEND = $backend
    $env:NFRU_DP4A_STOP_TO_SCRATCH = "1"
    $ok = 0; $bad = @()
    foreach ($n in 1..18) {
        $env:NFRU_DP4A_STOP_AFTER = "$n"
        $exp = "{0}/stage_{1:d2}.bin" -f $Packed, $n
        $o = & $exe $inFile $exp 480 270 2>&1
        $m = ($o | Select-String "mismatches")
        if ($m -and $m.ToString() -match ":\s*0\s*/") { $ok++ } else { $bad += $n }
    }
    Remove-Item Env:NFRU_DP4A_STOP_AFTER, Env:NFRU_DP4A_STOP_TO_SCRATCH
    Report $backend "stage ladder (18 dispatches)" ($ok -eq 18) "$ok/18 exact; failing: $($bad -join ',')"
}

function RunLayers([string]$backend, [string]$exe) {
    $env:NFRU_DP4A_BACKEND = $backend
    $env:NFRU_DP4A_DUMP_ACC = "1"
    $cases = @(
        "0:$Packed/golden_input_codes.bin:480:270", "1:$Packed/stage_01.bin:480:270",
        "2:$Packed/stage_02.bin:480:270", "3:$Packed/stage_03.bin:480:270",
        "4:$Packed/stage_04.bin:480:270", "5:$Packed/stage_05.bin:240:135",
        "6:$Packed/stage_06.bin:240:135", "7:$Packed/stage_07.bin:240:135",
        "8:$Packed/stage_08.bin:240:135", "9:$Packed/stage_09.bin:240:135",
        "10:$Packed/stage_10.bin:240:135", "11:$Packed/stage_11.bin:240:135",
        "13:$Packed/stage_13.bin:240:135", "15:$Packed/stage_15.bin:480:270",
        "17:$Packed/stage_17.bin:480:270", "18:$Packed/stage_18.bin:480:270")
    $a = @(); foreach ($c in $cases) { $a += "--case"; $a += $c }
    $o = & python tools/verify_layers.py @a --exe $exe 2>&1
    $lines = $o | Select-String "mismatches"
    $ok = ($lines | Where-Object { $_.ToString() -match ":\s*0\s*/" }).Count
    $bad = ($lines | Where-Object { $_.ToString() -notmatch ":\s*0\s*/" }).Count
    Remove-Item Env:NFRU_DP4A_DUMP_ACC
    Report $backend "16 convolutions in isolation" ($ok -eq 16 -and $bad -eq 0) `
        "$ok/16 bit-exact; $bad failing"
}

Write-Host "=== NFRU dp4a hardware regression ==="
if ($Backend -eq "both" -or $Backend -eq "vulkan") {
    RunGraph "vulkan" $vkExe
    RunLadder "vulkan" $vkExe
    RunLayers "vulkan" $vkExe
}
if ($Backend -eq "both" -or $Backend -eq "dx12") {
    RunGraph "dx12" $dxExe
    RunLadder "dx12" $dxExe
    RunLayers "dx12" $dxExe
}

if ($Backend -eq "both") {
    # Direct evidence for "the two backends agree": dump each one's output and diff bytes.
    # The dumps are removed first so that a harness that fails to run cannot leave last
    # run's files behind and be reported as agreement.
    Remove-Item "$Packed/_regress_vk.bin", "$Packed/_regress_dx.bin" -ErrorAction SilentlyContinue
    $env:NFRU_DP4A_DUMP = "$Packed/_regress_vk.bin"; $env:NFRU_DP4A_BACKEND = "vulkan"
    & $vkExe $inFile $refOut 480 270 > $null 2>&1
    $env:NFRU_DP4A_DUMP = "$Packed/_regress_dx.bin"; $env:NFRU_DP4A_BACKEND = "dx12"
    & $dxExe $inFile $refOut 480 270 > $null 2>&1
    Remove-Item Env:NFRU_DP4A_DUMP
    if (-not (Test-Path "$Packed/_regress_vk.bin") -or -not (Test-Path "$Packed/_regress_dx.bin")) {
        Report "cross" "vulkan == dx12 == cpu reference" $false `
            "a harness did not produce a dump; cannot compare"
    } else {
        $same = & python -c @"
import numpy as np
a=np.fromfile(r'$Packed/_regress_vk.bin',dtype=np.uint8)
b=np.fromfile(r'$Packed/_regress_dx.bin',dtype=np.uint8)
r=np.fromfile(r'$refOut',dtype=np.uint8)
print('%d %d %d' % (int((a!=b).sum()), int((a!=r).sum()), int((b!=r).sum())))
"@
        $f = $same.Trim().Split(' ')
        Report "cross" "vulkan == dx12 == cpu reference" `
            ($f[0] -eq "0" -and $f[1] -eq "0" -and $f[2] -eq "0") `
            "vk-vs-dx differing bytes $($f[0]); vk-vs-cpu $($f[1]); dx-vs-cpu $($f[2])"
    }
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED"
    exit 0
}
Write-Host "FAILURES: $($failures -join ', ')"
exit 1
