<#
.SYNOPSIS
  GPU-vs-CPU-reference correctness checks (Phase 2+).

.DESCRIPTION
  For each test scene: renders a screenshot with gsplat (the GPU renderer under test),
  renders the same camera with the CPU reference gsplat_ref (INRIA forward pass, and a
  second time without early termination), diffs them with imgdiff and prints a PSNR
  table (markdown). Fails (exit 1) if any GPU image is below its scene's threshold
  against the INRIA reference, or if any program fails or reports validation messages.

  Thresholds (dB, against the INRIA reference with early termination, 8-bit GPU output):
    synthetic 50, real 50. See docs/notes/phase2.md "Reference checks" for the
    measurements behind them: the default GPU path reaches 58-59 dB on real scenes,
    the 8-bit quantization of the GPU output alone limits PSNR to ~59 dB, a uniform
    1/255 error everywhere would be 48.1 dB, and the RGBA8 blend target (a real
    precision bug) gives 36-44 dB.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_reference_checks.ps1
  powershell -ExecutionPolicy Bypass -File tools/run_reference_checks.ps1 -GpuArgs '--splat-format','rgba8'
#>
param(
    [string]$Config = 'Release',
    [string]$Renderer = 'splat',
    [uint32]$Width = 960,
    [uint32]$Height = 540,
    [string]$OutDir = 'build/reference_checks',
    [double]$SyntheticMinPsnr = 50.0,
    [double]$RealMinPsnr = 50.0,
    [string[]]$GpuArgs = @(),     # extra gsplat flags, e.g. '--splat-format','rgba8'
    [string]$Bound = 'tile',      # coverage bound for BOTH the GPU (--splat-bound) and the reference (--bound)
    [string]$Bounds = '',         # Phase 6 binning bounds for BOTH (--bounds rect|tight|cutoff-only); '' = defaults
    [switch]$SkipReal
)

$ErrorActionPreference = 'Stop'
# 'powershell -File' passes an array argument as one comma-joined string: split it again.
$GpuArgs = @($GpuArgs | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root "build/bin/$Config"
foreach ($exe in 'gsplat.exe', 'gsplat_ref.exe', 'imgdiff.exe') {
    if (-not (Test-Path (Join-Path $bin $exe))) { throw "Missing $bin/$exe; build the $Config configuration first." }
}
New-Item -ItemType Directory -Force $OutDir | Out-Null

# name, scene, extra camera flags, kind. Synthetic scenes use a fixed oblique camera
# (auto-framing a 1-splat scene would put the camera inside it).
$synthCam = @('--camera', 'tools/cameras/synthetic.json')
$cases = @(
    @{ Name = 'single_sh0';   Scene = 'data/synthetic/single_sh0.ply';   Cam = $synthCam; Kind = 'synthetic' },
    @{ Name = 'single_sh3';   Scene = 'data/synthetic/single_sh3.ply';   Cam = $synthCam; Kind = 'synthetic' },
    @{ Name = 'overlap_sh0';  Scene = 'data/synthetic/overlap_sh0.ply';  Cam = $synthCam; Kind = 'synthetic' },
    @{ Name = 'grid_512_sh3'; Scene = 'data/synthetic/grid_512_sh3.ply'; Cam = $synthCam; Kind = 'synthetic' },
    @{ Name = 'shdir_sh3';    Scene = 'data/synthetic/shdir_sh3.ply';    Cam = $synthCam; Kind = 'synthetic' }
)
if (-not $SkipReal) {
    foreach ($s in 'utsuho_plush', 'train_7k', 'bonsai_30k') {
        $ply = "data/scenes/$s/point_cloud.ply"
        if (-not (Test-Path $ply)) { Write-Warning "skipping $s (not found)"; continue }
        # Training camera 0 where cameras.json exists; utsuho_plush has none (auto-framed).
        $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
        $cases += @{ Name = $s; Scene = $ply; Cam = $cam; Kind = 'real' }
    }
}

function Invoke-Checked([string]$exe, [string[]]$arguments, [string]$log) {
    & $exe @arguments *> $log
    return $LASTEXITCODE
}

function Get-Diff([string]$test, [string]$ref, [string]$json, [string]$heatmap) {
    $a = @($test, $ref, '--json', $json)
    if ($heatmap) { $a += @('--heatmap', $heatmap) }
    & (Join-Path $bin 'imgdiff.exe') @a | Out-Null
    if ($LASTEXITCODE -gt 1) { throw "imgdiff failed on $test vs $ref" }
    return Get-Content $json -Raw | ConvertFrom-Json
}

$rows = @()
$failed = $false
foreach ($c in $cases) {
    $n = $c.Name
    $gpuPng = Join-Path $OutDir "${n}_gpu.png"
    $refPfm = Join-Path $OutDir "${n}_ref.pfm"
    $refNoStopPfm = Join-Path $OutDir "${n}_ref_nostop.pfm"
    $size = @('--width', "$Width", '--height', "$Height")
    $common = @('--scene', $c.Scene) + $c.Cam + $size
    if ($Bounds) { $common += @('--bounds', $Bounds) }

    $gpuCmd = $common + @('--renderer', $Renderer, '--splat-bound', $Bound, '--screenshot', $gpuPng) + $GpuArgs
    $gpuCode = Invoke-Checked (Join-Path $bin 'gsplat.exe') $gpuCmd (Join-Path $OutDir "${n}_gpu.log")
    $refCode = Invoke-Checked (Join-Path $bin 'gsplat_ref.exe') ($common + @('--bound', $Bound, '--out', (Join-Path $OutDir "${n}_ref.png"), '--pfm', $refPfm, '--stats-json', (Join-Path $OutDir "${n}_ref_stats.json"))) (Join-Path $OutDir "${n}_ref.log")
    $ref2Code = Invoke-Checked (Join-Path $bin 'gsplat_ref.exe') ($common + @('--bound', $Bound, '--no-early-stop', '--pfm', $refNoStopPfm)) (Join-Path $OutDir "${n}_ref_nostop.log")
    if ($gpuCode -ne 0 -or $refCode -ne 0 -or $ref2Code -ne 0) {
        Write-Host "FAIL ${n}: exit codes gsplat=$gpuCode gsplat_ref=$refCode/$ref2Code (logs in $OutDir)"
        $failed = $true
        continue
    }

    $d1 = Get-Diff $gpuPng $refPfm (Join-Path $OutDir "${n}_diff.json") (Join-Path $OutDir "${n}_heatmap.png")
    $d2 = Get-Diff $gpuPng $refNoStopPfm (Join-Path $OutDir "${n}_diff_nostop.json") $null
    $d3 = Get-Diff $refNoStopPfm $refPfm (Join-Path $OutDir "${n}_diff_stop_effect.json") $null
    $min = if ($c.Kind -eq 'real') { $RealMinPsnr } else { $SyntheticMinPsnr }
    $ok = $d1.psnr_db -ge $min
    if (-not $ok) { $failed = $true }
    $rows += [pscustomobject]@{
        Scene = $n; Kind = $c.Kind; Psnr = $d1.psnr_db; PsnrNoStop = $d2.psnr_db; StopEffect = $d3.psnr_db
        MaxAbs = $d1.max_abs * 255; MeanAbs = $d1.mean_abs * 255; Over = $d1.pixels_over_threshold_pct
        Min = $min; Result = $(if ($ok) { 'PASS' } else { 'FAIL' })
    }
}

$inv = [System.Globalization.CultureInfo]::InvariantCulture
function Format-Db([double]$v) { if ($v -ge 999) { 'inf' } else { $v.ToString('F2', $inv) } }
$extra = if ($GpuArgs.Count) { ' ' + ($GpuArgs -join ' ') } else { '' }
Write-Host ""
$bnd = if ($Bounds) { ", bounds $Bounds" } else { '' }
Write-Host "GPU '$Renderer' (bound $Bound$bnd$extra) vs CPU reference, ${Width}x${Height}, $Config"
Write-Host ""
Write-Host "| Scene | PSNR vs INRIA ref (dB) | PSNR vs ref w/o early stop (dB) | early-stop effect: ref vs ref w/o stop (dB) | max abs (/255) | mean abs (/255) | px > 2/255 (%) | min (dB) | result |"
Write-Host "|---|---|---|---|---|---|---|---|---|"
foreach ($r in $rows) {
    Write-Host ([string]::Format($inv, "| {0} | {1} | {2} | {3} | {4:F1} | {5:F3} | {6:F3} | {7:F0} | {8} |", $r.Scene, (Format-Db $r.Psnr),
        (Format-Db $r.PsnrNoStop), (Format-Db $r.StopEffect), $r.MaxAbs, $r.MeanAbs, $r.Over, $r.Min, $r.Result))
}
Write-Host ""
if ($failed) { Write-Host "REFERENCE CHECKS FAILED (outputs in $OutDir)"; exit 1 }
Write-Host "All reference checks passed (outputs in $OutDir)"
exit 0
