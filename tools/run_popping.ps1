<#
.SYNOPSIS
  Phase 6 temporal-popping measurement: default (centre-depth) order vs StopThePop-style
  per-pixel re-sorting windows, against a per-pixel-sorted ground truth.

.DESCRIPTION
  For each scene: a slow orbit (gsplat --orbit-step) rendered as a frame sequence
  (gsplat --sequence) with splat-tiles for each --sort-window (0 = INRIA order), and the
  same camera path rendered by the CPU reference with an EXACT per-pixel sort by ray depth
  (gsplat_ref --sort-window exact --orbit-step s --orbit-frames i) as the ground truth.
  tools/flicker (build/bin/<Config>/flicker.exe) then reports, per configuration:
    PSNR vs GT (static), temporal RMSE (frame-to-frame change not in the GT), pop fraction.
  Writes a markdown table (stdout) and JSON per configuration into $OutDir.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_popping.ps1
  powershell -ExecutionPolicy Bypass -File tools/run_popping.ps1 -Scenes bonsai_30k -Frames 30 -Windows 0,4
#>
param(
    [string]$Config = 'Release',
    [string[]]$Scenes = @('train_7k', 'bonsai_30k'),
    [uint32]$Frames = 60,
    [double]$StepDeg = 0.5,
    [uint32]$Width = 960,
    [uint32]$Height = 540,
    [string[]]$Windows = @('0', '4', '8', '16', '32'),
    [string]$OutDir = 'build/popping'
)

$ErrorActionPreference = 'Stop'
$Scenes = @($Scenes | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Windows = @($Windows | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root "build/bin/$Config"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$step = $StepDeg.ToString($inv)
$rows = @()
foreach ($s in $Scenes) {
    $ply = "data/scenes/$s/point_cloud.ply"
    if (-not (Test-Path $ply)) { Write-Warning "skipping $s"; continue }
    $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
    $common = @('--scene', $ply) + $cam + @('--width', "$Width", '--height', "$Height")
    # Ground truth: exact per-pixel sort on the CPU, one frame at a time.
    $gt = Join-Path $OutDir "${s}_gt"
    for ($i = 0; $i -lt $Frames; $i++) {
        $f = '{0}_{1:D4}.pfm' -f $gt, $i
        if (Test-Path $f) { continue }
        & (Join-Path $bin 'gsplat_ref.exe') @($common + @('--sort-window', 'exact', '--orbit-step', $step,
            '--orbit-frames', "$i", '--pfm', $f)) *> "$gt.log"
        if ($LASTEXITCODE -ne 0) { throw "gsplat_ref failed (frame $i)" }
    }
    foreach ($k in $Windows) {
        $prefix = Join-Path $OutDir "${s}_k$k"
        $args = $common + @('--renderer', 'splat-tiles', '--bounds', 'tight', '--frames', "$Frames", '--orbit-step', $step,
                            '--sort-window', "$k", '--screenshot', "$prefix.png", '--sequence', $prefix)
        & (Join-Path $bin 'gsplat.exe') @args *> "$prefix.log"
        if ($LASTEXITCODE -ne 0) { throw "gsplat failed: $s K=$k (exit $LASTEXITCODE, see $prefix.log)" }
        $json = "$prefix.flicker.json"
        & (Join-Path $bin 'flicker.exe') --test $prefix --gt $gt --gt-ext .pfm --frames $Frames --json $json | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "flicker failed" }
        $j = Get-Content $json -Raw | ConvertFrom-Json
        $rows += [pscustomobject]@{ Scene = $s; K = $k; Psnr = $j.psnr_mean_db; Temporal = $j.temporal_rmse_255
            P95 = $j.temporal_rmse_p95_frame_255; Pops = 100.0 * $j.pop_fraction; Change = $j.change_rms_255 }
    }
}
Write-Host ""
Write-Host "Temporal popping vs exact per-pixel sort (CPU), ${Width}x${Height}, $Frames frames, orbit $StepDeg deg/frame"
Write-Host ""
Write-Host "| Scene | sort window K (0 = INRIA order) | PSNR vs GT (dB, mean) | temporal RMSE (/255) | worst-5% frame (/255) | pops > 8/255 (% px) | change RMS (/255) |"
Write-Host "|---|---|---|---|---|---|---|"
foreach ($r in $rows) {
    Write-Host ([string]::Format($inv, "| {0} | {1} | {2:F2} | {3:F3} | {4:F3} | {5:F4} | {6:F3} |", $r.Scene, $r.K, $r.Psnr,
        $r.Temporal, $r.P95, $r.Pops, $r.Change))
}
exit 0
