<#
.SYNOPSIS
  Phase 6 final consolidation: the whole optimization journey in one table.

.DESCRIPTION
  For each scene (utsuho_plush, train_7k, bonsai_30k) at 1920x1080 and each configuration
  below (Phase 2 baseline .. Phase 6 defaults and recommended variants; every flag spelled
  out, so the rows stay reproducible when defaults change):
    - benchmark: 300 frames on the 360-degree orbit from training camera 0 (utsuho:
      auto-framed): GPU frame ms (timestamps), FPS (CPU frame time), registered GPU memory
      and VMA device-local process usage;
    - quality: a 1080p screenshot of the start view vs the CPU reference (gsplat_ref, INRIA
      forward pass, uncompressed scene) -> PSNR / max error.
  Writes docs/notes/bench/final_summary.md (+ the benchmark JSONs, label final_<config>).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_final_report.ps1
  powershell -ExecutionPolicy Bypass -File tools/run_final_report.ps1 -Scenes bonsai_30k -Frames 120
#>
param(
    [string]$Config = 'Release',
    [string[]]$Scenes = @('utsuho_plush', 'train_7k', 'bonsai_30k'),
    [uint32]$Width = 1920,
    [uint32]$Height = 1080,
    [uint32]$Frames = 300,
    [string]$BenchDir = 'docs/notes/bench',
    [string]$OutDir = 'build/final_report',
    [string]$Markdown = 'docs/notes/bench/final_summary.md'
)
$ErrorActionPreference = 'Stop'
$Scenes = @($Scenes | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root "build/bin/$Config"
foreach ($exe in 'gsplat.exe', 'gsplat_ref.exe', 'imgdiff.exe') {
    if (-not (Test-Path (Join-Path $bin $exe))) { throw "Missing $bin/$exe; build the $Config configuration first." }
}
New-Item -ItemType Directory -Force $OutDir, $BenchDir | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture

# Phase 4/5 rows pin --bounds rect (and their list mode); Phase 6 rows use the new defaults.
$configs = @(
    @{ Id = 'p2_splat'; Phase = 'Phase 2'; Name = 'splat: CPU sort + instanced quads (baseline)'
       Args = @('--renderer', 'splat', '--bounds', 'rect') },
    @{ Id = 'p3_gpusort'; Phase = 'Phase 3'; Name = 'splat-gpusort: GPU cull + radix sort + indirect draw'
       Args = @('--renderer', 'splat-gpusort', '--bounds', 'rect') },
    @{ Id = 'p4_tiles'; Phase = 'Phase 4'; Name = 'splat-tiles: compute tile rasterizer'
       Args = @('--renderer', 'splat-tiles', '--bounds', 'rect', '--tile-list', 'fixed', '--render-splat', 'full') },
    @{ Id = 'p5_tiles_f16_p32_adaptive'; Phase = 'Phase 5'; Name = 'splat-tiles + f16 scene + p32 + adaptive list (recommended)'
       Args = @('--renderer', 'splat-tiles', '--bounds', 'rect', '--tile-list', 'adaptive', '--render-splat', 'p32', '--scene-format', 'f16') },
    @{ Id = 'p6_gpusort_tight'; Phase = 'Phase 6'; Name = 'splat-gpusort + tight bounds (raster, exact)'
       Args = @('--renderer', 'splat-gpusort', '--bounds', 'tight') },
    @{ Id = 'p6_default'; Phase = 'Phase 6'; Name = 'NEW DEFAULT: splat-tiles + tight bounds + adaptive list (exact)'
       Args = @() },
    @{ Id = 'p6_f16_vsh'; Phase = 'Phase 6'; Name = 'default + f16 scene + per-gaussian SH 0.25/255 (near-lossless, memory)'
       Args = @('--scene-format', 'f16', '--sh-threshold', '0.25') },
    @{ Id = 'p6_stp4'; Phase = 'Phase 6'; Name = 'default + StopThePop window 4 (quality option, not INRIA)'
       Args = @('--sort-window', '4') }
)

function Get-ImgDiff([string]$a, [string]$b) {
    $j = Join-Path $OutDir 'diff.json'
    & (Join-Path $bin 'imgdiff.exe') $a $b --json $j | Out-Null
    return (Get-Content $j -Raw | ConvertFrom-Json)
}

$rows = @()
foreach ($s in $Scenes) {
    $ply = "data/scenes/$s/point_cloud.ply"
    if (-not (Test-Path $ply)) { Write-Warning "skipping $s"; continue }
    $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
    $common = @('--scene', $ply) + $cam + @('--width', "$Width", '--height', "$Height")
    $refPfm = Join-Path $OutDir "${s}_ref.pfm"
    & (Join-Path $bin 'gsplat_ref.exe') @($common + @('--pfm', $refPfm)) *> "$refPfm.log"
    if ($LASTEXITCODE -ne 0) { throw "gsplat_ref failed on $s" }
    foreach ($c in $configs) {
        $png = Join-Path $OutDir "${s}_$($c.Id).png"
        & (Join-Path $bin 'gsplat.exe') @($common + $c.Args + @('--screenshot', $png)) *> "$png.log"
        if ($LASTEXITCODE -ne 0) { throw "screenshot failed: $s $($c.Id) (exit $LASTEXITCODE)" }
        $d = Get-ImgDiff $png $refPfm
        $json = Join-Path $BenchDir "final_$($c.Id)_$s.json"
        Write-Host "gsplat $s $($c.Id)"
        & (Join-Path $bin 'gsplat.exe') @($common + $c.Args + @('--frames', "$Frames", '--benchmark', $json)) *> "$json.log"
        if ($LASTEXITCODE -ne 0) { throw "benchmark failed: $s $($c.Id) (exit $LASTEXITCODE)" }
        Remove-Item "$json.log"
        $j = Get-Content $json -Raw | ConvertFrom-Json
        $usage = ($j.gpu_memory.vma_heaps | Where-Object { $_.device_local } | Measure-Object -Property process_usage_mb -Sum).Sum
        $rows += [pscustomobject]@{ Scene = $s; Phase = $c.Phase; Name = $c.Name; Id = $c.Id; Renderer = $j.renderer
            Frame = $j.gpu_ms.'frame (total)'.avg; Cpu = $j.cpu_frame_ms.avg; Fps = $j.fps_avg
            Mem = $j.gpu_memory.total_registered_mb; Usage = $usage; Psnr = $d.psnr_db; Max = 255 * $d.max_abs
            Args = ($c.Args -join ' ') }
    }
}

function F($v, $fmt) { ([double]$v).ToString($fmt, $inv) }
$md = @()
$md += '# Final summary: the optimization journey (Phase 2 -> Phase 6)'
$md += ''
$md += "Generated by ``tools/run_final_report.ps1`` ($Config, ${Width}x${Height}, $Frames-frame 360-degree orbit from training"
$md += 'camera 0 (utsuho: auto-framed), RTX 4060 Laptop). GPU frame = timestamp of the whole frame (renderer + blit to'
$md += 'the swapchain); FPS = 1000 / mean CPU frame time (IMMEDIATE present, 2 frames in flight); memory = GPU memory'
$md += 'registered by the app (scene + renderer + targets) and the VMA device-local process usage; PSNR = 1080p start view'
$md += 'vs the CPU reference (INRIA forward pass, uncompressed scene; 58.9 dB is the 8-bit output floor).'
$md += ''
foreach ($s in $Scenes) {
    $sr = @($rows | Where-Object { $_.Scene -eq $s })
    if ($sr.Count -eq 0) { continue }
    $base = $sr[0].Frame
    $md += "## $s"
    $md += ''
    $md += '| phase | configuration | GPU frame (ms) | speed-up vs Phase 2 | FPS | GPU memory registered / process usage (MB) | PSNR vs reference (dB) | max err (/255) |'
    $md += '|---|---|---|---|---|---|---|---|'
    foreach ($r in $sr) {
        $md += ("| {0} | {1} | {2} | {3}x | {4} | {5} / {6} | {7} | {8} |" -f $r.Phase, $r.Name, (F $r.Frame 'F2'),
            (F ($base / $r.Frame) 'F2'), (F $r.Fps 'F0'), (F $r.Mem 'F0'), (F $r.Usage 'F0'), (F $r.Psnr 'F2'), (F $r.Max 'F1'))
    }
    $md += ''
}
$md += '## Flags of each configuration'
$md += ''
$md += '| id | flags (on top of --scene, camera, size) |'
$md += '|---|---|'
foreach ($c in $configs) { $md += ("| {0} | ``{1}`` |" -f $c.Id, $(if ($c.Args.Count) { $c.Args -join ' ' } else { '(defaults)' })) }
$md | Set-Content -Encoding utf8 $Markdown
$md | ForEach-Object { Write-Host $_ }
