<#
.SYNOPSIS
  Benchmark protocol shared by all phases.

.DESCRIPTION
  For each scene (utsuho_plush, train_7k, bonsai_30k) and each requested renderer, runs
  the Release build of gsplat with --benchmark at a fixed resolution and writes
  docs/notes/bench/<Label>_<scene>_<renderer>.json. Then prints one markdown table per
  renderer with every GPU pass (timestamp scopes, avg ms), every CPU stage the renderer
  reports (cpu_ms, avg ms), the GPU frame total, the CPU frame time and FPS.

  Camera: training camera 0 (--camera-index 0) where the scene has a cameras.json;
  utsuho_plush has none and is auto-framed. The benchmark itself orbits the camera 360
  degrees around its pivot over all frames (see Application::runAutomated); the first
  frames/10 + 5 frames are warm-up and not measured.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_benchmarks.ps1 -Label phase2
  powershell -ExecutionPolicy Bypass -File tools/run_benchmarks.ps1 -Label phase2_rgba8 -ExtraArgs '--splat-format','rgba8'
  powershell -ExecutionPolicy Bypass -File tools/run_benchmarks.ps1 -Label phase3 -Renderers splat,splat-gpusort
  powershell -ExecutionPolicy Bypass -File tools/run_benchmarks.ps1 -Label phase4 -Renderers splat-gpusort,splat-tiles
  powershell -ExecutionPolicy Bypass -File tools/run_benchmarks.ps1 -Label phase5_f16 -Renderers splat-gpusort,splat-tiles -ExtraArgs '--scene-format','f16'
#>
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [string[]]$Renderers = @('splat'),
    [string[]]$Scenes = @('utsuho_plush', 'train_7k', 'bonsai_30k'),
    [uint32]$Width = 1920,
    [uint32]$Height = 1080,
    [uint32]$Frames = 300,
    [string]$Config = 'Release',
    [string[]]$ExtraArgs = @(),
    [string]$OutDir = 'docs/notes/bench'
)

$ErrorActionPreference = 'Stop'
# 'powershell -File' passes an array argument as one comma-joined string: split it again.
$ExtraArgs = @($ExtraArgs | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Renderers = @($Renderers | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Scenes = @($Scenes | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build/bin/$Config/gsplat.exe"
if (-not (Test-Path $exe)) { throw "Missing $exe; build the $Config configuration first." }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture
function F([double]$v, [string]$fmt = 'F3') { $v.ToString($fmt, $inv) }

$results = @{}  # renderer -> list of (scene, json)
$failed = $false
foreach ($r in $Renderers) {
    $results[$r] = @()
    foreach ($s in $Scenes) {
        $ply = "data/scenes/$s/point_cloud.ply"
        if (-not (Test-Path $ply)) { Write-Warning "skipping $s (not found)"; continue }
        $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
        $json = Join-Path $OutDir "${Label}_${s}_${r}.json"
        $cmd = @('--scene', $ply) + $cam + @('--renderer', $r, '--width', "$Width", '--height', "$Height",
                 '--frames', "$Frames", '--benchmark', $json) + $ExtraArgs
        Write-Host "gsplat $($cmd -join ' ')"
        & $exe @cmd *> (Join-Path $OutDir "${Label}_${s}_${r}.log")
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED with exit code $LASTEXITCODE (log: $OutDir/${Label}_${s}_${r}.log)"
            $failed = $true
            continue
        }
        Remove-Item (Join-Path $OutDir "${Label}_${s}_${r}.log")
        $results[$r] += , @($s, (Get-Content $json -Raw | ConvertFrom-Json))
    }
}

# Scopes that are part of every frame but not of the renderer.
$frameScopes = @('frame (total)', 'blit to swapchain', 'imgui', 'screenshot copy')
Write-Host ""
foreach ($r in $Renderers) {
    $rows = $results[$r]
    if ($rows.Count -eq 0) { continue }
    $gpuNames = @(); $cpuNames = @()
    foreach ($row in $rows) {
        foreach ($p in $row[1].gpu_ms.PSObject.Properties) { if ($frameScopes -notcontains $p.Name -and $gpuNames -notcontains $p.Name) { $gpuNames += $p.Name } }
        if ($row[1].cpu_ms) { foreach ($p in $row[1].cpu_ms.PSObject.Properties) { if ($cpuNames -notcontains $p.Name) { $cpuNames += $p.Name } } }
    }
    # Pipeline order for known stage names (JSON objects come back sorted by name).
    # Phase 4 (splat-tiles) stages first in their pipeline order, then the generic keywords.
    $order = @('upload', 'reset', 'preprocess', 'scan: visible', 'compact', 'depth sort', 'scan: tiles', 'tile setup',
               'duplicate', 'tile sort', 'sort', 'tile ranges', 'bin', 'draw', 'render', 'raster', 'resolve')
    $rank = { param($name) $i = 0; foreach ($k in $order) { if ($name -like "*$k*") { return $i }; $i++ }; return 99 }
    $gpuNames = @($gpuNames | Sort-Object { & $rank $_ }, { $_ })
    $cpuNames = @($cpuNames | Sort-Object { & $rank $_ }, { $_ })
    $settings = $rows[0][1].renderer_settings | ConvertTo-Json -Compress
    Write-Host "### $Label / renderer '$r' (${Width}x${Height}, $Frames frames, $Config) $($ExtraArgs -join ' ')"
    Write-Host "settings: $settings"
    Write-Host ""
    $head = @('Scene', 'Gaussians') + ($gpuNames | ForEach-Object { "GPU $_" }) + ($cpuNames | ForEach-Object { "CPU $_" }) +
            @('GPU frame total', 'CPU frame (avg)', 'CPU frame (median)', 'FPS')
    Write-Host ('| ' + ($head -join ' | ') + ' |')
    Write-Host ('|' + ('---|' * $head.Count))
    foreach ($row in $rows) {
        $j = $row[1]
        $cells = @($row[0], "$($j.scene_info.gaussians)")
        foreach ($n in $gpuNames) { $v = $j.gpu_ms.$n; $cells += $(if ($v) { F $v.avg } else { '-' }) }
        foreach ($n in $cpuNames) { $v = $j.cpu_ms.$n; $cells += $(if ($v) { F $v.avg } else { '-' }) }
        $cells += @((F $j.gpu_ms.'frame (total)'.avg), (F $j.cpu_frame_ms.avg), (F $j.cpu_frame_ms.median), (F $j.fps_avg 'F1'))
        Write-Host ('| ' + ($cells -join ' | ') + ' |')
    }
    Write-Host ""
}
# Phase 5: memory and load per run (scene format, GPU memory by owner, load/encode/upload).
Write-Host "### $Label / memory and load"
Write-Host ""
Write-Host "| Scene | Renderer | scene format | B/gaussian | scene MB | renderer MB | list MB | registered total MB | VMA device-local usage MB | PLY load ms | encode ms (incl. codebook) | upload ms | preprocess ms | duplicate ms | render/draw ms | GPU frame ms |"
Write-Host "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
foreach ($r in $Renderers) {
    foreach ($row in $results[$r]) {
        $j = $row[1]; $si = $j.scene_info; $m = $j.gpu_memory
        $owners = $m.by_owner
        $sceneMb = if ($owners.scene) { $owners.scene.mb } else { 0 }
        $rendMb = if ($owners.$r) { $owners.$r.mb } else { 0 }
        $listMb = if ($j.renderer_settings.list_mb) { F $j.renderer_settings.list_mb 'F1' } else { '-' }
        $pre = ($j.gpu_ms.PSObject.Properties | Where-Object { $_.Name -like 'preprocess*' } | Select-Object -First 1)
        $dup = ($j.gpu_ms.PSObject.Properties | Where-Object { $_.Name -like 'duplicate*' } | Select-Object -First 1)
        $ren = ($j.gpu_ms.PSObject.Properties | Where-Object { $_.Name -like 'render tiles*' -or $_.Name -like 'draw*' } | Select-Object -First 1)
        $vma = ($m.vma_heaps | Where-Object { $_.device_local } | Measure-Object -Property process_usage_mb -Sum).Sum
        $cells = @($row[0], $r, "$($si.scene_format)/$($si.scene_layout)", (F $si.bytes_per_gaussian 'F1'), (F $sceneMb 'F1'),
                   (F $rendMb 'F1'), $listMb, (F $m.total_registered_mb 'F1'), (F $vma 'F1'), (F $si.load_ms.total 'F0'),
                   (F $si.encode_ms 'F0'), (F $si.gpu_upload_ms 'F0'),
                   $(if ($pre) { F $pre.Value.avg } else { '-' }), $(if ($dup) { F $dup.Value.avg } else { '-' }),
                   $(if ($ren) { F $ren.Value.avg } else { '-' }), (F $j.gpu_ms.'frame (total)'.avg))
        Write-Host ('| ' + ($cells -join ' | ') + ' |')
    }
}
Write-Host ""
Write-Host "All values are averages in ms over the measured frames. JSON: $OutDir/${Label}_<scene>_<renderer>.json"
if ($failed) { exit 1 }
exit 0
