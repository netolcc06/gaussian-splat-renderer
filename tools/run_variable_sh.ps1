<#
.SYNOPSIS
  Phase 6: quality / time / memory curve of the per-gaussian SH degree (--sh-threshold)
  vs the global band drop (--scene-sh-degree).

.DESCRIPTION
  For each scene and configuration (splat-tiles, --bounds tight; scene format f32 and f16):
    - quality at 960x540, training camera 0 (utsuho auto-framed):
        (i)  PSNR vs the CPU reference on the uncompressed scene
        (ii) PSNR vs the same renderer with all bands (same scene format, no threshold)
    - time and memory at 1920x1080 (benchmark orbit, 300 frames): preprocess ms, GPU frame
      ms, scene MB, bytes per gaussian; and the kept-degree histogram from the log.
  Prints a markdown table; JSON in $BenchDir (phase6_sh_<config>_<scene>_splat-tiles.json).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_variable_sh.ps1
#>
param(
    [string]$Config = 'Release',
    [string[]]$Scenes = @('utsuho_plush', 'train_7k', 'bonsai_30k'),
    [string[]]$Formats = @('f32', 'f16'),
    [string[]]$Thresholds = @('0', '0.25', '0.5', '1', '2', '4', '8'),
    [string]$OutDir = 'build/variable_sh',
    [string]$BenchDir = 'docs/notes/bench',
    [switch]$SkipBenchmark
)
$ErrorActionPreference = 'Stop'
$Scenes = @($Scenes | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Formats = @($Formats | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Thresholds = @($Thresholds | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root "build/bin/$Config"
New-Item -ItemType Directory -Force $OutDir, $BenchDir | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture

function Get-ImgDiff([string]$a, [string]$b) {
    $j = Join-Path $OutDir 'diff.json'
    & (Join-Path $bin 'imgdiff.exe') $a $b --json $j | Out-Null
    return (Get-Content $j -Raw | ConvertFrom-Json)
}

$rows = @()
foreach ($s in $Scenes) {
    $ply = "data/scenes/$s/point_cloud.ply"
    if (-not (Test-Path $ply)) { continue }
    $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
    $small = @('--scene', $ply) + $cam + @('--width', '960', '--height', '540')
    $refPfm = Join-Path $OutDir "${s}_ref.pfm"
    if (-not (Test-Path $refPfm)) { & (Join-Path $bin 'gsplat_ref.exe') @($small + @('--pfm', $refPfm)) *> $null }
    foreach ($f in $Formats) {
        $configs = @(@{ Name = "$f all bands"; Args = @() })
        foreach ($t in $Thresholds) { $configs += @{ Name = "$f threshold $t/255"; Args = @('--sh-threshold', $t) } }
        foreach ($d in '2', '1', '0') { $configs += @{ Name = "$f global degree $d"; Args = @('--scene-sh-degree', $d) } }
        $basePng = $null
        foreach ($c in $configs) {
            $safe = ($c.Name -replace '[^A-Za-z0-9]+', '_').Trim('_')
            $png = Join-Path $OutDir "${s}_$safe.png"
            $log = Join-Path $OutDir "${s}_$safe.log"
            $a = $small + @('--renderer', 'splat-tiles', '--bounds', 'tight', '--scene-format', $f, '--screenshot', $png) + $c.Args
            & (Join-Path $bin 'gsplat.exe') @a *> $log
            if ($LASTEXITCODE -ne 0) { throw "gsplat failed: $s $($c.Name)" }
            if (-not $basePng) { $basePng = $png }
            $d1 = Get-ImgDiff $png $refPfm
            $d2 = Get-ImgDiff $png $basePng
            $hist = (Select-String -Path $log -Pattern 'per-gaussian SH degree.*' | Select-Object -First 1)
            $coeffs = if ($hist -and $hist.Line -match 'mean SH coefficients ([0-9.]+)') { [double]::Parse($Matches[1], $inv) } else { $null }
            $deg0 = if ($hist -and $hist.Line -match 'degree 0 ([0-9.]+)%') { $Matches[1] } else { '' }
            $pre = $null; $frame = $null; $mb = $null; $bpg = $null
            if (-not $SkipBenchmark) {
                $json = Join-Path $BenchDir ("phase6_sh_{0}_{1}_splat-tiles.json" -f $safe, $s)
                $b = @('--scene', $ply) + $cam + @('--renderer', 'splat-tiles', '--bounds', 'tight', '--scene-format', $f,
                       '--width', '1920', '--height', '1080', '--frames', '300', '--benchmark', $json) + $c.Args
                & (Join-Path $bin 'gsplat.exe') @b *> "$log.bench"
                if ($LASTEXITCODE -ne 0) { throw "benchmark failed: $s $($c.Name)" }
                $j = Get-Content $json -Raw | ConvertFrom-Json
                $pre = ($j.gpu_ms.PSObject.Properties | Where-Object { $_.Name -like 'preprocess*' } | Select-Object -First 1).Value.avg
                $frame = $j.gpu_ms.'frame (total)'.avg
                $mb = $j.scene_info.gpu_scene_mb
                $bpg = $j.scene_info.bytes_per_gaussian
            }
            $rows += [pscustomobject]@{ Scene = $s; Config = $c.Name; P1 = $d1.psnr_db; P2 = $d2.psnr_db; Max2 = 255 * $d2.max_abs
                Coeffs = $coeffs; Deg0 = $deg0; Pre = $pre; Frame = $frame; Mb = $mb; Bpg = $bpg }
        }
    }
}
function Db($v) { if ($v -ge 999) { 'inf' } else { ([double]$v).ToString('F2', $inv) } }
function N($v, $fmt) { if ($null -eq $v) { '-' } else { ([double]$v).ToString($fmt, $inv) } }
Write-Host ""
Write-Host "| Scene | configuration | mean SH coeffs kept | degree 0 (%) | (i) PSNR vs CPU ref (dB) | (ii) vs all bands (dB) | (ii) max (/255) | scene MB | B/gaussian | preprocess ms (1080p) | GPU frame ms |"
Write-Host "|---|---|---|---|---|---|---|---|---|---|---|"
foreach ($r in $rows) {
    Write-Host ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |" -f $r.Scene, $r.Config, (N $r.Coeffs 'F2'),
        $r.Deg0, (Db $r.P1), (Db $r.P2), (N $r.Max2 'F1'), (N $r.Mb 'F1'), (N $r.Bpg 'F1'), (N $r.Pre 'F3'), (N $r.Frame 'F3'))
}
