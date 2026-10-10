<#
.SYNOPSIS
  Phase 5 quality gates: every scene format / render-splat configuration vs the CPU
  reference and vs the same GPU renderer uncompressed.

.DESCRIPTION
  For each real scene (training camera 0; utsuho auto-framed) and each renderer:
    - the CPU reference (gsplat_ref, INRIA forward pass) on the UNCOMPRESSED scene -> PFM
    - the GPU renderer with the default f32 format -> the "uncompressed GPU" image
    - the GPU renderer for every configuration below, and two metrics per configuration:
        (i)  PSNR vs the CPU reference (uncompressed)
        (ii) PSNR vs the same GPU renderer uncompressed ("inf" = identical images)
  Thresholds per tier (dB; measured in Phase 5, see docs/notes/phase5.md "Quality gates"):
      tier          (i) vs CPU reference                 (ii) vs GPU f32
      lossless      >= the f32 GPU image's (i)            identical (inf)
      near          >= (i of f32) - 0.5                   >= 60
      q8            >= (i of f32) - 1.5                   >= 55
      lossy (cb)    >= 30                                 >= 30
      band drop     image identical to the same runtime --sh-degree (memory-only change)
  "identical" = bit-identical PNGs for splat-tiles (deterministic); for splat-gpusort it means
  >= 95 dB, because its atomic-append compaction is not deterministic run to run (Phase 3):
  the SAME configuration rendered twice already differs by one 1/255 step in a few pixels
  (103-110 dB measured).
  Writes error heatmaps (imgdiff, 0 .. 16/255) of every configuration vs the GPU f32 image,
  and a markdown table. Exit 1 if any gate fails or any run fails/reports validation messages.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_compression_checks.ps1
  powershell -ExecutionPolicy Bypass -File tools/run_compression_checks.ps1 -Renderers splat-tiles -Scenes bonsai_30k
#>
param(
    [string]$Config = 'Release',
    [string[]]$Renderers = @('splat-tiles', 'splat-gpusort'),
    [string[]]$Scenes = @('utsuho_plush', 'train_7k', 'bonsai_30k'),
    [uint32]$Width = 960,
    [uint32]$Height = 540,
    [string]$OutDir = 'build/compression_checks'
)

$ErrorActionPreference = 'Stop'
$Renderers = @($Renderers | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Scenes = @($Scenes | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root "build/bin/$Config"
foreach ($exe in 'gsplat.exe', 'gsplat_ref.exe', 'imgdiff.exe') {
    if (-not (Test-Path (Join-Path $bin $exe))) { throw "Missing $bin/$exe; build the $Config configuration first." }
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$inv = [System.Globalization.CultureInfo]::InvariantCulture

# name, gsplat arguments, tier, renderers it applies to ('*' = all), [runtime twin for band drops]
$configs = @(
    @{ Name = 'f32/split';        Args = @('--scene-layout', 'split');                     Tier = 'lossless'; Only = '*' },
    @{ Name = 'f32/aos';          Args = @('--scene-layout', 'aos');                       Tier = 'lossless'; Only = '*' },
    @{ Name = 'f16';              Args = @('--scene-format', 'f16');                       Tier = 'near';     Only = '*' },
    @{ Name = 'f16/aos';          Args = @('--scene-format', 'f16', '--scene-layout', 'aos'); Tier = 'near';  Only = '*' },
    @{ Name = 'q8';               Args = @('--scene-format', 'q8');                        Tier = 'q8';       Only = '*' },
    @{ Name = 'cb';               Args = @('--scene-format', 'cb');                        Tier = 'lossy';    Only = '*' },
    @{ Name = 'render-splat p32'; Args = @('--render-splat', 'p32');                       Tier = 'near';     Only = 'splat-tiles' },
    @{ Name = 'render-splat p24'; Args = @('--render-splat', 'p24');                       Tier = 'near';     Only = 'splat-tiles' },
    @{ Name = 'f16 + p32';        Args = @('--scene-format', 'f16', '--render-splat', 'p32'); Tier = 'near';  Only = 'splat-tiles' },
    @{ Name = 'q8 + p32';         Args = @('--scene-format', 'q8', '--render-splat', 'p32');  Tier = 'q8';    Only = 'splat-tiles' },
    @{ Name = 'cb + p24';         Args = @('--scene-format', 'cb', '--render-splat', 'p24');  Tier = 'lossy'; Only = 'splat-tiles' },
    @{ Name = 'adaptive list';    Args = @('--tile-list', 'adaptive');                     Tier = 'lossless'; Only = 'splat-tiles' },
    @{ Name = 'scene SH degree 2'; Args = @('--scene-sh-degree', '2'); Tier = 'banddrop'; Only = '*'; Twin = @('--sh-degree', '2') },
    @{ Name = 'scene SH degree 1'; Args = @('--scene-sh-degree', '1'); Tier = 'banddrop'; Only = '*'; Twin = @('--sh-degree', '1') },
    @{ Name = 'scene SH degree 0'; Args = @('--scene-sh-degree', '0'); Tier = 'banddrop'; Only = '*'; Twin = @('--sh-degree', '0') }
)

function Run-Gsplat([string[]]$a, [string]$log) {
    & (Join-Path $bin 'gsplat.exe') @a *> $log
    return $LASTEXITCODE
}
function Get-Diff([string]$test, [string]$ref, [string]$json, [string]$heatmap) {
    $a = @($test, $ref, '--json', $json)
    if ($heatmap) { $a += @('--heatmap', $heatmap) }
    & (Join-Path $bin 'imgdiff.exe') @a | Out-Null
    if ($LASTEXITCODE -gt 1) { throw "imgdiff failed on $test vs $ref" }
    return Get-Content $json -Raw | ConvertFrom-Json
}
function Bytes-Per-Gaussian([string]$log) {
    $m = Select-String -Path $log -Pattern '\(([0-9.]+) B/gaussian\)' | Select-Object -First 1
    if ($m) { return [double]::Parse($m.Matches[0].Groups[1].Value, $inv) } else { return 0.0 }
}
function Format-Db([double]$v) { if ($v -ge 999) { 'inf' } else { $v.ToString('F2', $inv) } }

$rows = @()
$failed = $false
foreach ($s in $Scenes) {
    $ply = "data/scenes/$s/point_cloud.ply"
    if (-not (Test-Path $ply)) { Write-Warning "skipping $s (not found)"; continue }
    $cam = if (Test-Path "data/scenes/$s/cameras.json") { @('--camera-index', '0') } else { @() }
    $common = @('--scene', $ply) + $cam + @('--width', "$Width", '--height', "$Height")
    $refPfm = Join-Path $OutDir "${s}_ref.pfm"
    & (Join-Path $bin 'gsplat_ref.exe') @($common + @('--pfm', $refPfm)) *> (Join-Path $OutDir "${s}_ref.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL ${s}: gsplat_ref exit $LASTEXITCODE"; $failed = $true; continue }

    foreach ($r in $Renderers) {
        $tag = "${s}_${r}"
        $identical = if ($r -eq 'splat-gpusort') { 95.0 } else { 999.0 }  # see the header
        $basePng = Join-Path $OutDir "${tag}_f32.png"
        $code = Run-Gsplat ($common + @('--renderer', $r, '--screenshot', $basePng)) (Join-Path $OutDir "${tag}_f32.log")
        if ($code -ne 0) { Write-Host "FAIL ${tag}: gsplat exit $code"; $failed = $true; continue }
        $base = Get-Diff $basePng $refPfm (Join-Path $OutDir "${tag}_f32_vs_ref.json") $null
        $rows += [pscustomobject]@{ Scene = $s; Renderer = $r; Config = 'f32 (default)'; Bytes = (Bytes-Per-Gaussian (Join-Path $OutDir "${tag}_f32.log"))
            P1 = $base.psnr_db; Max1 = $base.max_abs * 255; P2 = 999.0; Max2 = 0.0; Over2 = 0.0; Gate = 'baseline'; Result = 'PASS' }

        foreach ($c in $configs) {
            if ($c.Only -ne '*' -and $c.Only -ne $r) { continue }
            $safe = ($c.Name -replace '[^A-Za-z0-9]+', '_').Trim('_')
            $png = Join-Path $OutDir "${tag}_${safe}.png"
            $log = Join-Path $OutDir "${tag}_${safe}.log"
            $code = Run-Gsplat ($common + @('--renderer', $r, '--screenshot', $png) + $c.Args) $log
            if ($code -ne 0) { Write-Host "FAIL ${tag} $($c.Name): gsplat exit $code"; $failed = $true; continue }
            $d1 = Get-Diff $png $refPfm (Join-Path $OutDir "${tag}_${safe}_vs_ref.json") $null
            $d2 = Get-Diff $png $basePng (Join-Path $OutDir "${tag}_${safe}_vs_f32.json") (Join-Path $OutDir "${tag}_${safe}_vs_f32_heatmap.png")
            $ok = $true
            $gate = ''
            switch ($c.Tier) {
                'lossless' { $ok = $d2.psnr_db -ge $identical; $gate = 'identical to f32' }
                'near'     { $ok = ($d1.psnr_db -ge $base.psnr_db - 0.5) -and ($d2.psnr_db -ge 60); $gate = '(i) >= f32 - 0.5, (ii) >= 60' }
                'q8'       { $ok = ($d1.psnr_db -ge $base.psnr_db - 1.5) -and ($d2.psnr_db -ge 55); $gate = '(i) >= f32 - 1.5, (ii) >= 55' }
                'lossy'    { $ok = ($d1.psnr_db -ge 30) -and ($d2.psnr_db -ge 30); $gate = '(i) >= 30, (ii) >= 30' }
                'banddrop' {
                    $twinPng = Join-Path $OutDir "${tag}_${safe}_runtime.png"
                    $tc = Run-Gsplat ($common + @('--renderer', $r, '--screenshot', $twinPng) + $c.Twin) "$twinPng.log"
                    $dt = Get-Diff $png $twinPng (Join-Path $OutDir "${tag}_${safe}_vs_runtime.json") $null
                    $ok = ($tc -eq 0) -and ($dt.psnr_db -ge $identical)
                    $gate = "identical to runtime $($c.Twin -join ' ')"
                }
            }
            if (-not $ok) { $failed = $true }
            $rows += [pscustomobject]@{ Scene = $s; Renderer = $r; Config = $c.Name; Bytes = (Bytes-Per-Gaussian $log)
                P1 = $d1.psnr_db; Max1 = $d1.max_abs * 255; P2 = $d2.psnr_db; Max2 = $d2.max_abs * 255
                Over2 = $d2.pixels_over_threshold_pct; Gate = $gate; Result = $(if ($ok) { 'PASS' } else { 'FAIL' }) }
        }
    }
}

Write-Host ""
Write-Host "Scene formats / render splats: (i) vs the CPU reference on the uncompressed scene, (ii) vs the same GPU renderer with f32 (${Width}x${Height}, $Config)"
Write-Host ""
Write-Host "| Scene | Renderer | Configuration | scene B/gaussian | (i) PSNR vs CPU ref (dB) | (i) max abs (/255) | (ii) PSNR vs GPU f32 (dB) | (ii) max abs (/255) | (ii) px > 2/255 (%) | gate | result |"
Write-Host "|---|---|---|---|---|---|---|---|---|---|---|"
foreach ($x in $rows) {
    Write-Host ([string]::Format($inv, "| {0} | {1} | {2} | {3:F1} | {4} | {5:F1} | {6} | {7:F1} | {8:F3} | {9} | {10} |", $x.Scene, $x.Renderer,
        $x.Config, $x.Bytes, (Format-Db $x.P1), $x.Max1, (Format-Db $x.P2), $x.Max2, $x.Over2, $x.Gate, $x.Result))
}
Write-Host ""
if ($failed) { Write-Host "COMPRESSION CHECKS FAILED (outputs in $OutDir)"; exit 1 }
Write-Host "All compression checks passed (outputs in $OutDir)"
exit 0
