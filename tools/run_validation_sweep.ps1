<#
.SYNOPSIS
  Phase 6 Debug validation sweep: every Phase 6 option (and the defaults of every renderer)
  under the Khronos validation layer, once with the default checks and once with
  synchronization validation (VK_LAYER_VALIDATE_SYNC=1 + VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC=1).

.DESCRIPTION
  Each run: bonsai_30k, training camera 0, 960x540, 4 frames (screenshot) in the Debug build.
  Passes when gsplat exits 0 and prints "Validation messages: 0 errors, 0 warnings" (exit 3
  otherwise). Writes one line per run to $Out (default/sync appended to the name).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/run_validation_sweep.ps1 -Mode default
  powershell -ExecutionPolicy Bypass -File tools/run_validation_sweep.ps1 -Mode sync
#>
param(
    [ValidateSet('default', 'sync')][string]$Mode = 'default',
    [string]$Scene = 'data/scenes/bonsai_30k/point_cloud.ply',
    [string]$Out = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'build/bin/Debug/gsplat.exe'
if (-not $Out) { $Out = "docs/notes/bench/phase6_validation_$Mode.txt" }
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0'
if ($Mode -eq 'sync') {
    $env:VK_LAYER_VALIDATE_SYNC = '1'
    $env:VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC = '1'
} else {
    Remove-Item Env:VK_LAYER_VALIDATE_SYNC -ErrorAction SilentlyContinue
    Remove-Item Env:VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC -ErrorAction SilentlyContinue
}
$runs = @(
    '',  # the Phase 6 defaults: splat-tiles, tight, adaptive
    '--bounds rect', '--bounds cutoff-only', '--tile-size 8', '--tile-size 8 --bounds rect',
    '--tile-sort global', '--tile-duplicate per-splat', '--tile-sort global --tile-duplicate per-splat --tile-size 8',
    '--render-splat p32', '--render-splat p24', '--render-splat p24 --tile-size 8',
    '--sort-window 4', '--sort-window 16 --render-splat p32', '--sort-window 4 --splat-format rgba32f',
    '--render-batch 128', '--render-batch 512', '--render-vote-every 0', '--render-vote-every 2',
    '--render-shared aos', '--render-fp16', '--tile-stats build/val_stats',
    '--tile-budget-mb 4 --tile-list fixed', '--splat-bound radius', '--splat-format rgba8', '--splat-format rgba32f',
    '--sh-threshold 1', '--scene-format f16 --sh-threshold 1', '--scene-format q8 --sh-threshold 1',
    '--scene-format cb --sh-threshold 1', '--scene-format f16 --scene-layout aos --sh-threshold 1',
    '--renderer splat', '--renderer splat --bounds rect', '--renderer splat-gpusort', '--renderer splat-gpusort --bounds cutoff-only',
    '--renderer splat-gpusort --bounds tight --splat-bound radius', '--renderer points --sh-threshold 1',
    '--orbit-step 1 --sequence build/val_seq', 'PIPELINE_STATS'
)
$lines = @()
$fail = 0
foreach ($r in $runs) {
    $a = @('--scene', $Scene, '--camera-index', '0', '--width', '960', '--height', '540')
    if ($r -eq 'PIPELINE_STATS') { $a += @('--pipeline-stats', 'build/val_pipeline_stats.json') }
    else { $a += @('--frames', '4', '--screenshot', 'build/val.png') + @($r -split ' ' | Where-Object { $_ }) }
    $log = & $exe @a 2>&1 | Out-String
    $code = $LASTEXITCODE
    $m = [regex]::Match($log, 'Validation messages: (\d+) errors, (\d+) warnings')
    $msg = if ($m.Success) { $m.Value } else { 'no validation summary' }
    $ok = $code -eq 0 -and $m.Success -and $m.Groups[1].Value -eq '0' -and $m.Groups[2].Value -eq '0'
    if (-not $ok) { $fail++ }
    $line = "{0} [{1}] '{2}': exit {3}, {4}" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $Mode, $(if ($r) { $r } else { '(defaults)' }), $code, $msg
    $lines += $line
    Write-Host $line
}
$lines += "$($runs.Count) runs, $fail failures"
$lines | Set-Content -Encoding utf8 $Out
if ($fail) { exit 1 }
exit 0
