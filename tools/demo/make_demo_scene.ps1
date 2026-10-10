<#
.SYNOPSIS
  Builds the web demo scene "Knock Community Hall" (scbenoit, CC BY 4.0) from its SOG source.

.DESCRIPTION
  Reproduces data/demo/knock-community-hall/ (see docs/web/DEMO_SCENE.md):
    1. downloads knock-community-hall.sog + .txt (license) from the PlayCanvas engine repo,
       pinned to a commit, into source/ and checks their SHA-256 and the CC-BY-4.0 line
       (downloads are untrusted data: only parsed, never executed);
    2. installs the pinned @playcanvas/splat-transform (tools/demo/package-lock.json);
    3. SOG -> 3DGS PLY (full, 1,935,100 gaussians, SH degree 1), then canonical INRIA
       property order (tools/demo/demo_ply.py canon)          -> work/ (git-ignored);
    4. decimates to 300,000 gaussians with splat-transform --decimate on the CPU (-g cpu:
       the GPU path gives slightly different, though equally good, output), canonical
       order + provenance comments                              -> work/ (git-ignored);
    5. deterministic gzip                                       -> knock-community-hall_300k.ply.gz
    6. unless -SkipChecks: ply_info, screenshots of full and decimated scene from
       camera.json and a side camera with gsplat.exe, imgdiff PSNR, PNGs to docs/web/img/.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/demo/make_demo_scene.ps1
#>
param(
    [int]$Count = 300000,
    [string]$Config = 'Release',
    [uint32]$Width = 960,
    [uint32]$Height = 540,
    [switch]$SkipChecks
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $root

$commit = '289bc2b0f93b3d023cefdbe062976b97c97ea4b9'   # playcanvas/engine, last commit touching the .sog
$base = "https://raw.githubusercontent.com/playcanvas/engine/$commit/examples/assets/splats"
$sources = @{
    'knock-community-hall.sog' = '2c9ef67e619878cdb85d8f972cc6b1b8a8c76cbd7b4ed1137ac97bbd78785202'
    'knock-community-hall.txt' = '3145ed03e782475751edec0eb0fb258c28a2b715c2249ee9133ae2d0ff8ec8e5'
}
$dir = 'data/demo/knock-community-hall'
$src = "$dir/source"
$work = "$dir/work"
New-Item -ItemType Directory -Force $src, $work | Out-Null

function Run([string]$exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "$exe failed (exit $LASTEXITCODE)" }
}

# 1. Source + license check
foreach ($name in $sources.Keys) {
    $path = "$src/$name"
    if (-not (Test-Path $path)) { Run 'curl.exe' @('-sSfL', '-o', $path, "$base/$name") }
    $hash = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
    if ($hash -ne $sources[$name]) { throw "$path : SHA-256 $hash, expected $($sources[$name])" }
}
$license = Get-Content -Raw "$src/knock-community-hall.txt"
if ($license -notmatch 'license type:\s*CC-BY-4.0') { throw 'license file does not say CC-BY-4.0' }

# 2. Pinned converter
Push-Location tools/demo
try { Run 'npm.cmd' @('ci', '--no-audit', '--no-fund', '--silent') } finally { Pop-Location }
$st = 'tools/demo/node_modules/.bin/splat-transform.cmd'
$stVersion = 'splat-transform v' +
    (Get-Content -Raw 'tools/demo/node_modules/@playcanvas/splat-transform/package.json' | ConvertFrom-Json).version

# 3. Full conversion (SOG -> PLY), canonical property order
Run $st @('--no-tty', '-q', '-w', "$src/knock-community-hall.sog", "$work/st_full.ply")
Run 'python' @('-I', 'tools/demo/demo_ply.py', 'canon', "$work/st_full.ply", "$work/knock-community-hall_full.ply")

# 4. Decimation (CPU, deterministic), canonical order + provenance comments
Run $st @('--no-tty', '-q', '-w', '-g', 'cpu', "$work/st_full.ply", '--decimate', "$Count", "$work/st_dec.ply")
$out = "$work/knock-community-hall_300k.ply"
Run 'python' @('-I', 'tools/demo/demo_ply.py', 'canon', "$work/st_dec.ply", $out,
    'Knock Community Hall by scbenoit https://superspl.at/scene/0ff2e6dc',
    'License CC BY 4.0 https://creativecommons.org/licenses/by/4.0/',
    "Modified: converted from SOG to 3DGS PLY and decimated to $Count gaussians with @playcanvas/$stVersion")
Remove-Item "$work/st_dec.ply"

# 5. Deterministic gzip of the final PLY
Run 'python' @('-I', 'tools/demo/demo_ply.py', 'gzip', $out, "$dir/knock-community-hall_300k.ply.gz")

if ($SkipChecks) { return }

# 6. Checks and documentation screenshots
Run 'python' @('-I', 'tools/ply_info.py', $out)
if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0' }
$bin = "build/bin/$Config"
$img = 'docs/web/img'
New-Item -ItemType Directory -Force $img | Out-Null
$side = "$work/side_camera.json"
'{"mode":"orbit","position":[6.0,4.0,9.0],"target":[-2.0,2.0,-1.0],"fov_y_deg":55.0,"near":0.01,"far":1000.0}' |
    Set-Content -Encoding ascii $side
$views = @(@{ tag = ''; cam = "$dir/camera.json" }, @{ tag = '_side'; cam = $side })
foreach ($v in $views) {
    $full = "$img/demo_full$($v.tag).png"
    $dec = "$img/demo_300k$($v.tag).png"
    $common = @('--camera', $v.cam, '--frames', '5', '--width', "$Width", '--height', "$Height")
    Run "$bin/gsplat.exe" (@('--scene', "$work/knock-community-hall_full.ply", '--screenshot', $full) + $common)
    Run "$bin/gsplat.exe" (@('--scene', $out, '--screenshot', $dec) + $common)
    Run "$bin/imgdiff.exe" @($dec, $full, '--heatmap', "$img/demo_diff$($v.tag).png")
}
Get-Item "$dir/knock-community-hall_300k.ply.gz", $out, "$work/knock-community-hall_full.ply" |
    Format-Table Name, Length
