<#
.SYNOPSIS
  Builds the web version (Emscripten + WebGPU) into build-web/dist/.

.DESCRIPTION
  Activates emsdk for this process only, puts ninja on PATH if needed, then runs the CMake
  `web` preset (CMakePresets.json): configure + build. The native build (build/) is untouched.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/web/build_web.ps1
  powershell -ExecutionPolicy Bypass -File tools/web/build_web.ps1 -Clean
#>
param(
    [string]$Emsdk = $(if ($env:EMSDK) { $env:EMSDK } else { 'C:\dev\emsdk' }),
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

# emsdk: sets EMSDK, EM_CONFIG and PATH (emcc, node, python) for this process.
if (-not (Test-Path "$Emsdk\emsdk_env.ps1")) { throw "emsdk not found at $Emsdk (pass -Emsdk <dir>)" }
# (emsdk_env prints to stderr, which 'Stop' would turn into an error in Windows PowerShell.)
$env:EMSDK_QUIET = '1'
$ErrorActionPreference = 'Continue'
. "$Emsdk\emsdk_env.ps1" *> $null
$ErrorActionPreference = 'Stop'
if (-not (Get-Command emcc -ErrorAction SilentlyContinue)) { throw 'emcc not on PATH after activating emsdk' }

# Ninja: winget installs it without always updating PATH for running shells.
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    $ninja = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $ninja) { throw 'ninja not found (winget install Ninja-build.Ninja)' }
    $env:PATH = "$($ninja.DirectoryName);$env:PATH"
}

if ($Clean -and (Test-Path build-web)) { Remove-Item -Recurse -Force build-web }

cmake --preset web
if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }
cmake --build --preset web
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

Get-ChildItem build-web/dist | Format-Table Name, Length -AutoSize | Out-String | Write-Host
