<#
.SYNOPSIS
  Builds the web version and runs its automated checks in headless Chrome (one command per
  web phase, like the native tools/run_*.ps1 scripts).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools/web/run_web_checks.ps1 -Test w0
  powershell -ExecutionPolicy Bypass -File tools/web/run_web_checks.ps1 -Test w0 -SkipBuild
#>
param(
    [string]$Test = 'w0',
    [switch]$SkipBuild,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

if (-not $SkipBuild) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\build_web.ps1" @(if ($Clean) { '-Clean' })
    if ($LASTEXITCODE -ne 0) { throw "web build failed ($LASTEXITCODE)" }
}

if (-not (Test-Path "$PSScriptRoot\node_modules\puppeteer-core")) {
    Push-Location $PSScriptRoot
    npm ci --no-audit --no-fund
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "npm ci failed ($code)" }
}

node "$PSScriptRoot\run_web_checks.mjs" --test $Test
exit $LASTEXITCODE
