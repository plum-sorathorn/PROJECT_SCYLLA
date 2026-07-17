# ============================================================
# PROJECT: SCYLLA // Vendor Dependencies Script (PowerShell)
# Downloads Crow, Asio, and nlohmann/json headers into third_party/
# ============================================================

$ThirdPartyDir = "$PSScriptRoot\..\cpp_core\third_party"

function Download-Header {
    param([string]$Url, [string]$OutFile)
    Write-Host "  Downloading $OutFile..."
    $parent = Split-Path $OutFile
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
}

Write-Host "`n[SCYLLA] Fetching vendored C++ headers into $ThirdPartyDir`n"

# ---- 1. nlohmann/json (single header) ----
$nlohmannDir = "$ThirdPartyDir\nlohmann"
if (-not (Test-Path "$nlohmannDir\json.hpp")) {
    New-Item -ItemType Directory -Path $nlohmannDir -Force | Out-Null
    Download-Header `
        "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp" `
        "$nlohmannDir\json.hpp"
    Write-Host "  [OK] nlohmann/json.hpp"
} else { Write-Host "  [SKIP] nlohmann/json.hpp already exists" }

# ---- 2. Asio standalone (header-only) ----
$asioDir = "$ThirdPartyDir\asio"
if (-not (Test-Path "$asioDir\include\asio.hpp")) {
    New-Item -ItemType Directory -Path "$asioDir\include" -Force | Out-Null
    $asioZip = "$env:TEMP\asio.zip"
    Write-Host "  Downloading Asio standalone..."
    Invoke-WebRequest -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-28-2.zip" `
        -OutFile $asioZip -UseBasicParsing
    Write-Host "  Extracting Asio..."
    Expand-Archive -Path $asioZip -DestinationPath "$env:TEMP\asio_extract" -Force
    $asioExtracted = Get-ChildItem "$env:TEMP\asio_extract" -Directory | Select-Object -First 1
    Copy-Item -Path "$($asioExtracted.FullName)\asio\include\*" -Destination "$asioDir\include\" -Recurse -Force
    Remove-Item $asioZip -Force
    Remove-Item "$env:TEMP\asio_extract" -Recurse -Force
    Write-Host "  [OK] Asio standalone headers"
} else { Write-Host "  [SKIP] Asio already exists" }

# ---- 3. Crow (header-only) ----
$crowDir = "$ThirdPartyDir\crow"
if (-not (Test-Path "$crowDir\include\crow.h")) {
    New-Item -ItemType Directory -Path "$crowDir\include" -Force | Out-Null
    $crowZip = "$env:TEMP\crow.zip"
    Write-Host "  Downloading Crow..."
    Invoke-WebRequest -Uri "https://github.com/CrowCpp/Crow/releases/download/v1.2.1/crow-v1.2.1.zip" `
        -OutFile $crowZip -UseBasicParsing
    Write-Host "  Extracting Crow..."
    Expand-Archive -Path $crowZip -DestinationPath "$env:TEMP\crow_extract" -Force
    $crowExtracted = Get-ChildItem "$env:TEMP\crow_extract" -Recurse -Filter "crow.h" | Select-Object -First 1
    if ($crowExtracted) {
        $crowIncludeDir = $crowExtracted.DirectoryName
        Copy-Item -Path "$crowIncludeDir\*" -Destination "$crowDir\include\" -Recurse -Force
    }
    Remove-Item $crowZip -Force
    Remove-Item "$env:TEMP\crow_extract" -Recurse -Force
    Write-Host "  [OK] Crow headers"
} else { Write-Host "  [SKIP] Crow already exists" }

Write-Host "`n[SCYLLA] Vendor download complete.`n"
