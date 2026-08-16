# ============================================================
# PROJECT: SCYLLA // Vendor Dependencies Script (PowerShell)
# Downloads Crow, Asio, nlohmann/json, LightGBM C API, libcurl
# into cpp_core/third_party/
# Idempotent: re-running skips any vendor that is already in place.
# ============================================================

$ThirdPartyDir = "$PSScriptRoot\..\cpp_core\third_party"

function Download-File {
    param([string]$Url, [string]$OutFile)
    Write-Host "  Downloading $OutFile..."
    $parent = Split-Path $OutFile
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
}

function Download-Header {
    param([string]$Url, [string]$OutFile)
    Download-File -Url $Url -OutFile $OutFile
}

# Extract a ZIP into a temp dir, return the root of the extracted tree.
# PowerShell's Expand-Archive refuses non-.zip extensions, so we rename
# wheel files to .zip before extracting (a wheel *is* a zip file).
function Expand-ZipToTemp {
    param([string]$ZipPath, [string]$Tag)
    $dest = Join-Path $env:TEMP $Tag
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    if ($ZipPath -notmatch '\.zip$') {
        $renamed = "$ZipPath.zip"
        Copy-Item -Path $ZipPath -Destination $renamed -Force
        Expand-Archive -Path $renamed -DestinationPath $dest -Force
        Remove-Item $renamed -Force
    } else {
        Expand-Archive -Path $ZipPath -DestinationPath $dest -Force
    }
    return $dest
}

Write-Host "`n[SCYLLA] Fetching vendored C++ headers/libs into $ThirdPartyDir`n"

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
    Invoke-WebRequest -Uri "https://github.com/CrowCpp/Crow/archive/refs/tags/v1.2.0.zip" `
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

# ---- 4. LightGBM C API v4.3.0 (Windows x64) ----
# Source: official Windows wheel (MSVC-built) from the LightGBM v4.3.0 GitHub
# release, plus the public C API headers from the matching source tag.
# The wheel ships:
#   - lightgbm/bin/lib_lightgbm.dll   (runtime DLL)
#   - lightgbm/lib/lib_lightgbm.lib   (MSVC import library)
# The wheel does NOT ship the C headers. c_api.h transitively includes
# <LightGBM/arrow.h> and <LightGBM/export.h>, so we vendor the entire
# include/LightGBM/ directory from the source tag.
# Layout after vendoring (so #include <lightgbm/c_api.h> works):
#   third_party/lightgbm/bin/lib_lightgbm.dll
#   third_party/lightgbm/lib/lib_lightgbm.lib
#   third_party/lightgbm/include/lightgbm/c_api.h
#   third_party/lightgbm/include/lightgbm/arrow.h
#   third_party/lightgbm/include/lightgbm/export.h
#   ... (all other public headers)
$lgbDir        = "$ThirdPartyDir\lightgbm"
$lgbBin        = "$lgbDir\bin"
$lgbLib        = "$lgbDir\lib"
$lgbIncludeDir = "$lgbDir\include\lightgbm"
$lgbDone       = (Test-Path "$lgbBin\lib_lightgbm.dll") -and `
                 (Test-Path "$lgbLib\lib_lightgbm.lib") -and `
                 (Test-Path "$lgbIncludeDir\c_api.h") -and `
                 (Test-Path "$lgbIncludeDir\arrow.h")

if (-not $lgbDone) {
    New-Item -ItemType Directory -Path $lgbBin,$lgbLib,$lgbIncludeDir -Force | Out-Null
    $wheelZip = "$env:TEMP\lightgbm-4.3.0.whl"
    Write-Host "  Downloading LightGBM 4.3.0 wheel..."
    try {
        Invoke-WebRequest -Uri "https://github.com/lightgbm-org/LightGBM/releases/download/v4.3.0/lightgbm-4.3.0-py3-none-win_amd64.whl" `
            -OutFile $wheelZip -UseBasicParsing
    } catch {
        # Fallback: PyPI (same wheel, mirrored)
        Invoke-WebRequest -Uri "https://files.pythonhosted.org/packages/3c/91/31a4f60e8ed2cd6c8fa6a3e2df16c54a3d8b0e7ac4bcc5d8ebdcfe5e3a/lightgbm-4.3.0-py3-none-win_amd64.whl" `
            -OutFile $wheelZip -UseBasicParsing
    }
    $wheelRoot = Expand-ZipToTemp -ZipPath $wheelZip -Tag "lgb_wheel"
    Copy-Item -Path "$wheelRoot\lightgbm\bin\lib_lightgbm.dll" -Destination $lgbBin -Force
    Copy-Item -Path "$wheelRoot\lightgbm\lib\lib_lightgbm.lib" -Destination $lgbLib -Force
    Remove-Item $wheelZip -Force
    Remove-Item $wheelRoot -Recurse -Force

    # Fetch the public C API headers from the v4.3.0 source tag.
    # We need the whole include/LightGBM/ directory because c_api.h
    # transitively includes several sibling headers (arrow.h, export.h, ...).
    $srcZip = "$env:TEMP\lightgbm-src.zip"
    Write-Host "  Downloading LightGBM 4.3.0 source (for headers)..."
    Invoke-WebRequest -Uri "https://github.com/lightgbm-org/LightGBM/archive/refs/tags/v4.3.0.zip" `
        -OutFile $srcZip -UseBasicParsing
    $srcRoot = Expand-ZipToTemp -ZipPath $srcZip -Tag "lgb_src"
    $top = Get-ChildItem $srcRoot -Directory | Select-Object -First 1
    # Copy include/LightGBM/* -> include/lightgbm/* so that the include
    # path becomes <lightgbm/c_api.h> matching the wheel's directory layout.
    Copy-Item -Path "$($top.FullName)\include\LightGBM\*" -Destination $lgbIncludeDir -Recurse -Force
    Remove-Item $srcZip -Force
    Remove-Item $srcRoot -Recurse -Force
    Write-Host "  [OK] LightGBM C API v4.3.0 (lib_lightgbm.dll + .lib + headers)"
} else { Write-Host "  [SKIP] LightGBM already vendored" }

# ---- 5. libcurl (Windows x64) ----
# curl.se/windows/ no longer publishes an MSVC build; the current official
# build is MinGW-built (stable C ABI, linkable from MSVC via a generated
# import library). We use it and create libcurl.lib from the DLL's PE
# export table.
# Layout:
#   third_party/curl/bin/libcurl-x64.dll
#   third_party/curl/lib/libcurl.lib      (MSVC import lib, generated)
#   third_party/curl/lib/libcurl.def      (input for `lib /def`)
#   third_party/curl/include/curl/*.h
$curlDir        = "$ThirdPartyDir\curl"
$curlBin        = "$curlDir\bin"
$curlLib        = "$curlDir\lib"
$curlIncludeDir = "$curlDir\include"

# Phase A: ensure DLL + headers are vendored.
$curlAssetsDone = (Test-Path "$curlBin\libcurl-x64.dll") -and `
                  (Test-Path "$curlIncludeDir\curl\curl.h")
if (-not $curlAssetsDone) {
    New-Item -ItemType Directory -Path $curlBin,$curlLib,$curlIncludeDir -Force | Out-Null
    $curlZip = "$env:TEMP\curl.zip"
    Write-Host "  Downloading libcurl (official Windows x64 build)..."
    # Use the rolling `latest` redirects so this script does not pin a version
    # that becomes unavailable; the asset names have been stable since 7.x.
    Invoke-WebRequest -Uri "https://curl.se/windows/latest.cgi?p=win64-mingw.zip" `
        -OutFile $curlZip -UseBasicParsing -MaximumRedirection 5
    $curlRoot = Expand-ZipToTemp -ZipPath $curlZip -Tag "curl_extract"
    # The zip contains a single top-level dir like curl-8.10.1_2-win64-mingw/
    $top = Get-ChildItem $curlRoot -Directory | Select-Object -First 1
    Copy-Item -Path "$($top.FullName)\bin\libcurl-x64.dll" -Destination $curlBin -Force
    Copy-Item -Path "$($top.FullName)\include\*" -Destination $curlIncludeDir -Recurse -Force
    Remove-Item $curlZip -Force
    Remove-Item $curlRoot -Recurse -Force
} else { Write-Host "  [SKIP] libcurl DLL + headers already vendored" }

# Phase B: ensure the MSVC import lib is generated. The .def is small and
# deterministic, so we always re-write it. The .lib is the slow part; we
# only attempt it if it's missing AND `lib.exe` is available.
$defPath = "$curlLib\libcurl.def"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\generate_libcurl_def.ps1" `
    -DllPath "$curlBin\libcurl-x64.dll" -DefPath $defPath 2>&1 | Out-String | Write-Host

$libExe = $null
$w = Get-Command "lib.exe" -ErrorAction SilentlyContinue
if ($w) { $libExe = $w.Source }
if (-not $libExe) {
    $vsRoot = "${env:ProgramFiles(x86)}\Microsoft Visual Studio"
    if (Test-Path $vsRoot) {
        $libExe = Get-ChildItem -Path $vsRoot -Recurse -Filter "lib.exe" -ErrorAction SilentlyContinue |
                  Select-Object -First 1 -ExpandProperty FullName
    }
}

if (Test-Path "$curlLib\libcurl.lib") {
    Write-Host "  [OK] libcurl (libcurl-x64.dll + libcurl.lib) -- import lib already present"
} elseif ($libExe) {
    Write-Host "  Generating MSVC import lib via $libExe..."
    Push-Location $curlLib
    & $libExe /DEF:libcurl.def /MACHINE:X64 /OUT:libcurl.lib /NODEFAULTLIB /NOLOGO | Out-Null
    Pop-Location
    Write-Host "  [OK] libcurl (libcurl-x64.dll + libcurl.lib)"
} else {
    Write-Host "  [WARN] lib.exe not found; libcurl.lib NOT generated." -ForegroundColor Yellow
    $devPromptHint = "open 'x64 Native Tools Command Prompt'"
    $manualCmd = "lib /DEF:" + $defPath + " /MACHINE:X64 /OUT:" + $curlLib + "\libcurl.lib"
    Write-Host "         Build will fail. To fix: $devPromptHint" -ForegroundColor Yellow
    Write-Host "         and re-run this script, OR run:  $manualCmd" -ForegroundColor Yellow
}

Write-Host "`n[SCYLLA] Vendor download complete.`n"
