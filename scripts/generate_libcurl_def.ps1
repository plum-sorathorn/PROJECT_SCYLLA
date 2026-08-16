# ============================================================
# generate_libcurl_def.ps1
# Parses a Windows PE (DLL) and emits a .def file containing all
# exported symbol names, suitable for `lib.exe /DEF:foo.def`.
# Pure PowerShell + .NET Framework 4.x (no external dependencies).
# ============================================================
param(
    [Parameter(Mandatory=$true)][string]$DllPath,
    [Parameter(Mandatory=$true)][string]$DefPath
)

if (-not (Test-Path $DllPath)) {
    throw "DLL not found: $DllPath"
}

$bytes = [System.IO.File]::ReadAllBytes($DllPath)
if ($bytes.Length -lt 64) { throw "File too small to be a PE" }

# DOS header: e_lfanew at offset 0x3C
$peOffset = [BitConverter]::ToInt32($bytes, 0x3C)
if ($peOffset -le 0 -or ($peOffset + 24) -ge $bytes.Length) { throw "Bad PE offset" }

# PE signature "PE\0\0"
if ($bytes[$peOffset] -ne 0x50 -or $bytes[$peOffset+1] -ne 0x45) { throw "Missing PE signature" }

# IMAGE_FILE_HEADER starts at peOffset+4, 20 bytes
$numSections = [BitConverter]::ToInt16($bytes, $peOffset + 4 + 2)
$optHdrSize  = [BitConverter]::ToInt16($bytes, $peOffset + 4 + 16)
$optHdrStart = $peOffset + 4 + 20
$magic       = [BitConverter]::ToInt16($bytes, $optHdrStart)

# Magic: 0x10B = PE32, 0x20B = PE32+ (x64)
if ($magic -ne 0x10B -and $magic -ne 0x20B) { throw "Unknown PE magic 0x$('{0:X4}' -f $magic)" }
$is64 = ($magic -eq 0x20B)

# Data directories:
#   PE32  (magic 0x10B): start at optHdrStart + 96
#   PE32+ (magic 0x20B): start at optHdrStart + 112
# (PE32+ has an 8-byte ImageBase + 8-byte SizeOfStackReserve/etc. that
# push the data directories 16 bytes further than PE32.)
$ddStart = $optHdrStart + $(if ($is64) { 112 } else { 96 })
# Entry 0 is Export Table; each entry is 8 bytes (RVA + Size)
$expRva  = [BitConverter]::ToUInt32($bytes, $ddStart + 0)
$expSize = [BitConverter]::ToUInt32($bytes, $ddStart + 4)
if ($expRva -eq 0) { throw "DLL has no export table" }

# Section table follows optional header
$sectTableStart = $optHdrStart + $optHdrSize
# IMAGE_SECTION_HEADER = 40 bytes
$sections = @()
for ($i = 0; $i -lt $numSections; $i++) {
    $s = $sectTableStart + ($i * 40)
    $virtSize = [BitConverter]::ToUInt32($bytes, $s + 8)
    $virtAddr = [BitConverter]::ToUInt32($bytes, $s + 12)
    $rawPtr   = [BitConverter]::ToUInt32($bytes, $s + 20)
    $sections += [pscustomobject]@{ VA = $virtAddr; Size = $virtSize; Raw = $rawPtr }
}

function RvaToFileOffset([uint32]$rva) {
    foreach ($s in $script:sections) {
        if ($rva -ge $s.VA -and $rva -lt ($s.VA + [Math]::Max($s.Size, 1))) {
            return [int]($rva - $s.VA + $s.Raw)
        }
    }
    throw "RVA 0x$('{0:X8}' -f $rva) not in any section"
}

# Read IMAGE_EXPORT_DIRECTORY (40 bytes) at the export RVA
$expFile = RvaToFileOffset $expRva
$numNames    = [BitConverter]::ToUInt32($bytes, $expFile + 24)
$addrFuncs   = [BitConverter]::ToUInt32($bytes, $expFile + 28)
$addrNames   = [BitConverter]::ToUInt32($bytes, $expFile + 32)
$addrOrds    = [BitConverter]::ToUInt32($bytes, $expFile + 36)

$names = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $numNames; $i++) {
    $nameRva = [BitConverter]::ToUInt32($bytes, (RvaToFileOffset $addrNames) + ($i * 4))
    $nameOff = RvaToFileOffset $nameRva
    # Read null-terminated ASCII string
    $end = $nameOff
    while ($end -lt $bytes.Length -and $bytes[$end] -ne 0) { $end++ }
    $s = [System.Text.Encoding]::ASCII.GetString($bytes, $nameOff, $end - $nameOff)
    $names.Add($s)
}

# Build the .def file
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("; Auto-generated from $DllPath")
[void]$sb.AppendLine("; $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
[void]$sb.AppendLine("LIBRARY $([System.IO.Path]::GetFileNameWithoutExtension($DllPath))")
[void]$sb.AppendLine("EXPORTS")
foreach ($n in ($names | Sort-Object)) {
    # libcurl exports prefixed symbols like `curl_easy_init` and some
    # stdcall-like names are uncommon; ASCII is safe.
    [void]$sb.AppendLine("    $n")
}

[void](New-Item -ItemType Directory -Path (Split-Path $DefPath) -Force)
[System.IO.File]::WriteAllText($DefPath, $sb.ToString(), [System.Text.Encoding]::ASCII)
Write-Host "  Wrote $($names.Count) exports to $DefPath"
