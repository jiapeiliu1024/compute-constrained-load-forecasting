param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PackageRoot = $PSScriptRoot
$Raw = Join-Path $PackageRoot "data\raw"
$Processed = Join-Path $PackageRoot "data\processed"
$Downloads = Join-Path $Raw "downloads"
$Extracted = Join-Path $Raw "extracted"
New-Item -ItemType Directory -Force $Raw,$Processed,$Downloads,$Extracted | Out-Null

function Get-OfficialFile([string]$Url, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Destination)) {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Destination
    } else {
        Write-Host "Using existing $Destination"
    }
    if ((Get-Item -LiteralPath $Destination).Length -eq 0) {
        throw "Downloaded file is empty: $Destination"
    }
}

function Expand-UciArchive([string]$Archive, [string]$Name) {
    $Destination = Join-Path $Extracted $Name
    New-Item -ItemType Directory -Force $Destination | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    return $Destination
}

$SteelZip = Join-Path $Downloads "uci_851_steel.zip"
$TetouanZip = Join-Path $Downloads "uci_849_tetouan.zip"
$EclZip = Join-Path $Downloads "uci_321_ecl.zip"
$AppliancesZip = Join-Path $Downloads "uci_374_appliances.zip"
$OpsdCsv = Join-Path $Raw "opsd_time_series_60min_singleindex.csv"

Get-OfficialFile "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip" $SteelZip
Get-OfficialFile "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip" $TetouanZip
Get-OfficialFile "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip" $EclZip
Get-OfficialFile "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip" $AppliancesZip
Get-OfficialFile "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv" $OpsdCsv

$SteelExtracted = Expand-UciArchive $SteelZip "steel"
$TetouanExtracted = Expand-UciArchive $TetouanZip "tetouan"
$EclExtracted = Expand-UciArchive $EclZip "ecl"
$AppliancesExtracted = Expand-UciArchive $AppliancesZip "appliances"

$SteelSource = Get-ChildItem $SteelExtracted -Recurse -File -Filter "*.csv" | Select-Object -First 1
$TetouanSource = Get-ChildItem $TetouanExtracted -Recurse -File -Filter "*.csv" | Where-Object Name -Match "Tetuan|Tetouan" | Select-Object -First 1
$EclSource = Get-ChildItem $EclExtracted -Recurse -File | Where-Object Name -EQ "LD2011_2014.txt" | Select-Object -First 1
$AppliancesSource = Get-ChildItem $AppliancesExtracted -Recurse -File -Filter "*.csv" | Where-Object Name -Match "energydata" | Select-Object -First 1
if (-not $SteelSource -or -not $TetouanSource -or -not $EclSource -or -not $AppliancesSource) {
    throw "One or more expected files were not found in the official archives."
}

New-Item -ItemType Directory -Force (Join-Path $Raw "uci_steel"),(Join-Path $Raw "uci_tetouan_power"),(Join-Path $Raw "uci_electricity"),(Join-Path $Raw "uci_appliances") | Out-Null
Copy-Item -LiteralPath $SteelSource.FullName -Destination (Join-Path $Raw "uci_steel\data.csv") -Force
Copy-Item -LiteralPath $TetouanSource.FullName -Destination (Join-Path $Raw "uci_tetouan_power\Tetuan City power consumption.csv") -Force
Copy-Item -LiteralPath $EclSource.FullName -Destination (Join-Path $Raw "uci_electricity\LD2011_2014.txt") -Force
Copy-Item -LiteralPath $AppliancesSource.FullName -Destination (Join-Path $Raw "uci_appliances\data.csv") -Force
Copy-Item -LiteralPath $TetouanZip -Destination (Join-Path $Raw "uci_tetouan_power.zip") -Force

& $Python "code\preprocess_tetouan_load.py" --output "data\processed\uci_tetouan_hourly.csv"
if ($LASTEXITCODE -ne 0) { throw "Tetouan preprocessing failed." }
& $Python "code\preprocess_uci_load.py" --output "data\processed\uci_ecl_hourly.csv"
if ($LASTEXITCODE -ne 0) { throw "UCI-ECL preprocessing failed." }
& $Python "code\preprocess_opsd_aggregate.py"
if ($LASTEXITCODE -ne 0) { throw "OPSD preprocessing failed." }

$Manifest = Get-ChildItem $Raw -Recurse -File | Get-FileHash -Algorithm SHA256 | ForEach-Object {
    [PSCustomObject]@{ path = $_.Path.Substring($PackageRoot.Length + 1); sha256 = $_.Hash }
}
$Manifest | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Raw "DATA_SHA256.json")
Write-Host "Data preparation complete."

