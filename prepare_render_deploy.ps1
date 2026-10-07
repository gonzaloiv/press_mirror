[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$IncludeLatestSnapshot
)

$repoRoot = $PSScriptRoot
$runtimePaths = @(
    "requirements.txt",
    "server.py",
    "web_app",
    "shared",
    "static/css",
    "static/js",
    "static/favicon.png"
)

foreach ($path in $runtimePaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $path))) {
        throw "Required deployment path is missing: $path"
    }
}

$trackedSnapshots = @(& git -C $repoRoot ls-files -- "history/clusters_*.json")
if ($LASTEXITCODE -ne 0) {
    throw "Could not read tracked files from Git."
}

$includeSnapshot = $IncludeLatestSnapshot -or $trackedSnapshots.Count -eq 0
$snapshotPaths = @()
if ($includeSnapshot) {
    $historyDirectory = Join-Path $repoRoot "history"
    $latestCluster = Get-ChildItem -LiteralPath $historyDirectory -Filter "clusters_????-??-??.json" -File |
        Sort-Object Name -Descending |
        Where-Object {
            $date = $_.BaseName.Substring("clusters_".Length)
            Test-Path -LiteralPath (Join-Path $historyDirectory "headlines_$date.json")
        } |
        Select-Object -First 1

    if ($null -eq $latestCluster) {
        throw "No dated cluster/headline snapshot pair found. The app needs one to serve its pages."
    }

    $snapshotDate = $latestCluster.BaseName.Substring("clusters_".Length)
    $snapshotPaths = @(
        "history/$($latestCluster.Name)",
        "history/headlines_$snapshotDate.json"
    )
}

if ($PSCmdlet.ShouldProcess(($runtimePaths -join ", "), "Stage Render app files")) {
    & git -C $repoRoot add -- $runtimePaths
    if ($LASTEXITCODE -ne 0) {
        throw "Could not stage Render app files."
    }
}

if ($trackedSnapshots.Count -gt 0) {
    if ($PSCmdlet.ShouldProcess("tracked history changes", "Stage updates only")) {
        & git -C $repoRoot add -u -- history
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage tracked history updates."
        }
    }
}

if ($snapshotPaths.Count -gt 0) {
    if ($PSCmdlet.ShouldProcess(($snapshotPaths -join ", "), "Force-stage required snapshot data")) {
        & git -C $repoRoot add -f -- $snapshotPaths
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage the required snapshot data."
        }
    }
}

Write-Output "Render deployment files staged. No commit or push was performed."
