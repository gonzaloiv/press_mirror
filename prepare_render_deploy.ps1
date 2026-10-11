[CmdletBinding(SupportsShouldProcess = $true)]
param()

$repoRoot = $PSScriptRoot
$runtimePaths = @(
    "requirements.txt",
    "main.py",
    "web_app",
    "shared",
    "static/css",
    "static/js",
    "static/favicon.png"
)
$dataPaths = @(
    "history"
)

foreach ($path in $runtimePaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $path))) {
        throw "Required deployment path is missing: $path"
    }
}

foreach ($path in $dataPaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $path))) {
        throw "Required deployment path is missing: $path"
    }
}

$historyDirectory = Join-Path $repoRoot "history"
$snapshot = Get-ChildItem -LiteralPath $historyDirectory -Directory |
    Where-Object {
        $_.Name -match '^\d{4}-\d{2}-\d{2}$' -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "clusters.json")) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "headlines.json"))
    } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if ($null -eq $snapshot) {
    throw "No dated snapshot folder with clusters.json and headlines.json found. The app needs one to serve its pages."
}

if ($PSCmdlet.ShouldProcess(($runtimePaths -join ", "), "Stage Render app files")) {
    & git -C $repoRoot add -A -- $runtimePaths
    if ($LASTEXITCODE -ne 0) {
        throw "Could not stage Render app files."
    }
}

if ($PSCmdlet.ShouldProcess(($dataPaths -join ", "), "Stage snapshot data and map images")) {
    & git -C $repoRoot add -A -- $dataPaths
    if ($LASTEXITCODE -ne 0) {
        throw "Could not stage snapshot data and map images."
    }
}

$mainRef = "refs/heads/main"
$mainTip = "$(& git -C $repoRoot rev-parse --verify --quiet $mainRef)"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($mainTip)) {
    throw "Could not resolve the main branch ($mainRef)."
}

if (-not $PSCmdlet.ShouldProcess("main branch", "Create Render deployment commit")) {
    Write-Output "Render deployment files staged. No commit was performed."
    return
}

$tree = "$(& git -C $repoRoot write-tree)"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($tree)) {
    throw "Could not create the deployment tree."
}

$mainTree = "$(& git -C $repoRoot rev-parse "$mainRef^{tree}")"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the tree of the main branch."
}

if ($tree -eq $mainTree) {
    Write-Output "Nothing to deploy: main already contains the staged files."
    return
}

$message = "Deploy website update ($(Get-Date -Format 'yyyy-MM-dd'))"
$commit = "$(& git -C $repoRoot commit-tree $tree -p $mainTip -m $message)"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
    throw "Could not create the deployment commit."
}

& git -C $repoRoot update-ref $mainRef $commit $mainTip
if ($LASTEXITCODE -ne 0) {
    throw "Could not advance the main branch."
}

Write-Output "Created commit $commit on main. No checkout or push was performed."
