[CmdletBinding()]
param(
    [string]$AptosRoot,
    [ValidateSet(224, 384)][int]$ImageSize = 384,
    [ValidateSet(2026, 2027)][int]$Seed = 2026,
    [switch]$Execute,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experiment = if ($Seed -eq 2027) {
    'aptos_resolution_seed2027_v1'
} elseif ($ImageSize -eq 224) {
    'aptos_224_matched_v1'
} else { 'aptos_384_v1' }
$image = if ($ImageSize -eq 224 -or $Seed -eq 2027) {
    'sha256:03ea39d05bf7ece0eb68fa1539e18219b89d07b17e2920e95736c18e1d068ef7'
} else { 'qsart:aptos-training' }
if ($Resume -and -not $Execute) { throw 'Resume requires Execute.' }
if (-not $AptosRoot) { $AptosRoot = $env:QSART_APTOS_ROOT }
if (-not $AptosRoot) {
    $settings = Join-Path $project '.env'
    if (Test-Path -LiteralPath $settings) {
        $line = Get-Content -LiteralPath $settings |
            Where-Object { $_ -match '^\s*QSART_APTOS_ROOT\s*=' } | Select-Object -First 1
        if ($line) { $AptosRoot = ($line -split '=', 2)[1].Trim().Trim([char]34).Trim([char]39) }
    }
}
if (-not $AptosRoot) { $AptosRoot = 'data/raw/aptos2019' }
if (-not [IO.Path]::IsPathRooted($AptosRoot)) { $AptosRoot = Join-Path $project $AptosRoot }
$AptosRoot = (Resolve-Path -LiteralPath $AptosRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $AptosRoot 'train_images') -PathType Container)) {
    throw 'Expected the audited APTOS root containing train_images.'
}

$dockerArgs = @(
    'run', '--pull=never', '--read-only', '--tmpfs', '/tmp:rw,size=256m',
    '--network', 'none', '--cpus', '4', '--memory', '4g', '--memory-swap', '4g',
    '--env', 'OMP_NUM_THREADS=4', '--env', 'OPENBLAS_NUM_THREADS=1', '--workdir', '/work',
    '--mount', "type=bind,source=$project,target=/work,readonly",
    '--mount', "type=bind,source=$AptosRoot,target=/aptos,readonly"
)
if ($Execute) {
    $commit = git -C $project rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the source commit.' }
    $changes = git -C $project status --porcelain
    if ($LASTEXITCODE -ne 0 -or $changes) { throw 'Training requires a clean checkout.' }
    $active = docker ps --filter 'label=org.qsart.experiment' --quiet
    if ($LASTEXITCODE -ne 0 -or $active) { throw 'Check Docker and any active QSART training worker.' }
    $container = "qsart-aptos-$ImageSize-" + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $dockerArgs += @('--detach', '--name', $container, '--label', "org.qsart.experiment=$experiment")
    $outputAreas = @('artifacts', 'checkpoints')
    if ($Seed -eq 2026) { $outputAreas += 'reports' }
    foreach ($area in $outputAreas) {
        $output = Join-Path $project "$area/$experiment"
        New-Item -ItemType Directory -Path $output -Force | Out-Null
        $dockerArgs += @('--mount', "type=bind,source=$output,target=/work/$area/$experiment")
    }
} else {
    $dockerArgs += '--rm'
}
$dockerArgs += @($image, 'python', 'scripts/run_aptos_384.py', '--aptos-root', '/aptos', '--image-size', "$ImageSize", '--seed', "$Seed")
if ($Execute) { $dockerArgs += @('--execute', '--source-commit', $commit.Trim()) }
if ($Resume) { $dockerArgs += '--resume' }
docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw 'The training container could not start or complete its input check.' }
if ($Execute) { Write-Output $container }
