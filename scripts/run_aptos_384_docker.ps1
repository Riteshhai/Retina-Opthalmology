[CmdletBinding()]
param(
    [string]$AptosRoot,
    [switch]$Execute,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
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
    $active = docker ps --filter 'label=org.qsart.experiment=aptos_384_v1' --quiet
    if ($LASTEXITCODE -ne 0 -or $active) { throw 'Check Docker and any active 384-pixel worker.' }
    $container = 'qsart-aptos-384-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $dockerArgs += @('--detach', '--name', $container, '--label', 'org.qsart.experiment=aptos_384_v1')
    foreach ($area in @('artifacts', 'checkpoints', 'reports')) {
        $output = Join-Path $project "$area/aptos_384_v1"
        New-Item -ItemType Directory -Path $output -Force | Out-Null
        $dockerArgs += @('--mount', "type=bind,source=$output,target=/work/$area/aptos_384_v1")
    }
} else {
    $dockerArgs += '--rm'
}
$dockerArgs += @('qsart:aptos-training', 'python', 'scripts/run_aptos_384.py', '--aptos-root', '/aptos')
if ($Execute) { $dockerArgs += @('--execute', '--source-commit', $commit.Trim()) }
if ($Resume) { $dockerArgs += '--resume' }
docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw 'The training container could not start or complete its input check.' }
if ($Execute) { Write-Output $container }
