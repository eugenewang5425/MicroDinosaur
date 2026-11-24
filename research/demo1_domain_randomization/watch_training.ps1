param(
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$Study,
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$Pilot,
    [ValidateSet('continue_training.py', 'train_stop.py')][string]$Trainer = 'continue_training.py'
)

$ErrorActionPreference = 'Stop'
$trainerPath = Join-Path $PSScriptRoot $Trainer
$studyRoot = (Resolve-Path -LiteralPath $Study).Path
$outputRoot = (Resolve-Path -LiteralPath $Output).Path
$pilotFile = (Resolve-Path -LiteralPath $Pilot).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$statusFile = Join-Path $outputRoot 'status.json'
$marker = Join-Path $outputRoot 'STOP'
$watcherLog = Join-Path $outputRoot 'watcher.log'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'

function Write-WatcherNote([string]$Message) {
    $line = "{0:o} {1}" -f [DateTimeOffset]::Now, $Message
    Add-Content -LiteralPath $watcherLog -Encoding utf8 -Value $line
}

function Get-ActiveWorker {
    if (Test-Path -LiteralPath $statusFile) {
        $status = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
        $record = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($status.pid)"
        if ($record.CommandLine -and $record.CommandLine.Contains($trainerPath) -and $record.CommandLine.Contains($outputRoot)) {
            return $record
        }
    }
}

Write-WatcherNote 'Watching the actuator training worker.'
$started = 0
$lastLaunch = [DateTimeOffset]::MinValue
while (-not (Test-Path -LiteralPath $marker)) {
    if (-not (Get-ActiveWorker) -and ([DateTimeOffset]::Now - $lastLaunch).TotalSeconds -ge 75) {
        $started++
        $lastLaunch = [DateTimeOffset]::Now
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $command = '-u "{0}" --study "{1}" --output "{2}" --pilot "{3}"' -f $trainerPath, $studyRoot, $outputRoot, $pilotFile
        $launch = @{
            WindowStyle = 'Hidden'
            FilePath = $pythonPath
            ArgumentList = $command
            WorkingDirectory = $studyRoot
            RedirectStandardOutput = Join-Path $outputRoot "restart_${stamp}_out.log"
            RedirectStandardError = Join-Path $outputRoot "restart_${stamp}_err.log"
            PassThru = $true
        }
        $worker = Start-Process @launch
        Write-WatcherNote ("Started worker process {0} (restart {1})." -f $worker.Id, $started)
        if ($started -ge 3) {
            Start-Sleep -Seconds 45
        }
    }
    Start-Sleep -Seconds 15
}
Write-WatcherNote 'Stop marker observed. Worker will exit at an episode boundary.'
