# Stop neiro training on the box, and nothing else. See stop_training.bat
# for why each of the three conditions is required.
param([switch]$DryRun)

$TRAINERS = 'persona_train.py','stt_train.py','ser_train.py'
$targets = @(Get-CimInstance Win32_Process | Where-Object {
    $cl = $_.CommandLine
    $cl -and ($_.Name -match '^(python|uv)') -and ($cl -like "*neiro*") `
        -and ($TRAINERS | Where-Object { $cl -like "*$_*" })
})

if (-not $targets) { "no neiro trainers running"; exit 0 }

foreach ($p in $targets) {
    $short = $p.CommandLine.Substring(0, [Math]::Min(120, $p.CommandLine.Length))
    if ($DryRun) { "WOULD KILL PID $($p.ProcessId) $($p.Name) :: $short" }
    else {
        "killing PID $($p.ProcessId) $($p.Name) :: $short"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
