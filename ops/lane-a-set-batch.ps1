# Change lane A's micro-batch at its ACTUAL control surface.
#
# The trainer is not a process somebody started by hand -- it is the Windows
# scheduled task NeiroBoxPersonaTrain, running D:\neiro-data\box_persona_train.bat,
# and Neiro re-chains it when it exits. Killing the process and starting a
# replacement therefore does nothing lasting: the chainer relaunches from the
# .bat within seconds and the old batch size comes straight back. That is
# exactly what happened at 07:10 -- a "shrink" that reported success and left
# --batch 12 running.
#
# So the batch size is edited in the .bat and the TASK is restarted.
# $env:MM_BATCH / $env:MM_ACCUM are supplied by the caller.
$bat = "D:\neiro-data\box_persona_train.bat"
$backup = "$bat.orig"
if (-not (Test-Path $bat)) { "ERR=bat missing"; exit 1 }

# Keep one pristine copy, written once, so restore is exact rather than
# reconstructed from what this script believes the original was.
if (-not (Test-Path $backup)) { Copy-Item $bat $backup }

$text = Get-Content $bat -Raw
$new = $text -replace "--batch\s+\d+", "--batch $env:MM_BATCH" `
             -replace "--accum\s+\d+", "--accum $env:MM_ACCUM"
if ($new -eq $text) { "WARN=no substitution made" }
Set-Content -Path $bat -Value $new -NoNewline

Stop-ScheduledTask -TaskName "NeiroBoxPersonaTrain" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 6
# Any survivor would hold VRAM and race the new one for the same --out.
foreach ($p in @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*persona_train.py*" })) {
  Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 4
Start-ScheduledTask -TaskName "NeiroBoxPersonaTrain"
Start-Sleep -Seconds 15
$now = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*persona_train.py*" })
"RUNS=" + @($now | Where-Object { $_.Name -eq "uv.exe" }).Count
if ($now.Count -gt 0 -and $now[0].CommandLine -match "--batch\s+(\d+).*--accum\s+(\d+)") {
  "NOW_BATCH=" + $Matches[1] + " NOW_ACCUM=" + $Matches[2]
}
