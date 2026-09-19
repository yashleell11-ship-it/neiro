$p = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*persona_train.py*" }) | Select-Object -First 1
if ($p -and $p.CommandLine -match "--batch\s+(\d+).*--accum\s+(\d+)") {
  "BATCH=" + $Matches[1] + " ACCUM=" + $Matches[2] + " EFFECTIVE=" + ([int]$Matches[1] * [int]$Matches[2])
} else {
  "BATCH=? (no trainer running)"
}
