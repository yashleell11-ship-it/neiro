@echo off
REM v1 dispatcher for the box: run the v1 task list SERIALLY, one job at
REM a time, and never mistake "finished successfully" for "crashed".
REM
REM persona_train.py's own natural-completion signal is the FINAL
REM adapter it writes at ..\models\persona-lora\adapter -- distinct from
REM the periodic ..\checkpoint\ dir, which exists throughout the whole
REM run and is not evidence of anything finishing. Without checking
REM this, the watchdog would see "process not running" the moment the
REM job finishes on its own and relaunch it via --resume, looping
REM through another full epoch forever -- the exact opposite of "move to
REM the next v1 task when this one is done".
REM
REM Source of truth is ops\box\box_next.bat in the repo. Edit it there
REM and copy it over; a file that exists only on the box is a file
REM nobody can review.

REM Stopping for the day is a flag, not a disabled scheduled task. A task
REM somebody disables by hand at 6pm is a task nobody re-enables, and the
REM machine is then quietly idle for a week; a file in a directory anyone
REM can see is the reversible version of the same decision.
if exist "D:\neiro-data\PAUSE" (
  echo %date% %time% PAUSE file present -- not starting any training >> D:\neiro-data\watchdog.log
  exit /b 0
)

if exist "D:\neiro\models\persona-lora\adapter\adapter_model.safetensors" (
  echo %date% %time% persona LoRA finished -- adapter saved >> D:\neiro-data\watchdog.log
  goto smoke
)

powershell -NoProfile -Command "if (-not (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*persona_train.py*' })) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo %date% %time% persona training not running, relaunching >> D:\neiro-data\watchdog.log
  schtasks /run /tn NeiroBoxPersonaTrain
) else (
  echo %date% %time% persona training alive, ok >> D:\neiro-data\watchdog.log
)
exit /b 0

:smoke
REM Read a sentence out of the finished adapter before moving on. Eleven
REM hours of training had gone by with nobody having read one, and the
REM whole reason this epoch was retrained is a warmth fix nothing has
REM confirmed yet. On CPU, deliberately: a quality check that waits for
REM a free GPU on a machine that trains around the clock never runs.
REM Marker file, not a re-check of the output, so a failed smoke test
REM does not wedge the queue -- it leaves a log and the queue moves on.
if exist "D:\neiro-data\.smoke-english-done" goto stt_english
echo %date% %time% running English persona smoke test on CPU >> D:\neiro-data\watchdog.log
call D:\neiro-data\smoke_cpu.bat
echo done > D:\neiro-data\.smoke-english-done
echo %date% %time% smoke test finished -- transcript in smoke_english.log >> D:\neiro-data\watchdog.log

:stt_english
REM The next v1 task: English STT fine-tune on mls-english +
REM english-dialects-uk. See docs/DECISIONS.md, 2026-09-18, for why this
REM recipe needed generalizing (it was Hindi-only) and the bugs that
REM surfaced doing it. Same natural-completion signal as persona above
REM -- the final adapter, not the periodic checkpoint -- so this branch
REM also won't loop forever once STT is done, it just falls through.
if exist "D:\neiro\models\stt-english-lora\adapter\adapter_model.safetensors" (
  echo %date% %time% English STT LoRA finished -- adapter saved. no next v1 job wired up yet -- box idle by design, not by crash >> D:\neiro-data\watchdog.log
  exit /b 0
)

powershell -NoProfile -Command "if (-not (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*stt_train.py*' })) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo %date% %time% English STT training not running, launching >> D:\neiro-data\watchdog.log
  schtasks /run /tn NeiroBoxSttEnglishTrain
) else (
  echo %date% %time% English STT training alive, ok >> D:\neiro-data\watchdog.log
)
exit /b 0
