@echo off
REM Stop neiro training on the box, and NOTHING ELSE.
REM
REM This machine is a desktop that also runs games, a browser and other
REM projects. An earlier one-liner killed by command-line text alone --
REM `CommandLine -like "*persona_train.py*"` -- and that pattern reaches
REM any process whose arguments happen to contain the string, including
REM the neiro dashboard's own
REM     ssh ... "pgrep -af 'python.*ser_train.py|...'"
REM which is a monitor, not a trainer. Killing by text is how a stop
REM command takes something unrelated with it.
REM
REM Three conditions, all required: a python or uv process, running under
REM D:\neiro, naming one of our trainer scripts.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\neiro-data\stop_training.ps1 %*
