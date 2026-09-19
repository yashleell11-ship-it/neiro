@echo off
REM Reads the English persona checkpoint on CPU. Deliberately not CUDA:
REM both cards stay pinned by training, and a quality check that waits
REM for a free GPU never runs at all.
REM
REM Prefers the FINAL adapter when the run has produced one and falls
REM back to the in-progress checkpoint otherwise, so the same file works
REM mid-run (a sanity read) and at the end of the queue (the verdict on
REM the warmth fix).
cd /d D:\neiro
set PYTHONPATH=D:\neiro\src
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=8

set LORA=models\persona-lora\checkpoint
if exist "D:\neiro\models\persona-lora\adapter\adapter_model.safetensors" set LORA=models\persona-lora\adapter

echo ---- smoke start %date% %time% (%LORA%) ---- > D:\neiro-data\smoke_english.log
C:\Users\yashl\.local\bin\uv.exe run --project training python scripts\chat_smoke_test.py --device cpu --checkpoint %LORA% --model models\qwen3.5-4b-safetensors --max-new-tokens 120 >> D:\neiro-data\smoke_english.log 2>&1
echo ---- smoke exit %errorlevel% %date% %time% ---- >> D:\neiro-data\smoke_english.log
