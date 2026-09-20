# Report the 3090 Ti's state in flat lines the watchdog can parse.
#
# RUNS, not processes. One healthy lane is three processes -- uv.exe launches a
# venv python.exe which execs the real python.exe -- so a process count reads 3
# when nothing is wrong and 6 when there genuinely are two runs fighting over
# the same --out. Counting only the processes whose PARENT is not itself part of
# a training command collapses each tree to its launcher, so RUNS is 1 per run.
#
# Counting the trainer scripts rather than python matters too: the desktop
# always has other pythons, and "any python is alive" masks a dead trainer
# forever. It matches ALL of them, not persona_train alone -- the box's job
# queue moves on to stt_train.py when the persona LoRA finishes, and a probe
# that only knew about persona would report a healthy English STT run as
# RUNS=0 and have the watchdog "restart" a machine that was busy.
$gpu = (& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits) -split ','
"FREE=" + $gpu[0].Trim()
"UTIL=" + $gpu[1].Trim()

# The PROCESS must be a python, not merely a process that mentions one.
# Broadening the command-line match from persona_train alone to all three
# trainers introduced a false positive within the day: the box's own
# dashboard shells out to
#   ssh ... "pgrep -af 'python.*ser_train.py|...'"
# to poll the laptop, and that ssh.exe's command line contains all three
# script names. So the watchdog read a stopped box as "alive runs=1" --
# and an "alive" lane is never started, which is the silent version of
# leaving a card idle for a day.
# THREE conditions, not one. This box is not a neiro appliance -- it is a
# desktop with games, a browser and other projects on it, and a match on
# command-line text alone reaches all of them.
#   1. the process is a python or uv          (not any process that
#                                              merely MENTIONS a trainer)
#   2. its command line names neiro           (not another project that
#                                              happens to own a file of
#                                              the same name). In
#                                              practice this is the venv
#                                              path, D:\neiro\training\
#                                              .venv\...\python.exe, so
#                                              the uv.exe LAUNCHER does
#                                              not match and PROCS reads
#                                              1 per run rather than 3.
#                                              That is the number we
#                                              want: the launcher is not
#                                              what trains.
#   3. it names one of our trainer scripts
$TRAINERS = 'persona_train.py','stt_train.py','ser_train.py'
$all = @(Get-CimInstance Win32_Process | Where-Object {
    $cl = $_.CommandLine
    $cl -and ($_.Name -match '^(python|uv)') -and ($cl -like "*neiro*") `
        -and ($TRAINERS | Where-Object { $cl -like "*$_*" })
})
$ids = @($all | ForEach-Object { $_.ProcessId })
$roots = @($all | Where-Object { $ids -notcontains $_.ParentProcessId })
"RUNS=" + $roots.Count
"PROCS=" + $all.Count
