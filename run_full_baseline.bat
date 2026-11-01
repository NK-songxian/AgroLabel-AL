@echo off
cd /d F:\AgroLabel-AL
E:\anaco\envs\sam_gpu\python.exe -u training\train_seg_baseline.py --epochs 150 --workers 4 --name plantseg34_baseline --project F:/AgroLabel-AL/runs > train_full.log 2>&1
echo EXITCODE=%ERRORLEVEL% > train_full.DONE
