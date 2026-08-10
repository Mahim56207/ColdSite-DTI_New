@echo off
REM ===================================================================
REM  DeepDTA baseline grid: 2 datasets x 4 splits x 3 seeds = 24 runs
REM
REM  Resumable. Every cell writes results\<tag>_deepdta_results.json and
REM  --skip-if-done checks for it first, so if this dies at run 17 you
REM  just run it again and it picks up where it stopped.
REM
REM  Order is seeds-outermost on purpose: after the first pass you
REM  already have one complete seed for every cell, which is enough to
REM  see the shape of the ladder. Seeds 2 and 3 then buy the error bars.
REM  Dataset-outermost would leave you with all of DAVIS and none of
REM  KIBA if it stopped halfway.
REM
REM  Usage:  run_deepdta_grid.bat
REM ===================================================================

setlocal
cd /d "%~dp0"

for %%s in (1 2 3) do (
  for %%d in (davis kiba) do (
    for %%p in (random cold_drug cold_target cold_pair) do (
      echo.
      echo === %%d / %%p / seed %%s ===
      python -m src.model.train_deepdta ^
        --split-dir data\splits\%%d\%%p ^
        --dataset %%d --split %%p ^
        --task regression --seed %%s ^
        --epochs 100 --skip-if-done
      if errorlevel 1 (
        echo.
        echo FAILED on %%d / %%p / seed %%s -- stopping.
        echo Fix it, then re-run this file; finished cells are skipped.
        exit /b 1
      )
    )
  )
)

echo.
echo All 24 DeepDTA cells complete.
echo Check the table with:  python -m src.evaluation.aggregate
endlocal
