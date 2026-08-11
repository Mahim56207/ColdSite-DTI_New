@echo off
REM ===================================================================
REM  HyperAttentionDTI: 2 datasets x 4 splits x 3 seeds = 24 runs
REM
REM  Binary only -- the vendored head is nn.Linear(512, 2) with
REM  CrossEntropyLoss. That is not a configuration choice, it is what
REM  the model is, and it is why the audit's cross-model accuracy axis
REM  has to be binary. See results.md.
REM
REM  Batch size 32 and lr 5e-5 are the vendored defaults, kept on
REM  purpose: an audit that retrained a subject under a different
REM  recipe would be measuring a model its authors never released.
REM  Slower than DeepDTA's batch 256 as a result -- expect this grid to
REM  take considerably longer.
REM
REM  Resumable: finished cells are skipped.
REM
REM  Usage:  run_hyperattentiondti_grid.bat
REM ===================================================================

setlocal
cd /d "%~dp0"

for %%s in (1 2 3) do (
  for %%d in (davis kiba) do (
    for %%p in (random cold_drug cold_target cold_pair) do (
      echo.
      echo === %%d / %%p / seed %%s  [HyperAttentionDTI, binary] ===
      python -m src.model.train_hyperattentiondti ^
        --split-dir data\splits\%%d\%%p ^
        --dataset %%d --split %%p ^
        --seed %%s --epochs 100 --skip-if-done
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
echo All 24 HyperAttentionDTI cells complete.
endlocal
