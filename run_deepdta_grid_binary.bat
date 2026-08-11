@echo off
REM ===================================================================
REM  DeepDTA, BINARY task: 2 datasets x 4 splits x 3 seeds = 24 runs
REM
REM  Why a second grid exists
REM  ------------------------
REM  Two of the four audited models cannot do regression at all:
REM
REM    HyperAttentionDTI   nn.Linear(512, 2) + CrossEntropyLoss
REM    MolTrans            BCELoss, reports AUROC / AUPRC
REM
REM  DeepDTA and ColdSite-DTI are regression models. So the grid as first
REM  run produces concordance index for two models and AUROC for the other
REM  two, and those are not comparable numbers. The audit's "do the
REM  interpretable models pay an accuracy cost" question cannot be answered
REM  across a mixed axis.
REM
REM  Binary is the only task all four share, so it is the axis the
REM  cross-model comparison has to use. Thresholds are DeepDTA's own
REM  published values, already verified against the real data:
REM  DAVIS pKd >= 7.0 (8.3% positive), KIBA >= 12.1 (21.0% positive).
REM
REM  The regression grid is still worth keeping as a secondary table for
REM  the two models that support it -- it is what validated the port
REM  against the published DAVIS figures.
REM
REM  Resumable, same as the regression runner.
REM
REM  Usage:  run_deepdta_grid_binary.bat
REM ===================================================================

setlocal
cd /d "%~dp0"

for %%s in (1 2 3) do (
  for %%d in (davis kiba) do (
    for %%p in (random cold_drug cold_target cold_pair) do (
      echo.
      echo === %%d / %%p / seed %%s  [binary] ===
      python -m src.model.train_deepdta ^
        --split-dir data\splits\%%d\%%p ^
        --dataset %%d --split %%p ^
        --task binary --seed %%s ^
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
echo All 24 binary DeepDTA cells complete.
endlocal
