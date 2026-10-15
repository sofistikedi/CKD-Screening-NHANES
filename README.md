# CKD-Screening-NHANES
This project compares machine-learning approaches for identifying adults meeting a KDIGO-based chronic kidney disease (CKD) definition using NHANES 2021–2023 data.
Repository structure
- data/: input workbook and generated intermediate CSV files
- code/01_preprocessing.py: reads and standardizes the encoded workbook
- code/02_cohort_and_target.py: creates the adult cohort and KDIGO-based target
- code/03_models_and_evaluation.py: nested cross-validation, calibration, metrics, bootstrap intervals, decision curve, subgroup analyses, and benchmark models
- results/: generated result tables, figures, and out-of-fold predictions
Target definition
The script labels an adult as CKD-positive when egfr < 60 or albumin_creatinine_ratio >= 30; a participant is assigned a negative label when egfr >= 60 and albumin_creatinine_ratio < 30. Rows without enough information to determine the target are excluded. Only participants aged 20 or older are included.
Important interpretation warning: target leakage
Model 1 is not a valid screening-performance estimate. Its feature list includes egfr and albumin_creatinine_ratio, which are directly used to define ckd_kdigo. The resulting near-perfect AUROC is therefore expected target leakage, not evidence of clinically useful prediction. Treat Model 1 as a leakage/replication baseline only, and do not present its performance as a deployable screening result. Model 2 removes the four listed diagnostic markers; Model 3 uses the predefined lab-free feature list. Verify that no other feature encodes the target or its defining measurements before interpreting results.
Running the analysis
These are Python scripts; GitHub's browser editor stores code but does not execute it. To reproduce the results, run the scripts in a Python environment with the packages in requirements.txt installed, in this order:
1. code/01_preprocessing.py
2. code/02_cohort_and_target.py
3. code/03_models_and_evaluation.py
The original notebook workflow used a Kaggle input path. The repository scripts instead use paths relative to the project folder, so the workbook must be placed in data/ under the exact filename shown above.
Generated outputs
The modeling script writes the fold-level metrics CSV, pooled out-of-fold metrics with bootstrap confidence intervals, a PDF and PNG evaluation figure, and a pickle file containing pooled out-of-fold predictions into results/.
Limitations
This is a research analysis, not a clinically validated diagnostic tool. Performance estimates depend on the cohort definition, data quality, feature availability, validation design, and absence of target leakage. External validation and clinical review would be needed before any real-world use.
