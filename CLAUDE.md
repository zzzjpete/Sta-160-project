# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STA 160 (UC Davis) credit risk classification project. Binary classification task predicting loan `Risk` (good/bad) from the German Credit dataset.

## Data

| File | Description |
|------|-------------|
| `training_data.csv` | 1120 rows × 11 cols (includes `Risk` label) |
| `test_data.csv` | 280 rows × 10 cols (no label) |
| `kaggle_sample_solution_file.csv` | Public test labels for 72 rows — used to compute public-test AUC |
| `train_cleaned.rds` / `test_cleaned.rds` | After NA imputation and outlier capping |
| `train_for_model.rds` / `test_for_model.rds` | Fully preprocessed, dummy-encoded, ready for modeling |

Features: `Age`, `Sex`, `Job`, `Housing`, `Saving.accounts`, `Checking.account`, `Credit.amount`, `Duration`, `Purpose`.

**Data path note:** Source code references Windows paths (`C:/Users/Jinxi Zhang/Desktop/project/`). When running locally on Mac, update these to the actual file locations (same directory as the `.Rmd` files).

## Notebook Versions

The active notebook is `试验2.4.Rmd` (latest). Earlier versions are kept for reference:

| File | Status |
|------|--------|
| `实验1.Rmd` | EDA scratch notebook |
| `实验2.2.Rmd` | Word-doc output; first full pipeline |
| `实验2.3.Rmd` | Adds robust scaling and derived features |
| `试验2.4.Rmd` | **Current** — cleaner plots, improved CV, ensemble |

## How to Render

```r
rmarkdown::render("试验2.4.Rmd")           # PDF (xelatex)
rmarkdown::render("试验2.4.Rmd", output_format = "word_document")
```

Run individual chunks interactively in RStudio. The notebook is self-contained — all preprocessing through modeling runs top-to-bottom.

## Pipeline Architecture

1. **EDA** — histograms/boxplots (ggplot2 + patchwork), chi-squared tests for categoricals, t-tests for numerics
2. **Preprocessing**
   - Empty strings → `NA`; key categoricals (`Saving.accounts`, `Checking.account`) → `"unknown"`
   - Outlier capping: IQR-based for `Credit.amount`, `Duration`, `Age`
   - Rare `Purpose` categories (< 20 occurrences) merged into `"other"`
   - Derived features: `LoanRiskScore` and quantile-based bin features (computed from train, applied to test)
   - Robust scaling (median/IQR) — train statistics saved and reused on test
   - Dummy encoding via `fastDummies`; test columns aligned to training feature space
3. **Modeling** — stratified 80/20 split, 5-fold CV via `caret`
   - ElasticNet Logistic Regression (`glmnet`)
   - Random Forest (`ranger`, tuned `mtry`/`splitrule`/`min.node.size`)
   - XGBoost (`xgboost`, tuned `max_depth`/`eta`/`gamma`/`colsample_bytree`)
4. **Evaluation** — AUC (`pROC::roc`), confusion matrix (`caret::confusionMatrix`, positive class = `"bad"`)
5. **Ensemble** — 50/50 probability average of RF + XGB
6. **Feature importance** — Random Forest Gini importance, Top-10 bar chart

## Key R Libraries

```r
library(ggplot2); library(patchwork)   # plots
library(tidyverse); library(dplyr)     # wrangling
library(fastDummies)                   # dummy encoding
library(caret)                         # CV and training
library(glmnet)                        # ElasticNet
library(ranger)                        # Random Forest
library(xgboost)                       # XGBoost
library(pROC)                          # AUC
```
