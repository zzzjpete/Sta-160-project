# Credit Risk Classification — German Credit Dataset

A machine learning pipeline for binary credit risk prediction (good/bad loan) built in R. The project covers the full model development lifecycle: exploratory analysis, feature engineering, model training with cross-validation, ensemble scoring, and performance evaluation against a held-out test set.

---

## Results

| Model | Validation AUC | Notes |
|-------|---------------|-------|
| Random Forest | **0.9431** | Best single model |
| XGBoost | 0.9190 | Strong on high-dimensional dummies |
| ElasticNet LR | 0.7611 | Baseline linear model |
| RF + XGB Ensemble | — | 50/50 probability blend, evaluated on public test set |

> Validation set: stratified 20% hold-out from 1,120 training records. Public test AUC measured on 72 labeled records from `kaggle_sample_solution_file.csv`.

---

## Dataset

**Source:** German Credit dataset (1,000 records, split 80/20 train/test for this competition)

| Split | File | Rows | Columns |
|-------|------|------|---------|
| Train | `training_data.csv` | 1,120 | 11 (includes `Risk` label) |
| Test | `test_data.csv` | 280 | 10 |
| Public labels | `kaggle_sample_solution_file.csv` | 72 | For AUC benchmarking |

**Features:** `Age`, `Sex`, `Job`, `Housing`, `Saving.accounts`, `Checking.account`, `Credit.amount`, `Duration`, `Purpose`

**Target:** `Risk` — binary (`good` / `bad`), where `bad` is the positive class (default)

---

## Project Structure

```
├── 试验2.4.Rmd              # Main notebook (current version) — full pipeline
├── 实验1.Rmd                # EDA scratch notebook
├── 实验2.2.Rmd              # First full pipeline draft
├── 实验2.3.Rmd              # Added robust scaling and derived features
├── training_data.csv
├── test_data.csv
├── kaggle_sample_solution_file.csv
├── train_cleaned.rds        # Preprocessed training data
├── test_cleaned.rds         # Preprocessed test data
├── train_for_model.rds      # Final model-ready training matrix
└── test_for_model.rds       # Final model-ready test matrix
```

`试验2.4.Rmd` is the active notebook. Earlier `.Rmd` files document the iterative development process.

---

## Pipeline

### 1. Exploratory Analysis
- Distributions for all numeric features (`Credit.amount`, `Duration`, `Age`) with histograms and boxplots
- Chi-squared tests for categorical features vs. `Risk`; t-tests for numeric features
- Bivariate plots: loan amount and duration both show statistically significant separation between good/bad borrowers

### 2. Feature Engineering

**Missing value handling**
- Empty strings treated as `NA`
- `Saving.accounts` and `Checking.account` `NA`s imputed to `"unknown"` category (preserves signal — missingness correlates with risk)

**Outlier capping**
- IQR-based winsorization (3×IQR bounds) applied to `Credit.amount`, `Duration`, and `Age`
- Bounds computed on train set and applied to test set

**Rare category merging**
- `Purpose` categories with fewer than 20 training occurrences consolidated into `"other"`

**Derived features**
| Feature | Description |
|---------|-------------|
| `CreditPerMonth` | `Credit.amount / Duration` — monthly repayment burden |
| `AgeGroup` | Binned age: young / middle / senior / old |
| `CreditGroup` | Quartile-based loan size bins (thresholds from training set) |
| `DurationGroup` | Duration bins: short / medium / long / very_long |
| `HighRiskProfile` | Binary flag: duration > 36 months AND weak checking AND savings accounts |
| `AccountHealth` | Ordinal score (1–4) based on combined checking + savings account strength |

**Log transforms**
- `LogCreditAmt = log1p(Credit.amount)`, `LogDuration = log1p(Duration)` to compress right-skewed distributions

**Robust scaling**
- All numeric features scaled using median and IQR (not mean/std) to reduce outlier influence
- Scaling parameters stored from training set and applied identically to test set

**Dummy encoding**
- One-hot encoding via `fastDummies`; test feature columns aligned to training schema to prevent leakage

### 3. Modeling

All models trained using `caret` with 5-fold stratified cross-validation, optimizing for AUC.

**ElasticNet Logistic Regression** (`glmnet`)
- Grid search over `alpha` ∈ {0, 0.5, 1} and `lambda` sequence
- Provides a regularized linear baseline and feature coefficient interpretability

**Random Forest** (`ranger`)
- Tuned: `mtry`, `splitrule` (gini/extratrees), `min.node.size`
- 500 trees; probability mode enabled for AUC scoring
- Feature importance via Gini impurity (Top 10 visualized)

**XGBoost** (`xgboost`)
- Tuned: `max_depth`, `eta` ∈ {0.05, 0.1}, `gamma`, `colsample_bytree`, `min_child_weight`, `subsample`
- Early stopping (20 rounds) on a held-out watchlist
- Binary logistic objective

### 4. Ensemble
- Final predictions blend Random Forest and XGBoost probability outputs 50/50
- Both models retrained on full training data before generating test predictions

---

## How to Run

Open `试验2.4.Rmd` in RStudio and run chunks sequentially, or knit the whole document:

```r
rmarkdown::render("试验2.4.Rmd")                                    # PDF output
rmarkdown::render("试验2.4.Rmd", output_format = "word_document")   # Word output
```

**Note:** Data file paths in the notebook reference Windows-style paths. Update the `read.csv()` calls at the top to point to your local copies of `training_data.csv` and `test_data.csv`.

**Required packages:**

```r
install.packages(c("ggplot2", "patchwork", "tidyverse", "fastDummies",
                   "caret", "glmnet", "ranger", "xgboost", "pROC"))
```

---

## Key Findings

- **`Checking.account` and `Saving.accounts`** are the strongest predictors — account balance level is highly predictive of default
- **`Duration`** and **`Credit.amount`** are both significant: bad borrowers take larger, longer-term loans on average (~6 more months, ~$1,000 more)
- **`Age`** contributes weakly but significantly — good borrowers average ~2.6 years older
- **`Job`** was not statistically significant (chi-squared p = 0.65) and contributes minimal predictive value
- The `HighRiskProfile` engineered flag captures a specific high-default segment: long-term loans with weak account standing

---

UC Davis — STA 160
