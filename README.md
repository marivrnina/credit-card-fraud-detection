# Credit Card Fraud Detection

An end-to-end machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset.

## Overview

Fraud is rare, which makes it hard to detect. A model that labels every transaction as legit looks highly accurate but catches no fraud at all. This project focuses on handling class imbalance, choosing the right evaluation metrics, and tuning the model for real-world trade-offs between missed fraud and false alarms.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle: real transactions from European cardholders. Most features are PCA components, anonymized for privacy. The data is loaded directly with `kagglehub`.

## Project Workflow

**1. Data Cleaning**
- Checked for missing values
- Removed duplicate transactions to prevent train/test leakage

**2. Exploratory Data Analysis**
- Class distribution and imbalance
- Transaction amount patterns for fraud vs. legit
- Fraud behavior by time of day
- Feature correlations and class separation

**3. Feature Engineering & Prep**
- Log-transformed transaction amount
- Derived hour of day from timestamps
- Stratified train/test split
- Robust scaling fit on training data only

**4. Modeling**
- Baseline model for comparison
- Logistic Regression
- Random Forest
- XGBoost
- Class weighting to handle imbalance
- Stratified cross-validation
- Hyperparameter tuning with randomized search

**5. Evaluation**
- Precision, recall, and PR-AUC instead of accuracy
- Precision-recall curves
- Threshold tuning (F1-optimal and cost-based)
- Confusion matrix analysis
- Feature importance and SHAP explainability

## Key Takeaways

- Accuracy is misleading on imbalanced data
- Tree-based models capture patterns that linear models miss
- The decision threshold is a business choice, not just a technical one
- Explainability matters when a model's decision affects real customers

## Next Steps

- Time-based validation to mimic production
- Resampling techniques like SMOTE
- Probability calibration

## Run It

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · seaborn

---

Built by Mariia Voronina
