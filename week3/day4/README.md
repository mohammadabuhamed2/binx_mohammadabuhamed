# Telco Customer Churn Prediction — Hands-On Lab

## Overview
This lab compares four classification models (Decision Tree, Random Forest, SVM, k-NN) to predict customer churn using the **Telco Customer Churn** dataset. The goal is to identify which model performs best and understand which features most influence churn.

## Dataset
- **Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Features used:** `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `MonthlyCharges`, `Contract`
- **Target:** `Churn` (Yes/No)

## Project Structure

### Step 1 — Data Setup & Model Training
- Loaded the dataset and checked for missing values.
- Encoded `Partner` and `Dependents` as binary (Yes → 1, No → 0).
- Selected a subset of relevant features.
- One-hot encoded categorical variables (`Contract`) using `pd.get_dummies`.
- Split data into train/test sets (80/20, `random_state=42`).
- Trained four models:
  | Model | Configuration |
  |---|---|
  | Decision Tree | `max_depth=5` |
  | Random Forest | `n_estimators=100` |
  | SVM | `kernel='rbf'`, `probability=True` |
  | k-NN | `n_neighbors=5` |

### Step 2 — Model Comparison (F1-Score)
| Model | F1-score |
|---|---|
| **Decision Tree** | **0.5799** |
| Random Forest | 0.5310 |
| SVM | 0.5229 |
| k-NN | 0.5224 |

### Step 3 — Feature Importance (Random Forest)
| Feature | Importance |
|---|---|
| MonthlyCharges | 0.539 |
| tenure | 0.285 |
| Contract_Month-to-month | 0.086 |
| Contract_Two year | 0.032 |
| Contract_One year | 0.015 |
| SeniorCitizen | 0.015 |
| Dependents | 0.015 |
| Partner | 0.013 |

**Interpretation:**
`MonthlyCharges` (53.9%) and `tenure` (28.5%) are the two most important predictors, together accounting for over 82% of the model's decision weight. Contract type (especially Month-to-month) also plays a notable role, while demographic features (`Partner`, `Dependents`, `SeniorCitizen`) have minimal impact.

### Step 4 — Final Conclusion
The **Decision Tree** achieved the best F1-score (0.580), outperforming Random Forest, SVM, and k-NN.

**Why it likely won:**
Despite ensemble models generally being more robust, this dataset seems governed by clear, direct rules that a single Decision Tree can capture efficiently without heavy overfitting. Distance-based models (SVM, k-NN) struggled more with the feature space — an example of the "no free lunch" principle: simpler models can outperform complex ones depending on the data distribution.

## Requirements
```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run
1. Place the dataset CSV in your working directory.
2. Update the file path in the notebook (`pd.read_csv(...)`) to match your local path.
3. Run all cells sequentially.

## Notes
- The dataset path in the original notebook is hardcoded to a local Windows path — update it before running elsewhere.
- `probability=True` in SVC is used to enable probability estimates but slightly increases training time.