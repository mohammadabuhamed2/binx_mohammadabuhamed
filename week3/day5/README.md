# Bank Marketing — Supervised Learning Mini-Project

## Overview
This project builds a complete end-to-end supervised learning pipeline to predict whether a customer will subscribe to a term deposit (`deposit`: yes/no), using the **Bank Marketing dataset**. The task is a **binary classification** problem.

The pipeline follows: EDA → Preprocessing → Train/Test Split → Modeling → Evaluation.

---

## Step 1: Load the Data
Loaded the dataset (`bank.csv`) using `pandas` and inspected the column names (17 columns, including the target `deposit`).

---

## Step 2: Exploratory Data Analysis (EDA)

- **Data types**: Checked `df.dtypes` — mix of numerical (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) and categorical/text columns (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `deposit`).
- **Structure & nulls**: Used `df.info()` — confirmed 11,162 rows with no missing values.
- **Numerical statistics**: Used `df.describe()` to check distributions (mean, std, min/max) of numerical features.
- **Categorical statistics**: Used `df.describe(include='object')` to check the categorical columns.
- **Target balance**: Checked `deposit` value counts (%) to see how balanced the target classes are.
- **Group comparison**: Compared feature averages grouped by `deposit`. Found that customers who subscribed tend to have:
  - Longer call durations
  - Higher average balance
  - More previous successful contacts
- **Correlation heatmap**: Plotted a correlation heatmap of numerical features. Most correlations are weak; the strongest is between `pdays` and `previous` (0.51, moderate positive correlation). No strong multicollinearity found.
- **Categorical comparison**: Plotted a count plot of `job` vs `deposit` to compare subscription rates across job types.
- **Dropped irrelevant columns**: Removed `day`, `month`, and `contact` since they weren't considered useful for the model.

✅ EDA complete.

---

## Step 3: Preprocessing

1. **Split features and target**:
   - `X` = all columns except `deposit`
   - `y` = `deposit`

2. **Encoded the target**: Mapped `y` from text to numbers (`yes` → 1, `no` → 0) to make it usable by the models.

3. **Train/Test Split**: Split the data using `train_test_split` (80% train / 20% test, `random_state=42`) **before** scaling/encoding, to avoid data leakage.

4. **Separated column types**:
   - `numcols` → numerical columns (to be scaled)
   - `catcols` → categorical columns (to be one-hot encoded)

5. **Built a `ColumnTransformer`**:
   - Numerical columns → `StandardScaler`
   - Categorical columns → `OneHotEncoder(handle_unknown="ignore")`
   
   `handle_unknown="ignore"` ensures that if the test set contains a category not seen during training, it won't cause an error — those columns will just be encoded as all zeros.

6. **Fit/Transform correctly (no leakage)**:
   - `preprocessor.fit_transform(X_train)` → fit *and* transform on training data only
   - `preprocessor.transform(X_test)` → transform only on test data (using the parameters learned from training)

---

## Step 4: Modeling

Trained **three models** plus a **baseline**, all using the preprocessed (scaled + encoded) data:

| Model | Description |
|---|---|
| **Baseline** | `DummyClassifier(strategy="most_frequent")` — always predicts the most common class |
| **Logistic Regression** | Simple linear classifier |
| **SVM** | `SVC(kernel='rbf', probability=True)` |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, random_state=42)` |

---

## Step 5: Evaluation

Compared all models using **accuracy** as the metric:

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.49% |
| **SVM** | **79.76%** |
| Random Forest | 79.49% |
| Baseline (Dummy) | 52.22% |

### Model Selection & Justification

**SVM** was selected as the best model:

1. **Highest accuracy** — SVM scored 79.76%, marginally higher than Logistic Regression and Random Forest (~0.27% difference).
2. **Strong improvement over baseline** — All three models beat the baseline (52.22%) by ~27 percentage points, confirming they learned real patterns rather than guessing.
3. **Close performance across models** — Since all three models perform almost identically (within 0.3%), this suggests the current feature set has reached a performance ceiling, and the small differences between models are likely not statistically significant.

**Note**: Since the accuracy gap between models is so small, accuracy alone may not be sufficient for a confident final decision. Additional metrics (F1-score, precision/recall, confusion matrix) should be checked, especially given potential class imbalance in the target variable.

---

## Tools Used
- **Pandas** — data loading & manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization (heatmap, count plot)
- **Scikit-learn** — preprocessing (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`), modeling (`LogisticRegression`, `SVC`, `RandomForestClassifier`, `DummyClassifier`), evaluation (`accuracy_score`)