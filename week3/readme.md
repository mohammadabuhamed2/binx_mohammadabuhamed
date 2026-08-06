# 📘 Week 3 — Supervised Learning: Complete Reference Guide

> A full reference covering every concept, method, and function used across Days 1–5: Supervised Learning fundamentals, Linear Regression, Logistic Regression, Classification Model Comparison, and the End-to-End Mini-Project.

---

## 📑 Table of Contents
1. [Day 1 — Supervised Learning Concepts & the Scikit-learn API](#day-1)
2. [Day 2 — Linear Regression (House Price Prediction)](#day-2)
3. [Day 3 — Logistic Regression (Churn Prediction)](#day-3)
4. [Day 4 — Comparing Multiple Classification Models](#day-4)
5. [Day 5 — End-to-End Mini-Project (Bank Marketing)](#day-5)
6. [🔧 Full Methods Reference (A–Z)](#methods-reference)
7. [🧠 Key Concepts Glossary](#glossary)

---

<a name="day-1"></a>
# 📅 Day 1 — Supervised Learning Concepts & the Scikit-learn API

## Learning Objectives
- Understand Supervised Learning.
- Distinguish Regression vs Classification.
- Split a dataset into Features (X) and Target (y).
- Perform an 80/20 Train/Test Split.
- Understand why models must be evaluated on unseen data.

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

| Library | Purpose |
|---|---|
| Pandas | Load and manipulate tabular data (DataFrames) |
| NumPy | Numerical/array operations |
| Matplotlib | Base plotting library |
| Seaborn | Statistical visualization (built on Matplotlib) |
| Scikit-learn | Machine learning algorithms & tools |

## Dataset
`housing.csv` — target: **median house value**.

## Data Preprocessing

### Load the dataset
```python
df = pd.read_csv("housing.csv")
```
`pd.read_csv()` reads a CSV file into a Pandas DataFrame — the standard first step of any project.

### Handle Missing Values
```python
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
```
`.fillna()` replaces missing (`NaN`) values. We use the **median** (not mean) because it's robust to outliers/skewed distributions — a few extreme values won't distort it the way they would the mean.

## Explore the Dataset
```python
print(df.isna().sum())   # count of missing values per column
print(df.columns)        # list of column names
print(df.dtypes)         # data type of each column
```
- `.isna()` returns a boolean mask (True where value is missing); `.sum()` counts the Trues per column.
- `.columns` and `.dtypes` are quick structural checks before touching the data.

## Features (X) and Target (y)
```python
X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df["median_house_value"]
```
- `.drop(columns, axis=1)` removes columns (axis=1 = columns, axis=0 = rows).
- `median_house_value` is dropped from X because it's the **target**, not a feature.
- `ocean_proximity` is dropped because it's **categorical and unencoded** — feeding raw text into a model would break it.

## Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
| Parameter | Meaning |
|---|---|
| `test_size=0.2` | 20% of rows go to the test set, 80% to train |
| `random_state=42` | Fixes the random shuffle so results are reproducible every run |

## Check Dataset Shapes
```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```
`.shape` returns `(rows, columns)` — a quick sanity check that the split worked as expected.

## Why We Split the Data
The model must **never** see the test set during training. If it does, that's **data leakage** — the model effectively "memorizes" instead of learning generalizable patterns, producing misleadingly high scores that collapse on real, unseen data.

## ML Workflow Learned
Load Dataset → Handle Missing Values → Explore Data →
Separate X & y → Train/Test Split → Ready for Model Training
---

<a name="day-2"></a>
# 📅 Day 2 — Linear Regression: House Price Prediction

## Overview
Trained a **Linear Regression** model to predict house prices, interpreted its coefficients, evaluated it with regression metrics, and compared it to a baseline.

## Data Preparation
```python
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Training the Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predc = model.predict(X_test)
```
| Method | What it does |
|---|---|
| `LinearRegression()` | Creates an untrained linear regression object |
| `.fit(X_train, y_train)` | Learns the best-fit coefficients + intercept by minimizing squared error (Ordinary Least Squares) |
| `.predict(X_test)` | Applies the learned equation to generate predictions on new data |

Linear Regression fits a straight-line equation:
y = b0 + b1x1 + b2x2 + ... + bn*xn
where `b0` is the intercept and `b1...bn` are the coefficients.

## Model Coefficients & Feature Importance
```python
print(model.coef_)        # array of coefficients, one per feature
print(model.intercept_)   # the constant term (b0)
```
- **Positive coefficient** → increasing that feature increases the predicted price.
- **Negative coefficient** → increasing that feature decreases the predicted price.
- **Larger |coefficient|** → stronger effect on the prediction (assuming features are on comparable scales).
- Feature importance must always come from the actual `.coef_` values — never guessed or assumed.

## Model Evaluation — Regression Metrics

### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predc)
```
`MAE = average(|actual - predicted|)`
- Same units as the target (dollars, here).
- Treats all errors equally (doesn't punish big misses more than small ones).
- Easy to interpret: "on average, predictions are off by $X."

### Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predc))
```
`RMSE = sqrt(average((actual - predicted)²))`
- Squaring errors **penalizes large mistakes more heavily** than small ones.
- Also in the same units as the target.
- Use RMSE when large errors are especially costly/undesirable.

### R² Score (Coefficient of Determination)
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predc)
```
- Measures the **proportion of variance in y explained by the model** (0 to 1 typically).
- `R² = 1` → perfect predictions.
- `R² = 0` → model is no better than always predicting the mean.
- `R² < 0` → model is **worse** than the mean baseline.

## Baseline Comparison
```python
baseline = y_train.mean()
baseline_pred = np.full(len(y_test), baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
```
- `.mean()` computes the average of the training target.
- `np.full(len(y_test), baseline)` creates an array repeating that single mean value for every test row — this is the "dumbest possible" regression baseline: always predict the average.

### Results
| Model | RMSE |
|---|---|
| Baseline (mean prediction) | 114,485 |
| **Linear Regression** | **71,133** |

Since `RMSE(model) < RMSE(baseline)`, the model has learned genuinely useful patterns — it isn't just guessing the average.

## Conclusion
Linear Regression clearly outperforms the naive baseline, confirming the input features carry real predictive signal for house prices.

---

<a name="day-3"></a>
# 📅 Day 3 — Logistic Regression: Churn Prediction

## Overview
Trained a **Logistic Regression** classifier to predict customer churn, then evaluated it thoroughly using classification-specific metrics: confusion matrix, precision, recall, F1-score, and AUC-ROC.

## Step 1: Data Preparation
```python
y = df['Churn']
X = df.drop('Churn', axis=1)
```
Target meaning: `No` → customer stays, `Yes` → customer leaves.

## Step 2: Convert Categorical Features
```python
X = pd.get_dummies(X, drop_first=True)
```
- `pd.get_dummies()` performs **one-hot encoding**: converts each category into its own 0/1 column.
- `drop_first=True` removes one category per feature to avoid the **dummy variable trap** (perfect multicollinearity — one column being fully predictable from the others). The dropped category is implicitly represented when all its sibling columns are 0.

Example:
Contract: Month-to-month / One year / Two year
↓ becomes
Contract_One year, Contract_Two year
(Month-to-month = both are 0)
## Step 3: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Training data teaches the model; testing data checks it fairly on unseen examples.

## Step 4: Train Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
- Despite the name "Regression," this is a **classification** algorithm.
- It models the **probability** that a sample belongs to class 1, using the sigmoid function to squash a linear combination of features into a 0–1 range.
- `max_iter=1000` increases the number of optimization iterations allowed, preventing "did not converge" warnings on harder datasets.

## Step 5: Generate Predictions
```python
predictions = model.predict(X_test)
```
`.predict()` returns hard class labels (0 or 1) using a default 0.5 probability threshold.

## Step 6: Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
```
| | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | True Positive (TP) | False Negative (FN) |
| **Actual No** | False Positive (FP) | True Negative (TN) |

- **TP**: predicted churn, and they actually churned ✅
- **TN**: predicted stay, and they actually stayed ✅
- **FP**: predicted churn, but they actually stayed ❌ (false alarm)
- **FN**: predicted stay, but they actually churned ❌ (missed churner — costly!)

## Step 7: Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```
## Metric Definitions & Interpretation

### Precision
`Precision = TP / (TP + FP)`
"Of everyone I *predicted* as churn, how many actually churned?"
→ Here: **0.66** → 66% of predicted churners were correct.

### Recall (Sensitivity)
`Recall = TP / (TP + FN)`
"Of everyone who *actually* churned, how many did I catch?"
→ Here: **0.53** → the model caught only 53% of real churners.

### F1-score
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`
The harmonic mean of precision and recall — a single balanced score, useful especially with imbalanced classes.
→ Here: **0.59** → moderate performance on the churn class.

## Step 8: Precision vs Recall — Which Matters More?
For churn prediction, **Recall > Precision**. Missing a real churner (False Negative) is a lost customer and a lost chance to intervene — usually more costly than a few false alarms (lower precision, which just means wasted retention offers).

## Step 9: AUC-ROC
```python
probabilities = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, probabilities)
```
- `.predict_proba(X_test)` returns probability estimates for each class; `[:, 1]` selects the probability of class "1" (churn).
- **ROC-AUC** measures how well the model **ranks** positives above negatives across *all* possible thresholds — not just the default 0.5 cutoff.

| AUC Range | Meaning |
|---|---|
| 0.5 | Random guessing (no skill) |
| 0.7 – 0.8 | Acceptable |
| 0.8 – 0.9 | Good |
| 0.9 – 1.0 | Excellent |

## Conclusion
Logistic Regression achieved ~80% overall accuracy. It's strong at identifying customers who stay but weaker at catching churners. Since losing a customer is costly, **improving recall for the churn class** is the priority next step (e.g., adjusting the classification threshold, class weighting, or resampling).

---

<a name="day-4"></a>
# 📅 Day 4 — Comparing Multiple Classification Models (Telco Churn)

## Overview
Compared **four classification algorithms** — Decision Tree, Random Forest, SVM, k-NN — on the same churn dataset to see which performs best and which features matter most.

## Dataset
- **Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Features:** `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `MonthlyCharges`, `Contract`
- **Target:** `Churn` (Yes/No)

## Step 1: Data Setup & Model Training
- Checked missing values.
- Encoded `Partner`/`Dependents` manually as binary (Yes → 1, No → 0).
- Selected a relevant feature subset.
- One-hot encoded `Contract` via `pd.get_dummies`.
- Split 80/20 (`random_state=42`).

### The Four Models
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
```

| Model | Configuration | How it works |
|---|---|---|
| **Decision Tree** | `max_depth=5` | Splits data into branches using yes/no rules on features, based on the split that reduces impurity (e.g. Gini) most. `max_depth=5` limits tree depth to control overfitting. |
| **Random Forest** | `n_estimators=100` | An **ensemble** of 100 decision trees, each trained on a random subset of data/features (bagging); predictions are averaged/voted for stability and less overfitting. |
| **SVM (Support Vector Machine)** | `kernel='rbf'`, `probability=True` | Finds the boundary (hyperplane) that best separates classes with maximum margin. The `rbf` (Radial Basis Function) kernel lets it draw non-linear boundaries. `probability=True` enables `.predict_proba()` at the cost of extra training time. |
| **k-NN (k-Nearest Neighbors)** | `n_neighbors=5` | Classifies a point by majority vote among its 5 closest training points (by distance) — no explicit "training," just distance lookups at prediction time. |

## Step 2: Model Comparison (F1-Score)
```python
from sklearn.metrics import f1_score
f1_score(y_test, predictions)
```
| Model | F1-score |
|---|---|
| **Decision Tree** | **0.5799** |
| Random Forest | 0.5310 |
| SVM | 0.5229 |
| k-NN | 0.5224 |

F1-score (not accuracy) was chosen here because churn datasets are often imbalanced — F1 balances precision and recall rather than being dominated by the majority class.

## Step 3: Feature Importance (Random Forest)
```python
model.feature_importances_
```
`.feature_importances_` reflects how much each feature reduces impurity across all trees in the forest (averaged) — higher = more influential in the model's decisions.

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

**Interpretation:** `MonthlyCharges` and `tenure` together drive over 82% of the model's decisions. Contract type (especially month-to-month) matters too; demographics barely register.

## Step 4: Final Conclusion
The **Decision Tree** won on F1-score (0.580) — beating even the more complex ensemble and distance-based models.

**Why it likely won:** This dataset seems governed by fairly clear, rule-like relationships that a single tree captures efficiently without heavy overfitting (helped by `max_depth=5`). Distance-based models (SVM, k-NN) are more sensitive to feature scale/space and struggled more here — a real-world example of the **"No Free Lunch" theorem**: no single algorithm is best for every dataset; simplicity can win depending on the data's structure.

## Notes
- `probability=True` in `SVC` adds training overhead — only enable it if you need probability outputs (e.g., for ROC-AUC).
- Always double-check hardcoded file paths before re-running on a different machine.

---

<a name="day-5"></a>
# 📅 Day 5 — End-to-End Mini-Project: Bank Marketing (Term Deposit Prediction)

## Overview
A complete pipeline — **EDA → Preprocessing → Split → Modeling → Evaluation** — predicting whether a customer subscribes to a term deposit (`deposit`: yes/no). This is **binary classification**.

## Step 1: Load the Data
```python
df = pd.read_csv("bank.csv")
df.columns
```
17 columns loaded, including the target `deposit`.

## Step 2: Exploratory Data Analysis (EDA)

### Data types
```python
df.dtypes
```
Mix of numerical (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) and categorical/text (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `deposit`).

### Structure & nulls
```python
df.info()
```
Confirms **11,162 rows, no missing values** — one less preprocessing step needed.

### Numerical statistics
```python
df.describe()
```
Shows count, mean, std, min, quartiles, max for every numeric column — useful for spotting outliers (e.g., `balance` min of -6847 is a real outlier worth noting).

### Categorical statistics
```python
df.describe(include='object')
```
Shows count, unique values, most frequent category, and its frequency for text columns.

### Target balance
```python
df['deposit'].value_counts(normalize=True) * 100
```
- `.value_counts()` counts occurrences of each unique value.
- `normalize=True` converts counts to proportions (then ×100 for percentages) — checks whether the target classes are roughly balanced (important for choosing metrics later).

### Group comparison
```python
df.groupby('deposit').mean(numeric_only=True)
```
`.groupby()` splits the data by `deposit` value and computes the mean of numeric columns for each group — revealed that subscribers tend to have longer call durations, higher balances, and more previous successful contacts.

### Correlation heatmap
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
```
- `.corr()` computes pairwise Pearson correlation coefficients (-1 to 1) between numeric columns.
- `sns.heatmap(annot=True)` visualizes the matrix with numbers overlaid.
- Result: mostly weak correlations; strongest is `pdays` vs `previous` (0.51) — moderate, no serious multicollinearity concern.

### Categorical comparison
```python
plt.figure(figsize=(12,5))
sns.countplot(data=df, x="job", hue="deposit")
```
`sns.countplot(hue=...)` bars the count of each category, split/colored by the target — visually compares subscription rates across job types.

### Dropping irrelevant columns
```python
df = df.drop(["day", "month", "contact"], axis=1)
```
Removed because they weren't judged useful signal for predicting subscription.

✅ **EDA complete.**

## Step 3: Preprocessing

### 1. Split features and target
```python
X = df.drop("deposit", axis=1)
y = df["deposit"]
```

### 2. Encode the target
```python
y = y.map({"yes": 1, "no": 0})
```
`.map()` applies a dictionary lookup element-wise — converts the text labels into numeric 0/1 so models can use them.

### 3. Train/Test Split (before scaling!)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Splitting **before** any scaling/encoding is critical — it ensures the preprocessing steps that follow can be correctly restricted to "train-only fitting."

### 4. Separate column types
```python
catcols = X_train.select_dtypes(include='object').columns
numcols = X_train.select_dtypes(exclude='object').columns
```
`.select_dtypes()` filters columns by data type — here splitting into categorical (`object`) vs numerical (everything else) so each type gets the right transformation.

### 5. Build a ColumnTransformer
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numcols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), catcols)
    ]
)
```
- **`ColumnTransformer`**: applies *different* preprocessing to *different* column subsets in a single object, then concatenates the results into one array — this is the standard, leak-safe way to preprocess mixed data types.
- **`"num"` / `"cat"`**: arbitrary names for each sub-transformer (used internally for reference/inspection — not functionally required to be exactly these words).
- **`StandardScaler()`**: standardizes numeric features to mean = 0, std = 1, via `(x - mean) / std`. Crucial for distance-based/margin-based models (k-NN, SVM, logistic regression) where feature scale affects results.
- **`OneHotEncoder(handle_unknown="ignore")`**: converts categorical columns into binary indicator columns. `handle_unknown="ignore"` means if the test set has a category never seen in training, it won't raise an error — that category's one-hot columns are simply all zero.

### 6. Fit/Transform correctly (no leakage!)
```python
X_train_processed = preprocessor.fit_transform(X_train)  # fit AND transform — train only
X_test_processed = preprocessor.transform(X_test)          # transform ONLY — reuse train's fitted params
```
This is **the single most important rule** in preprocessing: `fit_transform` learns statistics (mean/std for scaling, categories for encoding) **only from training data**, then `transform` (no fit) applies those exact same learned parameters to the test set. Fitting on the full dataset would leak test-set information into training and inflate evaluation results.

## Step 4: Modeling
Trained **three models** plus a **baseline**, all on the preprocessed data:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
```

| Model | Code | Notes |
|---|---|---|
| **Baseline** | `DummyClassifier(strategy="most_frequent")` | Always predicts the majority class — the "floor" any real model must beat |
| **Logistic Regression** | `LogisticRegression()` | Linear probability-based classifier |
| **SVM** | `SVC(kernel='rbf', probability=True)` | Non-linear margin-based classifier |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, random_state=42)` | Ensemble of 100 decision trees |

```python
model1.fit(X_train_processed, y_train)
pred1 = model1.predict(X_test_processed)
# ... same pattern for model2 (SVM), model3 (Random Forest), baseline
```

## Step 5: Evaluation
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred1) * 100
```
`accuracy_score` = fraction of predictions that exactly match the true labels, expressed as a percentage here.

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.49% |
| **SVM** | **79.76%** |
| Random Forest | 79.49% |
| Baseline (Dummy) | 52.22% |

### Model Selection & Justification
**SVM** selected as the best model:
1. **Highest accuracy** — 79.76%, marginally ahead (~0.27%) of Logistic Regression and Random Forest.
2. **Strong improvement over baseline** — all three real models beat the baseline by ~27 points, confirming they learned genuine patterns rather than guessing.
3. **Close performance across models** — the ≤0.3% spread suggests the current feature set has hit a performance ceiling; the difference between models likely isn't statistically meaningful.

**Note:** Given the tiny accuracy gap, additional metrics (F1-score, precision/recall, confusion matrix) should be checked before finalizing — especially since accuracy alone can be misleading if `deposit` classes are imbalanced.

---

<a name="methods-reference"></a>
# 🔧 Full Methods Reference (A–Z)

## Pandas Methods

| Method | Purpose |
|---|---|
| `pd.read_csv(path)` | Load a CSV file into a DataFrame |
| `df.columns` | List all column names |
| `df.dtypes` | Show the data type of each column |
| `df.info()` | Summary: row/column counts, non-null counts, dtypes, memory usage |
| `df.describe()` | Summary statistics (count, mean, std, min, 25/50/75%, max) for numeric columns |
| `df.describe(include='object')` | Summary statistics for categorical/text columns (count, unique, top, freq) |
| `df.isna().sum()` | Count missing values per column |
| `df.fillna(value)` | Replace missing values with a specified value (often `.median()` or `.mean()`) |
| `df.drop(columns, axis=1)` | Remove one or more columns (`axis=1`) or rows (`axis=0`) |
| `df.corr(numeric_only=True)` | Pairwise Pearson correlation matrix between numeric columns |
| `df.groupby(col).mean()` | Group rows by a column's values and compute the mean per group |
| `series.value_counts(normalize=True)` | Count occurrences of each unique value; `normalize=True` gives proportions instead of raw counts |
| `series.map(dict)` | Element-wise replace values according to a dictionary (e.g., text labels → numbers) |
| `df.select_dtypes(include=/exclude=)` | Filter columns by data type (e.g., `include='object'` for text columns) |
| `pd.get_dummies(X, drop_first=True)` | One-hot encode categorical columns; `drop_first=True` avoids the dummy variable trap |
| `df.shape` | Returns `(n_rows, n_columns)` |

## Visualization

| Method | Purpose |
|---|---|
| `plt.figure(figsize=(w,h))` | Create/resize a new figure canvas before plotting |
| `sns.heatmap(matrix, annot=True, cmap=...)` | Visualize a matrix (e.g., correlations) as a colored grid with numeric labels |
| `sns.countplot(data, x=, hue=)` | Bar chart of category counts, optionally split/colored by another column |

## Preprocessing (Scikit-learn)

| Method | Purpose |
|---|---|
| `train_test_split(X, y, test_size=, random_state=)` | Randomly splits data into training and testing subsets |
| `StandardScaler()` | Scales numeric features to mean=0, std=1: `(x - mean) / std` |
| `OneHotEncoder(handle_unknown="ignore")` | Converts categorical values into binary indicator columns; ignores unseen categories at test time instead of erroring |
| `ColumnTransformer(transformers=[...])` | Applies different preprocessing pipelines to different column subsets, then combines the output |
| `.fit_transform(X_train)` | Learns preprocessing parameters from training data AND applies the transformation — use only on train |
| `.transform(X_test)` | Applies previously learned parameters to new data — use on test/validation, never re-fit |

## Models (Scikit-learn)

| Model | Type | Key Idea |
|---|---|---|
| `LinearRegression()` | Regression | Fits a straight-line/hyperplane equation minimizing squared error |
| `LogisticRegression(max_iter=)` | Classification | Models class probability via the sigmoid function over a linear combination of features |
| `DecisionTreeClassifier(max_depth=)` | Classification | Splits data recursively via feature thresholds that best separate classes |
| `RandomForestClassifier(n_estimators=, random_state=)` | Classification (ensemble) | Trains many decision trees on random data/feature subsets and combines their votes |
| `SVC(kernel=, probability=)` | Classification | Finds the maximum-margin boundary between classes; kernel choice (e.g. `rbf`) allows non-linear boundaries |
| `KNeighborsClassifier(n_neighbors=)` | Classification | Classifies by majority vote of the k nearest training points |
| `DummyClassifier(strategy=)` | Baseline | Makes naive predictions (e.g., always the most frequent class) to benchmark real models against |

## Model Methods (common to all Scikit-learn estimators)

| Method | Purpose |
|---|---|
| `.fit(X_train, y_train)` | Train the model on the training data |
| `.predict(X_test)` | Generate hard class/value predictions |
| `.predict_proba(X_test)` | Generate class probability estimates (classification only; needs `probability=True` for SVC) |
| `.coef_` | Learned coefficients (linear/logistic regression) |
| `.intercept_` | Learned intercept/bias term |
| `.feature_importances_` | Relative importance of each feature (tree-based models) |

## Evaluation Metrics — Regression

| Metric | Formula (concept) | Interpretation |
|---|---|---|
| `mean_absolute_error(y_true, y_pred)` | avg(\|actual - predicted\|) | Average magnitude of error, same units as target |
| `mean_squared_error(y_true, y_pred)` (then `np.sqrt()` for RMSE) | avg((actual - predicted)²) | Penalizes large errors more; RMSE returns to original units |
| `r2_score(y_true, y_pred)` | 1 - (SS_res / SS_tot) | Proportion of variance explained; 1=perfect, 0=no better than mean, <0=worse than mean |

## Evaluation Metrics — Classification

| Metric | Formula (concept) | Interpretation |
|---|---|---|
| `accuracy_score(y_true, y_pred)` | correct predictions / total predictions | Overall correctness; misleading on imbalanced data |
| `confusion_matrix(y_true, y_pred)` | TP, TN, FP, FN table | Breaks down exactly where the model is right/wrong |
| `precision_score` / in `classification_report` | TP / (TP + FP) | Of predicted positives, how many were correct |
| `recall_score` / in `classification_report` | TP / (TP + FN) | Of actual positives, how many were caught |
| `f1_score` / in `classification_report` | 2·(P·R)/(P+R) | Balance of precision and recall, good for imbalanced classes |
| `classification_report(y_true, y_pred)` | — | Prints precision, recall, F1, and support for every class at once |
| `roc_auc_score(y_true, y_proba)` | Area under ROC curve | Measures ranking quality across all thresholds; 0.5=random, 1.0=perfect |

---

<a name="glossary"></a>
# 🧠 Key Concepts Glossary

| Term | Definition |
|---|---|
| **Supervised Learning** | Learning a mapping from inputs (X) to known outputs (y) using labeled training data |
| **Regression** | Predicting a continuous numeric value (e.g., price) |
| **Classification** | Predicting a discrete category/label (e.g., churn: yes/no) |
| **Features (X)** | The input variables used to make predictions |
| **Target (y)** | The variable being predicted |
| **Train/Test Split** | Dividing data so the model is trained on one part and fairly evaluated on another, unseen part |
| **Data Leakage** | When information from outside the training data (often the test set) improperly influences training, inflating performance metrics |
| **One-Hot Encoding** | Converting a categorical variable into multiple binary (0/1) columns |
| **Dummy Variable Trap** | Perfect multicollinearity created when all one-hot columns for a feature are kept; solved by dropping one column (`drop_first=True`) |
| **Feature Scaling** | Rescaling numeric features to a comparable range (e.g., via `StandardScaler`) so no feature dominates due to its raw magnitude |
| **Baseline Model** | A simple, naive model (predict the mean/most frequent class) used as a minimum bar that any real model should beat |
| **Overfitting** | A model that fits training data too closely (including its noise) and generalizes poorly to new data |
| **Ensemble Model** | A model that combines multiple sub-models (e.g., Random Forest = many Decision Trees) to improve robustness |
| **Hyperparameter** | A configuration set before training (e.g., `max_depth`, `n_estimators`, `n_neighbors`) — not learned from data |
| **No Free Lunch Theorem** | No single algorithm is universally best; performance depends on the specific dataset/problem |
| **Class Imbalance** | When one class is much more frequent than another, which can make accuracy a misleading metric |

---

# 🧰 Tools Used Across the Week
- **Pandas** — data loading & manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — preprocessing, modeling, and evaluation (see full reference above)

# ✅ Overall Week 3 Conclusion
By the end of the week, the full supervised learning workflow was mastered end-to-end: understanding the problem type (regression vs classification), cleaning and preparing data without leakage, encoding and scaling features correctly, training and comparing multiple models, choosing metrics appropriate to the task, and documenting results against honest baselines — culminating in a complete, narrated mini-project pipeline (Day 5) that mirrors the structure used for the Phase 3 capstone.