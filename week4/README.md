# Day 5 — Scikit-learn Pipelines & Tuned Mini-Project

## 📌 Overview

In Day 5, we learned how to build a complete and professional Machine Learning workflow using **Scikit-learn Pipelines**.

The main goal was to combine:

* Data preprocessing
* Feature Engineering
* ColumnTransformer
* Machine Learning model
* GridSearchCV
* Cross-Validation
* Final Test Set evaluation

into one organized and **leak-free Pipeline**.

We worked with the **Bank Marketing dataset** and used `DecisionTreeClassifier` as the classification model.

---

## 🎯 Learning Objectives

By the end of this lesson, we learned how to:

* Build a Pipeline that combines preprocessing and modeling.
* Prevent data leakage using Pipelines.
* Use `ColumnTransformer` for numeric and categorical features.
* Apply different preprocessing methods to different column types.
* Add engineered features to the workflow.
* Tune a complete Pipeline using `GridSearchCV`.
* Use 5-fold Cross-Validation.
* Evaluate the final tuned model on the held-out Test Set.
* Compare the final model performance against a baseline.

---

# 5.1 Why Pipelines Exist

Previously, preprocessing and modeling were performed as separate steps.

This can cause **Data Leakage** if preprocessing is performed incorrectly.

For example, if we scale the entire dataset before splitting it, information from the Test Set can influence the preprocessing process.

A Pipeline solves this problem by chaining preprocessing and modeling into one object.

The Pipeline automatically applies the steps in the correct order.

The main benefit is that preprocessing is learned only from the appropriate training data.

During Cross-Validation, each fold also performs preprocessing using only its training portion.

Therefore, the workflow becomes **leak-free**.

---

# 5.2 Building a Pipeline

A Pipeline allows us to combine the preprocessing step and the Machine Learning model.

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])
```

The Pipeline works sequentially:

```text
Input Data
    ↓
Preprocessing
    ↓
Machine Learning Model
    ↓
Prediction
```

In our project, we used a `ColumnTransformer` as the preprocessing step and `DecisionTreeClassifier` as the model.

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pip = Pipeline([
    ("pre", pre),
    ("model", DecisionTreeClassifier(random_state=42))
])
```

The Pipeline therefore contains:

* `pre` → preprocessing
* `model` → Decision Tree Classifier

---

# 5.3 ColumnTransformer for Mixed Data

Real-world datasets usually contain different types of features.

For example:

* Numeric columns
* Categorical columns

These columns require different preprocessing methods.

We first separated the columns:

```python
catcols = X_train.select_dtypes(include='object').columns
numcols = X_train.select_dtypes(exclude='object').columns
```

Then we created a `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numcols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), catcols)
    ]
)
```

### Numeric Features

Numeric columns are processed using:

```python
StandardScaler()
```

This scales the numeric features.

### Categorical Features

Categorical columns are processed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

This converts categorical values into numerical features.

`handle_unknown='ignore'` allows the encoder to handle categories that were not seen during training.

---

# 5.4 Building the Complete Pipeline

After creating the `ColumnTransformer`, we put it inside the Pipeline together with the model.

```python
pip = Pipeline([
    ("pre", pre),
    ("model", DecisionTreeClassifier(random_state=42))
])
```

The complete workflow becomes:

```text
Raw Data
   ↓
ColumnTransformer
   ├── Numeric → StandardScaler
   └── Categorical → OneHotEncoder
   ↓
DecisionTreeClassifier
   ↓
Prediction
```

This makes the preprocessing and modeling part of one object.

---

# 5.5 Feature Engineering

We added two engineered features based on the work from Day 4.

## Feature 1 — Job Encoding

We calculated the average `deposit` rate for each `job` category using the training data.

```python
datatrain = X_train.copy()
datatrain["deposit"] = y_train

job = datatrain.groupby("job")["deposit"].mean()

X_train["job"] = X_train["job"].map(job)
X_test["job"] = X_test["job"].map(job)
```

The idea was to replace each job category with a numerical value representing its relationship with the target.

For example, a job category with a higher average deposit rate receives a higher value.

The mapping was calculated using the Training Set and then applied to both Training and Test Sets.

---

## Feature 2 — Long Call

We created a feature based on the `duration` column.

First, we calculated the median duration using the Training Set:

```python
threshold = X_train["duration"].median()
```

Then we converted `duration` into a binary feature:

```python
X_train["duration"] = (X_train["duration"] > threshold).astype(int)
X_test["duration"] = (X_test["duration"] > threshold).astype(int)
```

The new feature represents whether the call duration is above the Training Set median.

```text
Duration > median
       ↓
     True  → 1
     False → 0
```

The threshold was calculated from `X_train`, not from the Test Set.

---

# 5.6 Tuning the Full Pipeline with GridSearchCV

`GridSearchCV` allows us to search for the best Hyperparameters automatically.

Instead of manually trying different values, GridSearchCV tests the specified combinations and evaluates them using Cross-Validation.

For our Decision Tree, we tuned parameters such as:

```python
param_grid = {
    "model__max_depth": [3, 5, 7, 10, 15, 30, 20, None]
}
```

The important part is:

```python
"model__max_depth"
```

The `model__` prefix refers to the `model` step inside the Pipeline.

The `max_depth` parameter controls the maximum depth of the Decision Tree.

---

## GridSearchCV

We used:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    pip,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
```

### `cv=5`

This means we used **5-Fold Cross-Validation**.

The Training Set is divided into five folds.

The model is trained and validated multiple times using different combinations of training and validation folds.

This gives a more reliable estimate of model performance during tuning.

### `scoring="accuracy"`

We used **Accuracy** as the evaluation metric during GridSearchCV.

Accuracy measures the percentage of predictions that were correct.

---

# 5.7 Why Double Underscore Is Used

When tuning a parameter inside a Pipeline, we use:

```text
step_name__parameter_name
```

For example:

```python
"model__max_depth"
```

means:

```text
model
  ↓
max_depth
```

The `__` connects the Pipeline step with its parameter.

This allows `GridSearchCV` to access and tune parameters inside the Pipeline.

---

# 5.8 Cross-Validation

We used:

```python
cv=5
```

This means the Training Set is divided into five folds.

The model is trained and validated multiple times.

The purpose is to evaluate different Hyperparameter combinations without using the held-out Test Set.

The Test Set remains untouched during tuning.

---

# 5.9 Final Evaluation

After GridSearchCV finishes, the best Pipeline can be obtained using:

```python
best_model = grid.best_estimator_
```

The final model should then be evaluated on the held-out Test Set.

For example:

```python
from sklearn.metrics import accuracy_score

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", test_accuracy)
```

The Test Set is used only at the end to obtain the final performance of the tuned Pipeline.

---

# 5.10 Baseline

A **Baseline** is a simple reference point used to determine whether our Machine Learning model is actually performing better than a simple strategy.

For a classification problem, a simple baseline can be based on the majority class.

For example:

```python
baseline = y_train.value_counts(normalize=True).max()

print("Baseline Accuracy:", baseline)
```

We can then compare:

```text
Baseline Accuracy
        vs
Tuned Model Test Accuracy
```

If the tuned model performs substantially better than the baseline, this indicates that the model is learning useful patterns from the data.

---

# 🧪 Hands-On Lab: Tuned End-to-End Pipeline

## Step 1 — Build the Pipeline

We created a `ColumnTransformer` to handle numeric and categorical columns differently.

```text
Numeric Columns
      ↓
StandardScaler

Categorical Columns
      ↓
OneHotEncoder

Both
  ↓
Pipeline
  ↓
DecisionTreeClassifier
```

---

## Step 2 — Add Feature Engineering

We added two features from Day 4:

1. Job-based numerical encoding.
2. A duration-based binary feature.

The features were created using information from the Training Set.

---

## Step 3 — Tune the Full Pipeline

We used:

```python
GridSearchCV(
    pip,
    param_grid,
    cv=5,
    scoring="accuracy"
)
```

This allowed us to tune the Pipeline using 5-Fold Cross-Validation.

---

## Step 4 — Final Evaluation

After finding the best Hyperparameters, we evaluated the final tuned Pipeline once on the held-out Test Set.

We then compared its Accuracy against the Baseline.

```text
Baseline
   ↓
Compare
   ↓
Tuned Pipeline Test Accuracy
```

---

# 🔑 Key Takeaways

* A **Pipeline** combines preprocessing and modeling into one object.
* Pipelines help prevent **Data Leakage**.
* `ColumnTransformer` allows different preprocessing for different column types.
* `StandardScaler` is used for numeric features.
* `OneHotEncoder` is used for categorical features.
* Feature Engineering can improve the information available to the model.
* `GridSearchCV` searches for good Hyperparameter combinations.
* `cv=5` performs 5-Fold Cross-Validation.
* Parameters inside a Pipeline are accessed using `step__parameter`.
* The Test Set should remain untouched during tuning.
* The final tuned Pipeline should be evaluated on the Test Set only at the end.
* A Baseline provides a simple reference for judging model performance.

---

# 🏁 Final Workflow

The complete workflow learned in Day 5 can be summarized as:

```text
EDA
 ↓
Train / Test Split
 ↓
Feature Engineering
 ↓
ColumnTransformer
 ├── Numeric → StandardScaler
 └── Categorical → OneHotEncoder
 ↓
Pipeline
 ├── Preprocessing
 └── Model
 ↓
GridSearchCV
 ↓
5-Fold Cross-Validation
 ↓
Best Pipeline
 ↓
Held-Out Test Set
 ↓
Final Metric
 ↓
Compare Against Baseline
```

This workflow provides a **professional, reproducible, and leak-free Machine Learning process**.
