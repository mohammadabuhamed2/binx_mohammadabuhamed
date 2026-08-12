# Day 4 — Feature Engineering & Hyperparameter Tuning

## 📌 Overview

In Day 4, we learned how to improve Machine Learning models through **Feature Engineering** and **Hyperparameter Tuning**.

The main idea is that improving the input features can sometimes have a bigger impact on model performance than simply choosing a more complex model.

We learned how to:

* Create new features from existing data.
* Transform features using different techniques.
* Understand the difference between parameters and hyperparameters.
* Tune a Machine Learning model systematically using `GridSearchCV`.
* Use Cross-Validation during hyperparameter tuning.
* Understand `best_params_`, `best_score_`, and `best_estimator_`.
* Use `RandomizedSearchCV` when the hyperparameter search space is large.
* Compare a tuned model with an untuned baseline.

---

# 🎯 Learning Objectives

By the end of this lesson, we should be able to:

1. Engineer new features and apply appropriate transformations.
2. Distinguish between **parameters** learned by the model and **hyperparameters** chosen before training.
3. Tune a Machine Learning model systematically using `GridSearchCV` and Cross-Validation.

---

# 🔑 Key Topics

* Why Feature Engineering often beats model choice.
* Feature Creation.
* Binning.
* One-Hot Encoding.
* Datetime Feature Extraction.
* Scaling.
* Parameters vs. Hyperparameters.
* `GridSearchCV`.
* Hyperparameter Grid.
* Cross-Validation.
* `best_params_`.
* `best_score_`.
* `best_estimator_`.
* `RandomizedSearchCV`.

---

# 4.1 Why Feature Engineering Often Beats Model Choice

## What is Feature Engineering?

**Feature Engineering (هندسة الميزات)** is the process of creating, transforming, and selecting the input variables that a Machine Learning model learns from.

The basic idea is:

> Instead of only changing the model, we can improve the information that we give to the model.

For example, suppose we have:

```text
price
area
```

We can create a new feature:

```python
price_per_sqm = price / area
```

Suppose:

```text
price = 100000
area = 100
```

Then:

```text
price_per_sqm = 100000 / 100
              = 1000
```

Now the model has a new piece of information that represents the price relative to the area.

---

## Why is Feature Engineering important?

A Machine Learning model can only learn from the information that we provide.

The general flow is:

```text
Raw Data
   ↓
Feature Engineering
   ↓
Better / More Informative Features
   ↓
Machine Learning Model
   ↓
Prediction
```

If the original Features do not represent important patterns clearly, even a sophisticated model may struggle.

This is why a common professional principle is:

> Better Features can sometimes improve results more than choosing a more complicated model.

---

## Domain Understanding

Feature Engineering is also where **domain understanding** becomes important.

For example, in a banking dataset, we may have:

```text
campaign
previous
```

We can create:

```python
total_contacts = campaign + previous
```

Now we have a Feature that represents the total number of contacts with the customer.

This demonstrates how understanding the meaning of the data can help us create more useful Features.

---

# 4.2 Common Feature Engineering Techniques

There are several common Feature Engineering techniques covered in this lesson:

1. Feature Creation
2. Binning
3. One-Hot Encoding
4. Datetime Extraction
5. Scaling

---

# 1. Feature Creation

## What is Feature Creation?

**Feature Creation (إنشاء الميزات)** means creating a new Feature from one or more existing Features.

Example:

```python
price_per_sqm = price / area
```

Suppose:

```text
price = 100000
area = 100
```

Then:

```text
price_per_sqm = 1000
```

Instead of giving the model only:

```text
price = 100000
area = 100
```

we can also give it:

```text
price_per_sqm = 1000
```

The new Feature can provide a more informative representation of the data.

---

## Example with a Banking Dataset

Suppose we have:

```text
campaign = 4
previous = 2
```

We can create:

```python
total_contacts = campaign + previous
```

The result is:

```text
total_contacts = 6
```

This Feature represents the total number of contacts with the customer.

The important idea is that Feature Creation is not simply adding random columns.

We should create Features that have a logical relationship with the problem.

---

# 2. Binning

## What is Binning?

**Binning (تجميع القيم في فئات)** means converting a continuous numerical variable into groups or ranges.

For example, instead of using:

```text
age = 18
age = 25
age = 40
age = 70
```

we can create groups:

```text
18–30  → young
31–50  → adult
51+    → senior
```

The continuous numerical variable is therefore transformed into categories.

---

## Why use Binning?

Sometimes the exact numerical value is less important than the range it belongs to.

For example:

```text
20
22
25
28
```

could all represent a similar age group.

Binning can therefore provide the model with a simpler representation of the variable.

However, we should always test whether the new representation actually improves model performance.

---

# 3. One-Hot Encoding

## What is One-Hot Encoding?

**One-Hot Encoding (الترميز الأحادي)** converts categorical values into numerical columns so that Machine Learning models can work with them.

Suppose we have:

```text
city
```

with values:

```text
Nablus
Ramallah
Jenin
```

One-Hot Encoding can transform this into:

```text
city_Nablus
city_Ramallah
city_Jenin
```

For example, if the city is Nablus:

```text
city_Nablus    = 1
city_Ramallah  = 0
city_Jenin     = 0
```

If the city is Ramallah:

```text
city_Nablus    = 0
city_Ramallah  = 1
city_Jenin     = 0
```

This allows the Machine Learning model to work with categorical information numerically.

---

# 4. Datetime Extraction

Dates can contain useful information.

Instead of treating:

```text
order_date
```

as one Feature, we can extract information from it.

For example:

```text
order_date
     ↓
day_of_week
month
```

A date such as:

```text
2026-08-12
```

could provide:

```text
day_of_week = Wednesday
month = August
```

These extracted Features can contain patterns that may be useful to the model.

For example, a business might receive more orders on certain days of the week or during certain months.

---

# 5. Scaling

## What is Scaling?

**Scaling (تحجيم الميزات)** means putting numerical Features onto comparable ranges.

Suppose we have:

```text
age = 40
salary = 50000
```

The values have very different scales.

Scaling transforms the numerical Features so that their scales are more comparable.

Two techniques mentioned in this lesson are:

```text
StandardScaler
MinMaxScaler
```

---

## StandardScaler

`StandardScaler` standardizes numerical Features using their mean and standard deviation.

Conceptually, it transforms the values so that the Feature is centered around zero and has a standard deviation close to one.

---

## MinMaxScaler

`MinMaxScaler` transforms numerical values into a specified range.

A common range is:

```text
0 → 1
```

For example:

```text
10
20
30
```

could be transformed into values between:

```text
0 and 1
```

---

# 4.3 Hyperparameters vs. Parameters

This is one of the most important concepts in the lesson.

## Parameters

A **Parameter (المعامل)** is a value learned by the model during Training.

For example, in a regression model, the model learns regression coefficients from the Training data.

The general process is:

```text
Training Data
      ↓
    Model
      ↓
Learn Parameters
```

The model determines these values during the learning process.

---

# Hyperparameters

A **Hyperparameter (المعامل الفائق)** is a value that we set before Training.

The model does not learn the hyperparameter directly from the Training data.

Examples include:

### Random Forest

```text
max_depth
n_estimators
```

### k-NN

```text
k
```

### Ridge

```text
alpha
```

---

# Parameters vs. Hyperparameters

| Parameters                      | Hyperparameters           |
| ------------------------------- | ------------------------- |
| Learned by the model            | Set by us                 |
| Learned during Training         | Chosen before Training    |
| Determined from the data        | We search for good values |
| Example: regression coefficient | Example: `max_depth`      |
| Example: model weights          | Example: `n_estimators`   |

A simple way to remember:

```text
Parameter
    ↓
Model learns it

Hyperparameter
    ↓
We choose it
```

---

# Why do we tune Hyperparameters?

Suppose we have a Random Forest.

We could choose:

```text
max_depth = 5
```

But perhaps:

```text
max_depth = 10
```

performs better.

We could manually test:

```text
5
10
15
20
```

But this becomes slow and unsystematic when there are many hyperparameters and many possible values.

This is why Hyperparameter Tuning is treated as a **Search Problem (مشكلة بحث)**.

We can use:

```text
GridSearchCV
RandomizedSearchCV
```

to automate this process.

---

# 4.4 GridSearchCV

## What is GridSearchCV?

`GridSearchCV` is a Scikit-learn tool that automatically searches through different combinations of hyperparameter values and evaluates them using Cross-Validation.

Instead of manually trying different values, we define the values we want to test and let `GridSearchCV` systematically evaluate them.

The process is:

```text
Hyperparameter Grid
        ↓
Different Combinations
        ↓
Cross-Validation
        ↓
Evaluate Each Combination
        ↓
Compare Results
        ↓
Select Best Combination
```

---

# Hyperparameter Grid

We define the possible values in a dictionary:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}
```

This dictionary is called the **Hyperparameter Grid (شبكة المعاملات الفائقة)**.

It tells `GridSearchCV` which values it should test.

For `n_estimators`, we have:

```text
100
200
```

For `max_depth`, we have:

```text
5
10
None
```

---

# How many combinations?

We have:

```text
n_estimators → 2 values
max_depth    → 3 values
```

Therefore:

```text
2 × 3 = 6 combinations
```

The combinations are:

```text
n_estimators = 100, max_depth = 5
n_estimators = 100, max_depth = 10
n_estimators = 100, max_depth = None

n_estimators = 200, max_depth = 5
n_estimators = 200, max_depth = 10
n_estimators = 200, max_depth = None
```

---

# Cross-Validation

We use:

```python
cv=5
```

This means **5-Fold Cross-Validation (التحقق المتقاطع بخمس طيات)**.

Each hyperparameter combination is evaluated using 5 folds.

Since there are:

```text
6 combinations
```

and:

```text
5 folds
```

the total number of model fits is:

```text
6 × 5 = 30 model fits
```

This is why Grid Search can become computationally expensive.

---

# Complete GridSearchCV Example

The lesson provides:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

print(grid.best_params_)

print(grid.best_score_)

best_model = grid.best_estimator_
```

Now we explain each part.

---

# Importing GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
```

This imports `GridSearchCV` from Scikit-learn.

`GridSearchCV` is responsible for:

```text
Trying hyperparameter combinations
+
Cross-Validation
+
Evaluating the results
+
Finding the best combination
```

---

# Creating the Hyperparameter Grid

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}
```

We create a Python dictionary containing the hyperparameters we want to test.

---

## `n_estimators`

```python
"n_estimators": [100, 200]
```

`n_estimators` represents the number of trees in the Random Forest.

We are asking GridSearchCV to try:

```text
100 trees
200 trees
```

---

## `max_depth`

```python
"max_depth": [5, 10, None]
```

`max_depth` controls the maximum depth of the trees.

We are asking GridSearchCV to try:

```text
5
10
None
```

`None` means that there is no specified maximum depth.

---

# Creating GridSearchCV

```python
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)
```

This creates the GridSearchCV object.

---

## `RandomForestClassifier(random_state=42)`

This is the Machine Learning model that we want to tune.

We use `RandomForestClassifier` because this is a Classification problem.

`random_state=42` makes the random behavior reproducible.

---

## `param_grid`

```python
param_grid
```

contains the hyperparameter values that we want to test.

---

## `cv=5`

```python
cv=5
```

means that we use 5-Fold Cross-Validation.

Every hyperparameter combination is evaluated using five folds.

---

## `scoring="f1"`

```python
scoring="f1"
```

tells GridSearchCV to use **F1-score (درجة F1)** as the metric for comparing the different hyperparameter configurations.

Therefore, the best configuration is the one with the highest Cross-Validated F1-score.

---

# `grid.fit(X_train, y_train)`

```python
grid.fit(X_train, y_train)
```

This starts the search.

GridSearchCV will:

```text
1. Select a hyperparameter combination
        ↓
2. Train the model
        ↓
3. Perform Cross-Validation
        ↓
4. Calculate the F1-score
        ↓
5. Select another combination
        ↓
6. Repeat the process
        ↓
7. Compare all results
        ↓
8. Select the best combination
```

---

# `grid.best_params_`

```python
print(grid.best_params_)
```

`best_params_` returns the hyperparameter combination that achieved the best Cross-Validation score.

For example:

```python
{
    "max_depth": 10,
    "n_estimators": 200
}
```

This means that among all tested combinations, this combination achieved the highest score according to the selected scoring metric.

---

# `grid.best_score_`

```python
print(grid.best_score_)
```

`best_score_` gives the best Cross-Validated score found during the search.

Because we used:

```python
scoring="f1"
```

the returned value is the best Cross-Validated F1-score.

---

# `grid.best_estimator_`

```python
best_model = grid.best_estimator_
```

`best_estimator_` gives us the model configured with the best hyperparameters found by GridSearchCV.

Therefore:

```text
best_params_
    ↓
What were the best settings?

best_score_
    ↓
How well did the best settings perform?

best_estimator_
    ↓
Give me the best model itself.
```

---

# GridSearchCV Flow

```text
                    Hyperparameter Grid
                           ↓
          ┌────────────────────────────────┐
          │ n_estimators: [100, 200]       │
          │ max_depth: [5, 10, None]       │
          └────────────────────────────────┘
                           ↓
                    6 combinations
                           ↓
                 5-Fold Cross-Validation
                           ↓
                   30 model fits
                           ↓
                   Compare F1-scores
                           ↓
                    Best Parameters
                           ↓
                    best_params_
                           ↓
                    best_score_
                           ↓
                    best_estimator_
```

---

# Why Grid Search Can Become Expensive

The computational cost increases as the number of hyperparameter values increases.

In our example:

```text
2 values × 3 values × 5 folds
```

gives:

```text
30 model fits
```

But imagine:

```text
10 values
× 10 values
× 10 values
× 5 folds
```

Then:

```text
10 × 10 × 10 × 5
= 5000 model fits
```

This can take a long time, especially with large datasets or expensive models.

---

# RandomizedSearchCV

For large hyperparameter search spaces, we can use:

**RandomizedSearchCV (البحث العشوائي عن المعاملات الفائقة)**.

Instead of testing every possible combination, `RandomizedSearchCV` samples a subset of combinations.

The difference is:

```text
GridSearchCV
     ↓
Try every combination
     ↓
More exhaustive
     ↓
Can be expensive
```

while:

```text
RandomizedSearchCV
     ↓
Sample some combinations
     ↓
Much fewer model fits
     ↓
Usually faster for large search spaces
```

Therefore, `RandomizedSearchCV` is useful when the hyperparameter grid is very large.

---

# GridSearchCV vs RandomizedSearchCV

| GridSearchCV                         | RandomizedSearchCV                               |
| ------------------------------------ | ------------------------------------------------ |
| Tests every combination              | Tests a subset of combinations                   |
| Exhaustive search                    | Randomized search                                |
| Can become expensive                 | Usually faster for large spaces                  |
| Useful for smaller grids             | Useful for large grids                           |
| More combinations = more computation | Number of sampled combinations can be controlled |

---

# 🔄 Complete Day 4 Conceptual Flow

The complete idea of the lesson is:

```text
Dataset
   ↓
Understand the Features
   ↓
Feature Engineering
   ↓
Create / Transform useful Features
   ↓
Train a Baseline Model
   ↓
Choose Hyperparameters to Tune
   ↓
Create Hyperparameter Grid
   ↓
GridSearchCV
   ↓
Cross-Validation
   ↓
Evaluate Every Combination
   ↓
Find Best Hyperparameters
   ↓
best_params_
   ↓
best_score_
   ↓
best_estimator_
   ↓
Compare Against Baseline
```

---

# 🧠 Complete Summary

Day 4 focused on two major methods for improving Machine Learning models:

1. **Feature Engineering**
2. **Hyperparameter Tuning**

**Feature Engineering** means creating, transforming, or selecting Features so that the model receives more useful information.

Feature Engineering techniques covered in this lesson include:

* Feature Creation
* Binning
* One-Hot Encoding
* Datetime Extraction
* Scaling

Feature Engineering is important because the quality and representation of the input data strongly affect what the model can learn.

A well-engineered Feature can sometimes improve model performance more than simply replacing the model with a more complicated one.

We also learned the difference between **Parameters** and **Hyperparameters**.

Parameters are learned by the model during Training.

Hyperparameters are chosen before Training.

Examples include:

```text
Parameter:
Regression coefficient

Hyperparameters:
Random Forest → max_depth
Random Forest → n_estimators
k-NN → k
Ridge → alpha
```

Choosing good hyperparameters manually can be slow and unsystematic.

`GridSearchCV` solves this by systematically testing combinations of hyperparameter values and evaluating them using Cross-Validation.

For example:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}
```

There are:

```text
2 × 3 = 6 combinations
```

With:

```text
cv=5
```

we get:

```text
6 × 5 = 30 model fits
```

The most important outputs from `GridSearchCV` are:

```python
best_params_
```

The best hyperparameter combination.

```python
best_score_
```

The best Cross-Validated score.

```python
best_estimator_
```

The best model configured with the best hyperparameters.

When the hyperparameter search space becomes very large, `RandomizedSearchCV` can be used because it tests only a subset of combinations and is therefore usually faster.

The main principle of Day 4 is:

```text
Better Features
       +
Better Hyperparameters
       ↓
Potentially Better Model Performance
```
