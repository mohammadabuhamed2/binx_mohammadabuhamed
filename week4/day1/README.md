# Day 1 — Train / Validation / Test Splits

## 📌 Overview

In this lesson, we learned how to properly split a Machine Learning dataset into three main sets:

- **Training Set**
- **Validation Set**
- **Test Set**

The main goal is to prevent the **Test Set** from influencing the model development process, so that the final test score remains an honest estimate of how the model will perform on unseen data.

---

# 🎯 Learning Objectives

By the end of this lesson, I should be able to:

- Explain why a **Validation Set** is needed in addition to a **Test Set**.
- Create a correct **three-way split** using `train_test_split()` from Scikit-learn.
- Explain why tuning against the **Test Set** produces misleading results.
- Understand the different roles of Training, Validation, and Test sets.
- Apply a correct three-way split to a dataset.
- Train a model using the Training Set.
- Tune a model using the Validation Set only.
- Evaluate the final model using the Test Set only at the end.

---

# 🧠 The Main Idea

A basic Machine Learning workflow can start with two sets:

```text
Dataset
   │
   ├── Training Set
   │
   └── Test Set
```

However, a problem appears when we repeatedly use the Test Set while developing the model.

For example:

```text
Model A → Test → 80%
Model B → Test → 83%
Model C → Test → 85%
Model D → Test → 87%
```

We might choose Model D because it achieved the highest test score.

However, this means that the Test Set influenced our decisions.

The Test Set is no longer an independent and honest final evaluation.

The professional solution is to use three sets:

```text
Dataset
   │
   ├── Training Set
   ├── Validation Set
   └── Test Set
```

---

# 📚 Key Concepts

## 1. Training Set

The **Training Set** is the data used by the model to learn.

It is used during:

```python
model.fit(X_train, y_train)
```

The model learns relationships between:

```text
Features → Target
```

For example:

```text
Age
Balance
Job
Housing
Loan
   ↓
Churn
```

The Training Set is responsible for teaching the model patterns from the available data.

---

## 2. Validation Set

The **Validation Set** is used during model development.

It is used to:

- Compare different models.
- Select the best model.
- Tune hyperparameters.
- Try different settings.
- Make development decisions.

For example:

```text
Model A → Validation Accuracy = 80%
Model B → Validation Accuracy = 84%
Model C → Validation Accuracy = 82%
```

We would choose Model B because it achieved the best validation performance.

---

## 3. Test Set

The **Test Set** is used for the final evaluation of the model.

It should not be used during:

- Model Selection
- Feature Selection
- Hyperparameter Tuning
- Development

The Test Set should only be used after all decisions have been finalized.

```text
Development
     ↓
Training + Validation
     ↓
Final Model
     ↓
Test
     ↓
Final Score
```

---

# 🏆 The Golden Rule

> **Train to learn, Validation to decide, Test to evaluate.**

```text
Training   → Learn
Validation → Decide
Test       → Final Evaluation
```

---

# ⚠️ The Problem With a Single Test Set

If we repeatedly use the Test Set while developing the model, the test results will influence our decisions.

For example:

```text
Test Accuracy = 80%
       ↓
Modify Model
       ↓
Test Accuracy = 83%
       ↓
Modify Hyperparameter
       ↓
Test Accuracy = 86%
```

The Test Set has now become part of the development process.

This creates a form of:

**Data Leakage**

Information from the Test Set has influenced decisions that should have been made without seeing the Test Set results.

---

# ❌ Why Is This a Problem?

The Test Set is supposed to represent completely unseen data.

We want it to simulate what happens when our model receives new real-world data after development is finished.

If we repeatedly check the Test Set and modify the model based on its results, we are indirectly adapting our decisions to that specific Test Set.

As a result:

```text
High Test Score
       ≠
Guaranteed High Real-World Performance
```

The final test score may become overly optimistic.

Therefore, it may no longer be an **honest estimate** of performance on truly unseen data.

---

# 🔀 The Three-Way Split

The professional solution is to divide the dataset into three sets:

| Set | Purpose | When It Is Used |
|---|---|---|
| Training Set | Train the model | During `.fit()` |
| Validation Set | Tune and select the model | During development |
| Test Set | Final performance evaluation | Once at the end |

A common split is:

```text
60% Training
20% Validation
20% Test
```

The exact percentages are less important than keeping the Test Set untouched until the final evaluation.

---

# 🧩 Three-Way Split Workflow

```text
                     Dataset
                        │
              ┌─────────┴─────────┐
              │                   │
        Development              Test
              │                   │
        ┌─────┴─────┐             │
        │           │             │
      Train      Validation       │
        │           │             │
        │       Tune / Select     │
        │           │             │
        └───────────┘             │
              │                   │
              ▼                   │
         Final Model              │
              │                   │
              └───────────────────┘
                        │
                        ▼
                 Final Evaluation
```

---

# 💻 Creating the Split in Scikit-learn

We use `train_test_split()` from Scikit-learn.

```python
from sklearn.model_selection import train_test_split
```

First, we hold out 20% of the data as the final Test Set:

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Now we have:

```text
80% → X_temp / y_temp
20% → X_test / y_test
```

The `X_temp` and `y_temp` contain the data that will later be divided into Training and Validation.

---

# 🔄 Second Split

We split the remaining 80% into Training and Validation:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)
```

Why `0.25`?

Because we want 25% of the remaining 80% to become Validation:

```text
80% × 25% = 20%
```

The remaining 75% becomes Training:

```text
80% × 75% = 60%
```

Therefore:

```text
Training   = 60%
Validation = 20%
Test       = 20%
```

---

# 📐 Numerical Example

Suppose our dataset contains:

```text
1000 rows
```

### First Split

```text
Test = 20% of 1000
     = 200 rows

Remaining = 800 rows
```

### Second Split

```text
Validation = 25% of 800
           = 200 rows

Training = 75% of 800
         = 600 rows
```

Final result:

```text
Training   = 600 rows = 60%
Validation = 200 rows = 20%
Test       = 200 rows = 20%
```

---

# 🎲 Why Use `random_state`?

`train_test_split()` performs a random split.

Without a fixed `random_state`, the split may change when the code is executed again.

For example:

```python
random_state=42
```

allows us to reproduce the same split.

The number `42` does not have any special Machine Learning meaning.

Other values can also be used:

```python
random_state=10
```

or:

```python
random_state=100
```

The important thing is to keep the value fixed when reproducibility is required.

---

# 🧪 Training, Validation, and Test in Practice

The correct workflow is:

```text
X_train, y_train
       ↓
    Training
       ↓
     Model
       ↓
X_val, y_val
       ↓
 Validation Score
       ↓
 Tune / Select
       ↓
 Final Model
       ↓
X_test, y_test
       ↓
 Final Test Score
```

---

# ⚙️ Model Tuning

**Tuning** means trying different model settings and selecting the configuration that performs best.

For example, with an SVM model, we might try different values of `C`:

```text
C = 0.1 → Validation Accuracy = 78%
C = 1   → Validation Accuracy = 81%
C = 10  → Validation Accuracy = 83%
C = 100 → Validation Accuracy = 82%
```

We choose:

```text
C = 10
```

because it achieved the best validation score.

The important rule is:

```text
Training Set   → Train
Validation Set → Tune
Test Set       → Do NOT touch yet
```

---

# 🚫 Never Tune Against the Test Set

Incorrect workflow:

```text
Model A → Test
Model B → Test
Model C → Test
        ↓
Choose the best
```

Correct workflow:

```text
Model A → Validation
Model B → Validation
Model C → Validation
        ↓
Choose the best
        ↓
Final Model
        ↓
Test once
```

---

# 🧪 Step 2 — Train and Tune Using Validation

For example, when using SVM:

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(C=1)

model.fit(X_trainpre, y_train)

pred = model.predict(X_valpre)

acc_val = accuracy_score(y_val, pred)

print("Validation Accuracy:", acc_val)
```

The model is trained using:

```text
X_trainpre, y_train
```

and evaluated during tuning using:

```text
X_valpre, y_val
```

The Test Set is not used.

---

# 🔧 Preprocessing and the Three-Way Split

When preprocessing is required, we must also respect the separation between Training, Validation, and Test.

For example:

```python
X_trainpre = preprocessor.fit_transform(X_train)

X_valpre = preprocessor.transform(X_val)

X_testpre = preprocessor.transform(X_test)
```

The important difference is:

```text
Training   → fit_transform()
Validation → transform()
Test       → transform()
```

The preprocessor learns its parameters from the Training Set only.

It should not learn from Validation or Test.

---

# ⚠️ Common Mistake

A common mistake is:

```python
model.fit(X_train, y_train)
```

when `X_train` still contains categorical text values such as:

```text
job = "unemployed"
```

Models such as SVM and Decision Tree implementations in Scikit-learn generally require numerical input.

Therefore, categorical features must first be encoded.

After preprocessing:

```text
X_train
   ↓
Preprocessor
   ↓
X_trainpre
   ↓
Model
```

The same preprocessing learned from Training data is then applied to Validation and Test.

---

# 📊 Why One Validation Set Is Not Always Enough

A single Validation Set has its own weakness.

It may happen to be an unusual sample of the dataset.

For example:

```text
Training Data
→ mostly normal customers

Validation Data
→ unusually difficult customers
```

or:

```text
Training Data
→ difficult cases

Validation Data
→ unusually easy cases
```

The resulting validation score may then depend partly on the particular random split.

This is especially problematic with smaller datasets.

---

# 🍀 Luck vs. Signal

A validation result can sometimes reflect:

```text
Luck
```

instead of the actual:

```text
Signal
```

The real signal represents genuine patterns in the data.

If the Validation Set is unusual, we might make model decisions based on a result that is specific to that particular split.

This motivates the use of:

**Cross-Validation**

which is covered in **Day 2**.

---

# 🧪 Hands-On Lab

## Step 1 — Create a 60/20/20 Split

```python
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)
```

Expected proportions:

```text
Training   → 60%
Validation → 20%
Test       → 20%
```

---

## Step 2 — Train and Tune

Train the model using:

```text
Training Set
```

Then evaluate different settings using:

```text
Validation Set
```

For example:

```text
Setting 1 → Validation Score
Setting 2 → Validation Score
Setting 3 → Validation Score
```

Select the best setting based only on Validation performance.

---

## Step 3 — Final Evaluation

After all decisions are finished:

```text
Final Model
     ↓
Test Set
     ↓
Final Score
```

The Test Set should be evaluated only at the end.

---

## Step 4 — Markdown Explanation

If we tuned the model against the Test Set, the Test results would influence our decisions when selecting the model or its hyperparameters.

This means information from the Test Set would leak into the development process.

As a result, the final Test score could be overly optimistic and might no longer be an honest estimate of how the model performs on unseen data.

Therefore:

```text
Validation → Tuning
Test → Final Evaluation
```

---

# 📌 Important Rules

## Rule 1

Never tune against the Test Set.

```text
❌ Test → Tuning
```

Use:

```text
✅ Validation → Tuning
```

---

## Rule 2

Use the Test Set only after all decisions are final.

```text
Development
     ↓
Final Model
     ↓
Test
```

---

## Rule 3

Fit preprocessing only on Training data.

```text
Training   → fit_transform()
Validation → transform()
Test       → transform()
```

---

## Rule 4

The exact split percentages are less important than maintaining the correct methodology.

A common split is:

```text
60 / 20 / 20
```

---

# 🧠 Final Mental Model

Think of the three sets like this:

```text
┌──────────────────────────────────────────┐
│                 DATASET                  │
├──────────────────────────────────────────┤
│                                          │
│  TRAINING                                │
│  "Learn from me"                         │
│                                          │
├──────────────────────────────────────────┤
│                                          │
│  VALIDATION                              │
│  "Use me to make decisions"              │
│                                          │
├──────────────────────────────────────────┤
│                                          │
│  TEST                                    │
│  "Judge the final model once"            │
│                                          │
└──────────────────────────────────────────┘
```

The complete Machine Learning workflow is:

```text
Dataset
   ↓
Split
   ↓
Training ──────────→ Learn
   │
   ↓
Validation ────────→ Tune / Select
   │
   ↓
Final Model
   │
   ↓
Test ──────────────→ Final Evaluation
```

---

# 📖 Complete Summary

The main purpose of a three-way split is to separate **learning**, **decision-making**, and **final evaluation**.

The **Training Set** is used to train the model.

The **Validation Set** is used during development to compare models and tune settings.

The **Test Set** is kept untouched until the end and is used only for the final performance evaluation.

Using the Test Set repeatedly during tuning causes information from the Test Set to influence model development. This can lead to an overly optimistic test score and means that the score may no longer be an honest estimate of performance on unseen data.

A typical split is:

```text
60% Training
20% Validation
20% Test
```

The split can be created using two calls to `train_test_split()`:

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42
)
```

The first split creates the final 20% Test Set.

The second split divides the remaining 80% into:

```text
75% of 80% → 60% Training
25% of 80% → 20% Validation
```

A fixed `random_state` makes the split reproducible.

Finally, a single Validation Set can still be misleading, especially with small datasets, because it may be an unusual sample. This motivates **Cross-Validation**, which will be covered in Day 2.

---

# 🔑 Most Important Concepts

```text
Training Set
→ Used to learn model parameters

Validation Set
→ Used for model selection and tuning

Test Set
→ Used once for final evaluation

Three-Way Split
→ Training / Validation / Test

Data Leakage
→ Information incorrectly influences the development process

Tuning
→ Adjusting model choices/settings using Validation data

Honest Estimate
→ A trustworthy estimate of performance on unseen data

random_state
→ Makes the random split reproducible

train_test_split()
→ Splits data into separate subsets

Cross-Validation
→ Addresses the weakness of relying on a single Validation Set
```

---

# 📚 Important Terminology

| English Term | Meaning |
|---|---|
| Training Set | Training data |
| Validation Set | Validation data |
| Test Set | Test data |
| Three-Way Split | Splitting data into three sets |
| Tuning | Adjusting and improving the model |
| Hyperparameter | Model configuration parameter |
| Model Selection | Choosing the best model |
| Data Leakage | Information incorrectly influencing development |
| Honest Estimate | Trustworthy estimate |
| Unseen Data | Data not seen during development |
| `train_test_split()` | Function used to split data |
| `random_state` | Controls reproducibility |
| Cross-Validation | Cross-validation |
| Development | Model development stage |
| Final Evaluation | Final model evaluation |
| Training | Model training |
| Validation | Model validation |
| Testing | Final testing |

---

# ⭐ Golden Rule

> **Train to Learn → Validation to Decide → Test to Evaluate**