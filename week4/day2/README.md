# Day 2 — Cross-Validation

## Overview

In this lesson, I learned how to use **Cross-Validation** to get a more reliable estimate of a Machine Learning model's performance.

The main idea is that evaluating a model using only one train/test split can sometimes give a lucky or unlucky result. Cross-Validation solves this problem by evaluating the model multiple times using different parts of the training data.

In this practical work, I used the **Bank Marketing Dataset** with a **Decision Tree Classifier** and compared the result from a single split with the result from **5-Fold Cross-Validation**.

---

# Learning Objectives

By the end of this lesson, I learned how to:

* Explain how **k-Fold Cross-Validation** produces a reliable performance estimate.
* Use `cross_val_score` to perform Cross-Validation.
* Interpret the **Mean** and **Standard Deviation** of Cross-Validation scores.
* Understand why Cross-Validation is more reliable than depending on a single validation split.
* Understand how **Stratified K-Fold** works for classification.
* Understand why preserving class proportions is important, especially for imbalanced datasets.
* Compare a single-split score with a Cross-Validation estimate.

---

# Key Topics

* Why Cross-Validation is better than a single validation split.
* How **k-Fold Cross-Validation** works.
* How the folds rotate between training and validation.
* `cross_val_score`.
* Mean score.
* Standard deviation.
* Stratified K-Fold for classification.
* Comparison between a single-split evaluation and Cross-Validation.

---

# 1. Why a Single Validation Set Is Not Always Enough

A single validation set has a weakness.

If the validation set happens to be an unusual part of the dataset, the model's score may be unusually high or unusually low.

For example, imagine that a dataset is split like this:

```text
Dataset
   |
   +---- Training Data
   |
   +---- Validation Data
```

The validation data is only one particular subset of the dataset.

If that subset is easier than normal:

```text
Model Score = 90%
```

The score may look excellent.

But if the validation subset is harder:

```text
Model Score = 75%
```

The same model may suddenly look much worse.

This means that the result can depend heavily on **which samples happened to be placed in the validation set**.

This is especially problematic with smaller datasets because every sample has a larger effect on the final result.

Cross-Validation solves this problem by evaluating the model on multiple different validation folds instead of depending on only one split.

---

# 2. Three-Way Split

Before Cross-Validation, the lesson introduced a **Three-Way Split**:

```text
Dataset
   |
   +------------------+
   |                  |
Training          Validation
   |                  |
   +------------------+
            |
          Test
```

The three sets are:

### Training Set

Used to train the Machine Learning model.

### Validation Set

Used to make decisions such as tuning a model setting.

### Test Set

Used for the final evaluation of the model.

The important rule is:

> The Test Set should not be used while tuning the model.

If the test data is repeatedly used to make decisions, the final test score is no longer a fair estimate of how the model performs on unseen data.

---

# 3. Why Cross-Validation Exists

A single validation split can be lucky or unlucky.

Cross-Validation replaces this:

```text
One validation split
        |
        ↓
    One score
```

with:

```text
Multiple validation folds
        |
        ↓
Multiple scores
        |
        ↓
Mean + Standard Deviation
```

This gives us a more stable estimate of model performance.

---

# 4. What Is k-Fold Cross-Validation?

**k-Fold Cross-Validation** is a technique that divides the training data into `k` approximately equal parts called **Folds**.

The model is then trained `k` times.

Each time:

* One Fold is used for validation.
* The remaining `k - 1` Folds are used for training.

For example, if:

```text
k = 5
```

we get:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

The model is trained five times.

---

# 5. How 5-Fold Cross-Validation Works

With `k = 5`, the process looks like this:

| Round | Training Data    | Validation Data |
| ----- | ---------------- | --------------- |
| 1     | Folds 2, 3, 4, 5 | Fold 1          |
| 2     | Folds 1, 3, 4, 5 | Fold 2          |
| 3     | Folds 1, 2, 4, 5 | Fold 3          |
| 4     | Folds 1, 2, 3, 5 | Fold 4          |
| 5     | Folds 1, 2, 3, 4 | Fold 5          |

So the validation Fold rotates:

```text
Round 1 → Fold 1 validates
Round 2 → Fold 2 validates
Round 3 → Fold 3 validates
Round 4 → Fold 4 validates
Round 5 → Fold 5 validates
```

Every data point is used:

* Exactly once for validation.
* Four times for training.

Therefore, no single validation split dominates the final result.

---

# 6. Why Is This Better?

Suppose we have only one validation split:

```text
Accuracy = 80%
```

We do not know whether this result is representative.

With 5-Fold Cross-Validation, we might get:

```text
Fold 1 → 81%
Fold 2 → 82%
Fold 3 → 83%
Fold 4 → 82%
Fold 5 → 83%
```

Now we have much more information.

We can calculate:

```text
Mean
Standard Deviation
```

The Mean tells us the average performance.

The Standard Deviation tells us how much the performance changes from one Fold to another.

---

# 7. Common Values of k

Common choices for `k` are:

```text
k = 5
k = 10
```

In this lesson, the main example uses:

```text
cv = 5
```

This means that the training data is divided into five Folds.

---

# 8. cross_val_score

Scikit-learn provides the function:

```python
from sklearn.model_selection import cross_val_score
```

`cross_val_score` performs Cross-Validation and returns one score for each Fold.

The general structure is:

```python
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)
```

---

# 9. Understanding the Code

## Importing cross_val_score

```python
from sklearn.model_selection import cross_val_score
```

This imports `cross_val_score` from Scikit-learn.

`cross_val_score` is used to perform Cross-Validation automatically.

Instead of manually creating the Folds, training the model, predicting, and calculating the score five times, the function handles this process for us.

---

# 10. Parameters of cross_val_score

The important parameters are:

```python
cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)
```

### `model`

The Machine Learning model that we want to evaluate.

For example:

```python
DecisionTreeClassifier(...)
```

### `X_train`

The training Features.

These are the input variables used by the model.

### `y_train`

The training Target.

This contains the correct class for each training example.

### `cv=5`

This tells Scikit-learn to use 5-fold Cross-Validation.

### `scoring="f1"`

This tells Scikit-learn which metric to calculate.

In this practical work, I used `accuracy` instead:

```python
scoring="accuracy"
```

---

# 11. What Does cross_val_score Actually Do?

When we write:

```python
scores = cross_val_score(
    model,
    X_trainpre,
    y_train,
    cv=5,
    scoring="accuracy"
)
```

we do not need to manually train the model five times.

`cross_val_score` performs the process internally.

Conceptually:

```text
Training Data
      |
      ↓
Split into 5 Folds
      |
      +-------------------------+
      |                         |
      ↓                         ↓
Train on 4 Folds         Validate on 1 Fold
      |
      ↓
Calculate Score
      |
      ↓
Repeat 5 Times
      |
      ↓
[Score 1, Score 2, Score 3, Score 4, Score 5]
```

The function returns the scores from all folds.

---

# 12. Mean of the Scores

We can calculate the average using:

```python
scores.mean()
```

The **Mean** represents the average performance across all Folds.

For example:

```text
Scores:

0.81
0.82
0.83
0.82
0.83
```

The Mean is approximately:

```text
0.822
```

or:

```text
82.2%
```

Therefore, the Mean gives us a more reliable estimate of the model's performance than one single score.

---

# 13. Standard Deviation

We calculate the Standard Deviation using:

```python
scores.std()
```

The **Standard Deviation** tells us how much the scores vary between the Folds.

For example:

```text
Fold 1 → 82%
Fold 2 → 82%
Fold 3 → 83%
Fold 4 → 82%
Fold 5 → 83%
```

The scores are close to each other.

Therefore:

```text
Standard Deviation → Low
```

This means the model is relatively stable.

---

# 14. Mean and Standard Deviation Together

The two values should be interpreted together.

### High Mean + Low Standard Deviation

```text
Mean = High
Std  = Low
```

This is a good situation.

The model performs well and its performance is stable across the Folds.

### High Mean + High Standard Deviation

```text
Mean = High
Std  = High
```

The average performance is good, but the model behaves very differently across different Folds.

Therefore, the high score may partly depend on which data it receives.

---

# 15. My Practical Experiment

In the notebook, I used the same model from the previous work:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
```

The model is a **DecisionTreeClassifier**.

The important setting here is:

```python
max_depth=10
```

This limits the maximum depth of the Decision Tree.

The `random_state=42` makes the model's random behavior reproducible.

---

# 16. Preparing the Dataset

The dataset used in the notebook is the Bank Marketing dataset.

The target column is:

```python
deposit
```

I separated the Features and Target:

```python
X = df.drop("deposit", axis=1)
y = df["deposit"]
```

`X` contains the input Features.

`y` contains the Target that the model tries to predict.

The Target was then converted from text to numbers:

```python
y = y.map({
    "yes": 1,
    "no": 0
})
```

Therefore:

```text
yes → 1
no  → 0
```

This converts the classification Target into numerical labels.

---

# 17. Train/Test Split

The notebook uses:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates:

```text
80% → Training
20% → Test
```

The `random_state=42` ensures that the same split can be reproduced.

---

# 18. Preprocessing

The notebook separates categorical and numerical columns:

```python
catcols = X_train.select_dtypes(include='object').columns
numcols = X_train.select_dtypes(exclude='object').columns
```

Then a `ColumnTransformer` is created:

```python
preprocceser = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numcols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), catcols)
    ]
)
```

Numerical columns are processed using:

```python
StandardScaler()
```

Categorical columns are processed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

The preprocessing is fitted only on the training data:

```python
X_trainpre = preprocceser.fit_transform(X_train)
```

Then the already-fitted preprocessing is applied to the test data:

```python
X_testpre = preprocceser.transform(X_test)
```

This prevents the test data from being used to fit the preprocessing.

---

# 19. Running Cross-Validation

The notebook contains:

```python
num = [2,3,4,5,6,7,8,9,10]

from sklearn.model_selection import cross_val_score

for n in num:
    scores = cross_val_score(
        model,
        X_trainpre,
        y_train,
        cv=n,
        scoring='accuracy'
    )

    print(
        f'the score of {n} number of folds '
        f'the accuracy AVG is: {scores.mean()*100} '
        f'and the std is: {scores.std()}'
    )
```

This code tests different values of `k`.

The values are:

```text
2
3
4
5
6
7
8
9
10
```

Therefore, the model is evaluated using:

```text
2-Fold CV
3-Fold CV
4-Fold CV
5-Fold CV
...
10-Fold CV
```

---

# 20. Why Use a for Loop?

The loop allows us to see how the Cross-Validation results change when we change the number of Folds.

```python
for n in num:
```

Here `n` becomes:

```text
2
3
4
5
...
10
```

Then:

```python
cv=n
```

uses the current value of `n` as the number of Folds.

For example:

```text
n = 2 → cv=2
n = 3 → cv=3
n = 4 → cv=4
n = 5 → cv=5
```

This allows us to compare different Cross-Validation configurations.

---

# 21. The Result from 5-Fold Cross-Validation

The 5-Fold result from the practical work was:

```text
Mean Accuracy = 82.24889507874819%
Standard Deviation = 0.004654958985399483
```

The Mean can be approximately written as:

```text
82.25%
```

The Standard Deviation is approximately:

```text
0.00465
```

or about:

```text
0.47 percentage points
```

The low Standard Deviation means that the scores across the five Folds were very close to each other.

Therefore, the model's performance was relatively stable across the Folds.

---

# 22. Comparison with Day 1

The single-split Accuracy from Day 1 was:

```text
80.70%
```

The 5-Fold Cross-Validation Mean was:

```text
82.25%
```

So the comparison is:

| Evaluation Method               | Accuracy |
| ------------------------------- | -------: |
| Day 1 — Single Split            |   80.70% |
| Day 2 — 5-Fold Cross-Validation |   82.25% |

The difference is approximately:

```text
82.25% - 80.70% = 1.55 percentage points
```

The Cross-Validation result is slightly higher.

---

# 23. What Does This Difference Mean?

It does NOT mean that Cross-Validation improved the model.

Cross-Validation is an evaluation technique.

It does not automatically make the model better.

Instead:

```text
Day 1
↓
Evaluate on one split
↓
80.70%
```

while:

```text
Day 2
↓
Evaluate across 5 different folds
↓
Mean = 82.25%
```

The difference exists because the two methods evaluate the model using different subsets and different evaluation procedures.

Therefore, Cross-Validation gives us more information about how the model behaves across different parts of the training data.

---

# 24. Why the Low Standard Deviation Matters

The Standard Deviation was:

```text
0.00465
```

This is relatively low.

That means the model's Accuracy did not change dramatically between the different Folds.

Conceptually:

```text
Fold 1 → Similar score
Fold 2 → Similar score
Fold 3 → Similar score
Fold 4 → Similar score
Fold 5 → Similar score
```

Therefore, the model appears relatively stable across these Folds.

---

# 25. Stratified K-Fold

For classification problems, especially imbalanced classification problems, ordinary K-Fold can accidentally create Folds with different class proportions.

For example, suppose the dataset contains:

```text
Class 0 → 70%
Class 1 → 30%
```

We want each Fold to have approximately the same distribution:

```text
Fold 1 → 70% / 30%
Fold 2 → 70% / 30%
Fold 3 → 70% / 30%
Fold 4 → 70% / 30%
Fold 5 → 70% / 30%
```

This is the purpose of **Stratified K-Fold**.

---

# 26. Why Stratification Matters

Imagine that one Fold accidentally contains:

```text
90% Class 0
10% Class 1
```

while another contains:

```text
60% Class 0
40% Class 1
```

These Folds are very different.

The model is therefore being evaluated on validation sets with different class distributions.

This can make the evaluation less reliable.

Stratification reduces this problem by preserving the original class balance approximately in each Fold.

---

# 27. Stratified K-Fold in the Current Code

The current code uses:

```python
scores = cross_val_score(
    model,
    X_trainpre,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

Because the model is a **Classifier**, Scikit-learn automatically uses **StratifiedKFold** when an integer value such as `cv=5` is provided.

Therefore, no code change is required to use stratified folds in this classification task.

The `5` means:

```text
Number of Folds = 5
```

It does not mean "Stratified".

If we change it:

```python
cv=3
```

we get 3 stratified folds.

If we change it:

```python
cv=10
```

we get 10 stratified folds.

The important point is that the model is a classifier.

---

# 28. K-Fold vs Stratified K-Fold

### K-Fold

Divides the data into Folds without specifically trying to preserve class proportions.

### Stratified K-Fold

Divides the data while trying to preserve the class proportions in each Fold.

For Classification, Stratified K-Fold is particularly useful because the class distribution can strongly affect evaluation.

---

# 29. Complete Cross-Validation Flow

The complete idea can be represented as:

```text
Original Dataset
       |
       ↓
Train / Test Split
       |
       +------------------+
       |                  |
   Training Data        Test Data
       |
       ↓
Cross-Validation
       |
       ↓
+------+------+------+------+------+
|Fold 1|Fold 2|Fold 3|Fold 4|Fold 5|
+------+------+------+------+------+
       |
       ↓
Train and Validate 5 Times
       |
       ↓
5 Accuracy Scores
       |
       ↓
+----------------------+
| Mean Accuracy        |
| Standard Deviation   |
+----------------------+
       |
       ↓
Reliable Performance Estimate
```

The Test Set remains separate and is not used during Cross-Validation.

---

# 30. Cross-Validation vs Single Split

| Single Split                             | Cross-Validation                            |
| ---------------------------------------- | ------------------------------------------- |
| Uses one split                           | Uses multiple folds                         |
| Produces one evaluation result           | Produces multiple scores                    |
| Can be affected by a lucky/unlucky split | Less dependent on one split                 |
| Gives less information about stability   | Mean and Std show performance and stability |
| Faster                                   | More computationally expensive              |
| Simple                                   | More reliable for estimating performance    |

---

# 31. Important Difference: Evaluation vs Training

Cross-Validation is mainly being used here to **evaluate** the model.

When we call:

```python
cross_val_score(model, X_trainpre, y_train, cv=5)
```

Scikit-learn internally trains and evaluates copies of the model for the different Folds.

Conceptually:

```text
Fold 1:
Train → Folds 2,3,4,5
Validate → Fold 1

Fold 2:
Train → Folds 1,3,4,5
Validate → Fold 2

...

Fold 5:
Train → Folds 1,2,3,4
Validate → Fold 5
```

So we do not need to manually call:

```python
model.fit()
```

before using `cross_val_score` just for the purpose of Cross-Validation.

---

# 32. Important Rule About the Test Set

The Test Set should remain untouched during model selection and tuning.

The general idea is:

```text
Training Data
      |
      ↓
Cross-Validation
      |
      ↓
Choose / evaluate model
      |
      ↓
Final Model
      |
      ↓
Test Set
      |
      ↓
Final Evaluation
```

The Test Set should not repeatedly influence decisions about the model.

If we repeatedly look at the Test Set and change the model based on its score, we are effectively allowing the model-development process to adapt to the Test Set.

Then the Test Set is no longer a completely independent final evaluation.

---

# 33. Hands-On Lab — Step 1

The first step of the Cross-Validation lab was:

> Take a Week 3 model and evaluate it with 5-fold cross-validation using `cross_val_score`.

In the notebook, I used the Decision Tree model:

```python
model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
```

Then:

```python
scores = cross_val_score(
    model,
    X_trainpre,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

This evaluates the model using five Folds.

---

# 34. Hands-On Lab — Step 2

The second step was to report:

* Mean
* Standard Deviation

The code is:

```python
print(scores.mean())
print(scores.std())
```

The Mean represents the average Accuracy.

The Standard Deviation represents how much the Accuracy changes between Folds.

In this practical work:

```text
Mean Accuracy ≈ 82.25%
Std ≈ 0.00465
```

---

# 35. Hands-On Lab — Step 3

The third step was to compare the Cross-Validation estimate with the single-split score from Day 1.

The results were:

```text
Day 1:
80.70%

Day 2:
82.25%
```

The difference is approximately:

```text
1.55 percentage points
```

The Cross-Validation estimate is slightly higher.

The important conclusion is that the Day 1 score came from one particular split, while the Day 2 estimate is based on several different folds.

The low Standard Deviation also indicates that the model's performance was relatively stable across the five folds.

---

# 36. Hands-On Lab — Step 4

The fourth step was to confirm that stratified folds are being used.

Because this is a classification task and the model is a classifier, Scikit-learn automatically uses Stratified K-Fold when:

```python
cv=5
```

is passed to `cross_val_score`.

This is important because stratification maintains approximately the same class proportions across the Folds.

Therefore, the evaluation is more reliable, especially when the classes are imbalanced.

No change to the existing Cross-Validation code is required.

---

# 38. Most Important Ideas From Day 2

The most important concept is:

> A single split can be lucky or unlucky.

Cross-Validation reduces our dependence on one particular split.

Instead of:

```text
One split → One score
```

we use:

```text
Multiple folds → Multiple scores → Mean + Standard Deviation
```

The **Mean** tells us approximately how well the model performs on average.

The **Standard Deviation** tells us how stable that performance is across the Folds.

For Classification, **Stratified K-Fold** is important because it preserves the class proportions across the Folds.

---

# 39. Important Terminology

| Term                    | Meaning                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| Cross-Validation        | A technique for evaluating a model using multiple data splits          |
| k-Fold Cross-Validation | Divides data into `k` Folds and rotates the validation Fold            |
| Fold                    | One part of the data used in Cross-Validation                          |
| Training Fold           | The Folds used to train the model                                      |
| Validation Fold         | The Fold used to evaluate the model in one round                       |
| Mean                    | Average performance across all Folds                                   |
| Standard Deviation      | Measures how much the scores vary between Folds                        |
| Stratified K-Fold       | K-Fold that preserves class proportions                                |
| Class Imbalance         | A situation where classes have very different numbers of samples       |
| `cross_val_score`       | Scikit-learn function used to perform Cross-Validation                 |
| `cv`                    | Parameter controlling the Cross-Validation strategy or number of Folds |
| `scoring`               | Parameter specifying the evaluation metric                             |
| Accuracy                | Percentage of correct predictions                                      |
| Classifier              | A model used for Classification                                        |
| Validation Set          | Data used to evaluate/tune a model during development                  |
| Test Set                | Data reserved for final evaluation                                     |

---

# Complete Summary

Cross-Validation is used because a single validation split can give a misleading result depending on which samples happen to be included in that split.

In **k-Fold Cross-Validation**, the training data is divided into `k` Folds. The model is trained `k` times. In every round, one Fold is used for validation while the remaining `k-1` Folds are used for training.

With 5-Fold Cross-Validation:

```text
5 Folds
↓
5 training/validation rounds
↓
5 scores
↓
Mean + Standard Deviation
```

The Mean gives an overall estimate of model performance, while the Standard Deviation tells us how stable that performance is across the Folds.

In the practical experiment, the single-split Accuracy from Day 1 was:

```text
80.70%
```

while the 5-Fold Cross-Validation Mean was:

```text
82.25%
```

with:

```text
Std = 0.00465
```

The Cross-Validation result was slightly higher, but this does not mean that Cross-Validation improved the model. It means that the model was evaluated using multiple different folds instead of only one split.

For Classification, especially when classes are imbalanced, **Stratified K-Fold** is important because it preserves approximately the same class proportions in every Fold.

When using a Classification model with:

```python
cross_val_score(
    model,
    X_trainpre,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

Scikit-learn automatically uses Stratified K-Fold for the classifier.

The central lesson is:

> **A single split gives one view of model performance, while Cross-Validation gives multiple views and combines them into a more reliable and stable performance estimate.**
