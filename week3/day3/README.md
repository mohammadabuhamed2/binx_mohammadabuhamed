# Logistic Regression - Churn Prediction

## Overview

In this project, we built a **Logistic Regression Classification Model** to predict whether a customer will churn or not.

The goal is to train a classification model, evaluate its performance using different classification metrics, and understand how well the model detects customers who are likely to leave the company.

---

# Step 1: Data Preparation

First, we separate the dataset into:

- **Features (X):** Customer information used by the model for prediction.
- **Target (y):** The value we want the model to predict (Churn).

```python
y = df['Churn']

X = df.drop('Churn', axis=1)
```

The target variable:

- `No` → Customer stays
- `Yes` → Customer leaves (Churn)

---

# Step 2: Convert Categorical Features

Machine learning models require numerical data, so categorical columns must be converted into numerical values.

We use:

```python
X = pd.get_dummies(X, drop_first=True)
```

`drop_first=True` removes one category from each categorical feature to avoid the dummy variable trap.

Example:

Before:

```
Contract:
- Month-to-month
- One year
- Two year
```

After:

```
Contract_One year
Contract_Two year
```

The removed category can be represented when both values are False.

---

# Step 3: Train/Test Split

We split the dataset into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Why do we split the data?

The model learns from the training data and is evaluated on unseen testing data.

- Training data → used for learning.
- Testing data → used to measure model performance.

---

# Step 4: Train Logistic Regression Model

Logistic Regression is a classification algorithm that predicts the probability of belonging to a class.

Although its name contains "Regression", it is used for classification problems.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

The model learns the relationship between customer features and churn behavior.

---

# Step 5: Generate Predictions

After training, we use the model to predict churn labels.

```python
predictions = model.predict(X_test)
```

The output is:

- `0` → No Churn
- `1` → Churn

---

# Step 6: Confusion Matrix

The confusion matrix shows how many predictions were correct and incorrect.

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
```

The confusion matrix contains:

| | Predicted Yes | Predicted No |
|-|-|-|
| Actual Yes | True Positive (TP) | False Negative (FN) |
| Actual No | False Positive (FP) | True Negative (TN) |

### TP (True Positive)

The model predicted churn and the customer actually churned.

### TN (True Negative)

The model predicted no churn and the customer stayed.

### FP (False Positive)

The model predicted churn, but the customer stayed.

### FN (False Negative)

The model predicted no churn, but the customer actually left.

---

# Step 7: Classification Report

We calculate Precision, Recall, and F1-score using:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
```

The result:

```
              precision    recall   f1-score

No              0.84       0.90      0.87

Yes             0.66       0.53      0.59
```

---

# Model Evaluation

## Precision

Precision measures how many customers predicted as churn actually churned.

For the churn class (Yes), the precision is **0.66**.

This means that **66% of customers predicted as churn were actually churn customers**.

---

## Recall

Recall measures how many actual churn customers were successfully detected by the model.

For the churn class (Yes), the recall is **0.53**.

This means that the model detected **53% of customers who actually left the company**.

---

## F1-score

F1-score is the balance between precision and recall.

The F1-score for churn customers is **0.59**.

This indicates that the model has moderate performance in identifying customers who are likely to leave.

---

# Step 8: Precision vs Recall Decision

For churn prediction, **Recall is more important than Precision**.

The reason is that the main goal is to identify as many customers as possible who are likely to leave the company.

Missing a real churn customer (**False Negative**) can cause a business loss because the company loses the opportunity to retain that customer.

Although improving recall may increase some false alarms (**False Positives**), it is usually better to detect more potential churn customers and take retention actions.

---

# Step 9: AUC-ROC Evaluation

AUC-ROC measures the ability of the model to distinguish between churn and non-churn customers.

First, we calculate probabilities:

```python
probabilities = model.predict_proba(X_test)[:,1]
```

Then calculate AUC:

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, probabilities)

print("AUC:", auc)
```

---

# AUC Interpretation

The AUC value ranges between 0.5 and 1.0:

- 0.5 → Random guessing
- 0.7 - 0.8 → Acceptable performance
- 0.8 - 0.9 → Good performance
- 0.9 - 1.0 → Excellent performance

Example:

If the model achieves:

```
AUC = 0.82
```

This indicates that the Logistic Regression model has a good ability to distinguish between customers who are likely to churn and customers who are likely to stay.

The model performs better than random guessing and can rank churn customers with good performance.

---

# Final Conclusion

The Logistic Regression model achieved good overall performance with an accuracy of around 80%.

The model performs very well in identifying customers who stay with the company, but detecting churn customers is more challenging.

Since losing customers has a significant business impact, improving the recall of the churn class would be an important next step.