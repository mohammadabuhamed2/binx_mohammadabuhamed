# 🏠 Day 1 — Supervised Learning Concepts & the Scikit-learn API

## 📌 Overview

In this lesson, I learned the basic concepts of **Supervised Learning** and the standard **Machine Learning workflow** using **Scikit-learn**. I loaded a dataset, handled missing values, separated the features and target, split the data into training and testing sets, and understood why evaluating a model on unseen data is essential.

---

# 🎯 Learning Objectives

- Understand the concept of Supervised Learning.
- Distinguish between Regression and Classification.
- Separate a dataset into Features (X) and Target (y).
- Perform an 80/20 Train/Test Split.
- Understand the importance of evaluating a model on unseen data.

---

# 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

| Library | Purpose |
|----------|---------|
| Pandas | Load and manipulate datasets |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning tools |

---

# 📂 Dataset

**Dataset:** `housing.csv`

The goal of this dataset is to predict the **median house value**.

---

# 🧹 Data Preprocessing

### Load the dataset

```python
df = pd.read_csv("housing.csv")
```

### Handle Missing Values

The `total_bedrooms` column contains missing values, so they were replaced using the median.

```python
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
```

---

# 🔍 Explore the Dataset

Check missing values:

```python
print(df.isna().sum())
```

Display all column names:

```python
print(df.columns)
```

Display data types:

```python
print(df.dtypes)
```

---

# 🎯 Features (X) and Target (y)

### Features

```python
X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
```

`median_house_value` was removed because it is the target column.

`ocean_proximity` was removed because it is a categorical feature and has not been encoded yet.

### Target

```python
y = df["median_house_value"]
```

---

# ✂️ Train/Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Parameters

- `test_size=0.2` → 20% of the dataset is used for testing.
- `random_state=42` → Ensures the same data split every time the code is executed.

---

# 📏 Check Dataset Shapes

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

This confirms that the data was split correctly.

---

# ⚠️ Why Do We Split the Data?

The model should never see the test set during training because this causes **data leakage**. If the model learns from the test data, it may memorize the dataset instead of learning general patterns. This leads to misleadingly high accuracy that does not reflect the model's true performance on unseen data.

---

# 🔄 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Handle Missing Values
      │
      ▼
Explore the Dataset
      │
      ▼
Separate Features (X) and Target (y)
      │
      ▼
Train/Test Split
      │
      ▼
Ready for Model Training
```

---

# 📚 Concepts Learned

- Supervised Learning
- Regression
- Classification
- Features (X)
- Target (y)
- Dataset
- Data Preprocessing
- Missing Values
- Median Imputation
- Train/Test Split
- Training Data
- Testing Data
- Data Leakage
- Scikit-learn

---

# ✅ Conclusion

In this lesson, I learned the first step of every Machine Learning project. I loaded a dataset, handled missing values, explored the data, separated the features and target, performed an 80/20 train-test split, and understood why the test set must never be used during training.