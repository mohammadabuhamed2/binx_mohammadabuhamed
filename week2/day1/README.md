# Day 1 — Descriptive Statistics 📊

## Overview
In the first day of the data analysis journey, I learned the fundamentals of **Descriptive Statistics** and how to summarize and understand datasets before building Machine Learning models.

Descriptive statistics helps us understand the **center, spread, and distribution** of data, which is an essential step in **Exploratory Data Analysis (EDA)** and model evaluation.

---

# Learning Objectives 🎯

During this day, I learned how to:

- Calculate **Mean, Median, and Mode**.
- Choose the appropriate measure of central tendency depending on the dataset.
- Calculate and interpret:
  - Variance
  - Standard Deviation
  - IQR (Interquartile Range)
- Understand how **outliers** affect statistical measurements.
- Analyze a real dataset using Python.

---

# Topics Covered 📚

## 1. Why Statistics Comes Before Modeling

Machine Learning models learn patterns from data.  
Before creating any model, we need to understand the dataset:

- Where is the data centered?
- How spread out is it?
- Are there unusual values (outliers)?
- Is the distribution balanced or skewed?

Descriptive statistics provides the tools needed to answer these questions.

---

# 2. Measures of Central Tendency

Central tendency describes the typical or central value in a dataset.

| Measure | Description | When to Use |
|---------|-------------|-------------|
| Mean | Arithmetic average of values | Symmetric data without extreme outliers |
| Median | Middle value after sorting data | Data with outliers or skewed distributions |
| Mode | Most frequent value | Categorical data or finding the most common value |

### Important Insight:
- **Mean is sensitive to outliers** because extreme values can pull the average.
- **Median is more robust** because it is not affected significantly by extreme values.

Example:

```python
import numpy as np

data = np.array([10, 12, 12, 13, 100])

print(np.mean(data))
# 29.4

print(np.median(data))
# 12.0
```
## 3. Measures of Spread

Spread describes how far the values are distributed around the center of the dataset.

| Measure | Meaning |
|---|---|
| Range | The difference between the maximum and minimum values in the dataset. It is simple to calculate but highly affected by outliers. |
| Variance | Measures the average squared distance of each value from the mean. |
| Standard Deviation | The square root of variance. It shows how much data is spread around the mean using the same units as the original data. |
| IQR (Interquartile Range) | Represents the range of the middle 50% of the data (Q3 - Q1). It is more robust and less affected by outliers. |

## 4. Percentiles and Quartiles

Percentiles divide the dataset into sections based on the position of values.  
They help us understand the distribution of data and identify the location of values within the dataset.

| Quartile | Meaning |
|---|---|
| Q1 (25th Percentile) | 25% of the data values are below this point. |
| Q2 (50th Percentile) | Represents the median value, where 50% of the data is below and 50% is above. |
| Q3 (75th Percentile) | 75% of the data values are below this point. |

The Interquartile Range (IQR) is calculated as:

```python
IQR = Q3 - Q1
```
# Practical Work 🚢 Titanic Dataset

I applied descriptive statistics on the **Titanic dataset** using the **Age** column.

The goal was to understand the distribution of passenger ages by calculating different statistical measurements.

---

## Steps Performed:

1. Load the dataset using **Pandas**.
2. Select the **Age** column.
3. Remove missing values from the dataset.
4. Convert the data into a **NumPy array**.
5. Calculate the following statistical measures:

- Mean
- Median
- Mode
- Standard Deviation
- IQR (Interquartile Range)

---

# Implementation 💻

```python
import pandas as pd
import numpy as np

# Load dataset
titanic = pd.read_csv(
    r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week2\day1\tested.csv"
)

# Select Age column
age = titanic['Age']

# Remove missing values
age = age.dropna()

# Convert to NumPy array
npage = np.array(age)

# Calculate Mean
print(np.mean(npage))

# Calculate Median
print(np.median(npage))

# Calculate Mode
print(age.mode())

# Calculate Standard Deviation
print(np.std(age))

# Calculate IQR
q1, q3 = np.percentile(age, [25, 75])

IQR = q3 - q1

print(IQR)
```
# Results Interpretation 📈

After analyzing the **Age** column, we can understand the main characteristics of passenger ages using descriptive statistics.

| Statistic | Meaning |
|---|---|
| Mean | Represents the average passenger age in the dataset. |
| Median | Represents the middle passenger age after sorting all ages. |
| Mode | Represents the most common age that appears in the dataset. |
| Standard Deviation | Shows how much passenger ages are spread around the mean. |
| IQR (Interquartile Range) | Shows the range of the middle 50% of passenger ages and helps identify outliers. |

### Conclusion:

The calculated statistical measures help us understand the center and spread of passenger ages.  
By comparing the **Mean** and **Median**, we can determine whether the data contains outliers and choose the most representative value for the dataset.