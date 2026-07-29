# Exploratory Data Analysis (EDA) – Part 1: Distributions & Outliers

This repository contains my hands-on practice for **Exploratory Data Analysis (EDA)** using the **Titanic Dataset**. The goal was to understand data distributions, detect outliers, explore categorical variables, and document insights before building Machine Learning models.

---

# Topics Covered

- Exploratory Data Analysis (EDA)
- Numerical vs Categorical Variables
- Histograms
- Box Plots
- Outlier Detection
- IQR (Interquartile Range) Method
- Count Plots
- Class Imbalance
- Data Interpretation using Markdown

---

# Dataset

- Titanic Dataset

---

# Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

---

# Step 1 — Histogram Analysis

Created histograms for numerical variables to understand their distributions.

### Numerical Variables

- Age
- Fare

### Learned

- Distribution of numerical data
- Right-skewed and left-skewed distributions
- Data spread
- Frequency distribution
- Identifying unusual values visually

---

# Step 2 — Box Plot Analysis

Created box plots for numerical variables.

### Used

- Fare

### Learned

- Median
- Quartiles (Q1, Q2, Q3)
- IQR
- Whiskers
- Visual detection of outliers

---

# Step 3 — Outlier Detection Using IQR

Implemented the IQR method to detect outliers.

### Formula

```
IQR = Q3 − Q1

Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

### Learned

- Calculate Q1 and Q3
- Compute IQR
- Detect outliers using Pandas
- Decide whether to keep or remove outliers

### Decision

The detected outliers in the **Fare** column were kept because they represent real passengers who paid expensive ticket prices rather than data entry errors.

---

# Step 4 — Count Plot Analysis

Created count plots for categorical variables.

### Categorical Variables

- Sex
- Embarked

### Learned

- Category frequency
- Class imbalance
- Comparing categorical distributions

---

# Step 5 — Distribution Interpretation

Documented insights for every visualization using Markdown.

Examples include:

- Age distribution
- Fare distribution
- Fare outliers
- Sex distribution
- Embarked distribution

---

# Key Concepts Learned

- Exploratory Data Analysis (EDA)
- Histogram
- Box Plot
- Count Plot
- Outliers
- IQR Method
- Quartiles
- Median
- Distribution
- Numerical Variables
- Categorical Variables
- Class Imbalance
- Statistical Visualization

---

# Skills Practiced

- Loading datasets with Pandas
- Selecting specific columns
- Creating Histograms
- Creating Box Plots
- Creating Count Plots
- Detecting Outliers
- Applying the IQR Method
- Interpreting visualizations
- Writing Markdown documentation

---

# Conclusion

This project demonstrates the importance of performing Exploratory Data Analysis before training Machine Learning models. By understanding data distributions, identifying outliers, and analyzing categorical variables, we can make informed preprocessing decisions and build more reliable models.