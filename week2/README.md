# Week 2 — Data Analysis Foundations & Exploratory Data Analysis (EDA)

## Overview

During Week 2, I learned the fundamental concepts required to understand, analyze, and prepare datasets before building Machine Learning models.

The main goal was to learn how to explore data, understand its behavior, discover relationships between features, and prepare meaningful insights.

Topics covered:

- Descriptive Statistics
- Probability & Distributions
- Linear Algebra for Machine Learning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Outlier Detection
- Correlation Analysis
- Data Storytelling

---

# Learning Objectives

After completing this week, I learned how to:

- Calculate and interpret statistical measurements.
- Understand data distributions.
- Detect and analyze outliers.
- Apply probability rules.
- Understand common probability distributions.
- Represent data using vectors and matrices.
- Perform dot product and matrix multiplication.
- Analyze datasets using EDA techniques.
- Discover relationships between variables.
- Communicate findings using Data Storytelling.

---

# Day 1 — Descriptive Statistics

## Overview

Descriptive Statistics is used to summarize and understand the main characteristics of data before building Machine Learning models.

It helps answer:

- Where is the data centered?
- How spread out is the data?
- Are there unusual values?
- What is the distribution of the data?

---

# Measures of Central Tendency

Central tendency describes the typical value in a dataset.

| Measure | Description |
|---|---|
| Mean | Average value of all observations |
| Median | Middle value after sorting data |
| Mode | Most frequent value |

---

## Mean

The average of all values.

Example:

```python
np.mean(data)
```

Mean is sensitive to outliers because extreme values can change the average.

---

## Median

The middle value after sorting the dataset.

```python
np.median(data)
```

Median is more robust when outliers exist.

---

## Mode

The value that appears most frequently.

Used mainly for categorical data.

---

# Measures of Spread

Spread explains how data values are distributed.

| Measure | Meaning |
|---|---|
| Range | Difference between maximum and minimum values |
| Variance | Average squared distance from the mean |
| Standard Deviation | How much values spread around the mean |
| IQR | Range of the middle 50% of data |

---

# Quartiles

Quartiles divide data into sections:

| Quartile | Meaning |
|---|---|
| Q1 | 25% of values are below it |
| Q2 | Median |
| Q3 | 75% of values are below it |

IQR:

```python
IQR = Q3 - Q1
```

---

# Titanic Dataset Practice

Applied descriptive statistics on the Titanic dataset.

Steps:

1. Load dataset using Pandas.
2. Select Age column.
3. Remove missing values.
4. Calculate:

- Mean
- Median
- Mode
- Standard Deviation
- IQR

---

# Day 2 — Probability & Distributions

## Overview

Probability measures how likely an event is to happen.

Probability range:

```
0 → Impossible

1 → Certain
```

---

# Probability Rules

## Complement Rule

Probability of an event not happening.

Formula:

```
P(Not A) = 1 - P(A)
```

---

## Addition Rule

Used when calculating probability of either event.

Formula:

```
P(A or B)=P(A)+P(B)-P(A and B)
```

---

## Multiplication Rule

Used for events happening together.

For independent events:

```
P(A and B)=P(A) × P(B)
```

---

# Conditional Probability

Calculates the probability of an event given another event.

Formula:

```
P(A|B)=P(A and B)/P(B)
```

Machine Learning uses conditional probability to make predictions based on available features.

---

# Bayes' Theorem

Bayes theorem updates probability when new evidence appears.

Applications:

- Spam detection
- Medical diagnosis
- Naive Bayes classifier
- Text classification

---

# Probability Distributions

## Normal Distribution

A bell-shaped distribution.

Characteristics:

- Most values are close to the mean.
- Extreme values are less common.

Examples:

- Human height
- Measurement errors

---

## Binomial Distribution

Represents the number of successes in repeated yes/no experiments.

Examples:

- Coin flips
- Classification results

---

## Uniform Distribution

All possible outcomes have equal probability.

Example:

Rolling a fair dice.

---

# Practical Probability Work

Implemented:

- Coin flip simulation using NumPy.
- Normal distribution simulation.
- Conditional probability calculations.

Libraries:

- NumPy
- Matplotlib

---

# Day 3 — Linear Algebra for Machine Learning

## Overview

Machine Learning models work with numerical data.

Linear Algebra is the mathematical foundation used to represent and process data.

---

# Vectors

A vector represents one data sample.

Example:

```python
v = np.array([25,50000,3])
```

Represents:

- Age
- Salary
- Experience

---

# Matrices

A matrix represents the complete dataset.

Example:

```python
X = np.array([
[25,50000,3],
[40,80000,10],
[33,62000,5]
])
```

Rows represent samples.

Columns represent features.

---

# Dot Product

The dot product multiplies corresponding elements and sums the results.

Example:

```python
np.dot(features, weights)
```

Used by Machine Learning models to calculate predictions.

---

# Matrix Multiplication

Used to calculate predictions for multiple samples.

Example:

```python
predictions = X @ weights
```

---

# Matrix Shape Rule

Matrix multiplication follows:

```
(m × n) × (n × p) = (m × p)
```

The inner dimensions must match.

Example:

```
(3 × 3) × (3 × 1)

= (3 × 1)
```

---

# Day 4 — EDA Part 1: Distributions & Outliers

## Overview

Exploratory Data Analysis (EDA) is the process of understanding data before building Machine Learning models.

---

# Topics Covered

- Numerical variables
- Categorical variables
- Histograms
- Box plots
- Count plots
- Outlier detection
- IQR method

---

# Histograms

Used to understand numerical distributions.

Analyzed:

- Age
- Fare

Helps understand:

- Frequency
- Data spread
- Distribution shape
- Skewness

---

# Box Plot

Used to analyze:

- Median
- Quartiles
- Data spread
- Outliers

---

# Outlier Detection Using IQR

Formula:

```
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Outliers are values outside these limits.

---

# Count Plot

Used for categorical variables.

Examples:

- Sex
- Embarked

Helps analyze:

- Category frequency
- Class imbalance

---

# Day 5 — EDA Part 2: Correlation & Data Storytelling

## Overview

Focused on finding relationships between variables and explaining insights from data.

---

# Bivariate Analysis

Bivariate analysis studies the relationship between two variables.

Techniques:

- Scatter plots
- Grouped box plots

---

# Scatter Plot

Used to visualize relationships between two numerical variables.

Example:

```python
sns.scatterplot(
data=df,
x="age",
y="income"
)
```

Shows:

- Positive relationships
- Negative relationships
- No relationship

---

# Grouped Box Plot

Compares numerical values between categories.

Example:

```python
sns.boxplot(
data=df,
x="category",
y="income"
)
```

---

# Correlation

Correlation measures how strongly two numerical variables move together.

Range:

```
-1 -------- 0 -------- +1
```

---

# Correlation Matrix

Created using:

```python
corr = df.corr(numeric_only=True)
```

Shows correlation between all numerical features.

---

# Correlation Heatmap

Created using:

```python
sns.heatmap(
corr,
annot=True,
cmap="coolwarm"
)
```

Used to find:

- Strong relationships
- Important features
- Possible predictors

---

# Correlation Does Not Mean Causation

Correlation does not prove that one variable causes another.

Example:

Ice cream sales and swimming accidents increase together.

The actual reason:

```
Hot Weather
```

---

# Pairplot

Creates multiple plots showing relationships between numerical variables.

Example:

```python
sns.pairplot(
df,
hue="target"
)
```

Used for:

- Quickly exploring relationships.
- Finding patterns.

---

# Data Storytelling

Data Storytelling means converting analysis results into understandable conclusions.

A good EDA story explains:

1. What data contains.
2. What patterns were found.
3. What problems exist.
4. How findings affect Machine Learning.

---

# Complete EDA Workflow

```
EDA Notebook

|
├── Data Overview
|
├── Descriptive Statistics
|
├── Univariate Analysis
|
├── Outlier Detection
|
├── Bivariate Analysis
|
├── Correlation Analysis
|
└── Final Insights
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Week 2 Final Summary

During Week 2, I learned:

✅ Descriptive Statistics  
✅ Probability Fundamentals  
✅ Probability Distributions  
✅ Linear Algebra Basics for ML  
✅ Exploratory Data Analysis (EDA)  
✅ Data Visualization  
✅ Outlier Detection  
✅ Correlation Analysis  
✅ Data Storytelling  

These concepts build the foundation needed for preparing datasets and creating effective Machine Learning models.