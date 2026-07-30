# Day 5 — EDA Part 2: Correlation & Data Storytelling

## Overview

In this day, we continued the **Exploratory Data Analysis (EDA)** process and focused on discovering relationships between variables and communicating insights from the data.

We learned how to:

- Analyze relationships between two variables using **Bivariate Analysis**.
- Use **Scatter Plots** and **Grouped Box Plots** to discover patterns.
- Calculate and visualize **Correlation Matrix** using **Heatmaps**.
- Understand why **Correlation does not mean Causation**.
- Use **Pairplots** to quickly scan relationships between variables.
- Transform analysis results into a clear **Data Storytelling narrative**.

---

# Learning Objectives

After completing this lesson, I can:

- Perform **Bivariate Analysis** using:
  - Scatter plots
  - Grouped box plots

- Compute and interpret a **Correlation Matrix**.

- Create and understand a **Correlation Heatmap**.

- Explain the difference between:
  - Correlation
  - Causation

- Use `pairplot()` to explore relationships between numerical variables.

- Build a complete EDA notebook that includes:

  - Data overview
  - Descriptive statistics
  - Univariate analysis
  - Outlier detection
  - Bivariate analysis
  - Correlation analysis

- Create a data story that explains the important findings from the dataset.

---

# 5.1 Bivariate Analysis

## What is Bivariate Analysis?

**Bivariate Analysis** is the process of analyzing the relationship between two variables.

Unlike univariate analysis, which studies one variable, bivariate analysis focuses on understanding how two variables interact.

It helps answer questions such as:

- Do two variables have a relationship?
- Do they increase or decrease together?
- Is there a pattern that can help a Machine Learning model?

Machine Learning is based on finding relationships between:

- **Features** → Input variables
- **Target** → Output variable

---

# Scatter Plot

## What is a Scatter Plot?

A **Scatter Plot** is a visualization used to show the relationship between two numerical variables.

Example:

We want to study the relationship between:

- Age
- Income

Code:

```python
sns.scatterplot(data=df, x="age", y="income")
```

---

## Code Explanation

### `data=df`

Specifies the dataset we want to use.

### `x="age"`

The numerical variable displayed on the x-axis.

### `y="income"`

The numerical variable displayed on the y-axis.

---

## Scatter Plot Interpretation

A scatter plot can show different types of relationships:

---

## Positive Relationship

When one variable increases, the other variable also increases.

Example:

```
Age ↑  →  Income ↑
```

---

## Negative Relationship

When one variable increases, the other variable decreases.

Example:

```
Price ↑  →  Demand ↓
```

---

## No Relationship

When there is no clear pattern between the variables.

---

# Grouped Box Plot

## What is a Grouped Box Plot?

A **Grouped Box Plot** is used to compare a numerical variable across different categories.

Example:

Comparing income between different groups.

Code:

```python
sns.boxplot(data=df, x="category", y="income")
```

---

## Why Use Box Plots?

Box plots help us understand:

- Median
- Data spread
- Distribution
- Outliers
- Differences between groups

Example:

```
Category A → Higher Income
Category B → Lower Income
```

This information may help identify important features for Machine Learning models.

---

# 5.2 Correlation

## What is Correlation?

**Correlation** measures how strongly two numerical variables move together.

The correlation value ranges between:

```
-1 ---------------- 0 ---------------- +1
```

---

# Correlation Values Explanation

## Positive Correlation (+1)

Both variables move in the same direction.

Example:

```
Study Hours ↑
Grades ↑
```

Increasing study hours is associated with higher grades.

---

## Negative Correlation (-1)

Variables move in opposite directions.

Example:

```
Price ↑
Demand ↓
```

---

## Zero Correlation (0)

There is no clear linear relationship between the variables.

---

# Correlation Matrix

## Creating Correlation Matrix

Code:

```python
corr = df.corr(numeric_only=True)
```

---

## What Does This Code Do?

It calculates the correlation between all numerical columns in the dataset.

Example:

| Feature | Age | Salary | Score |
|---|---|---|---|
| Age | 1.0 | 0.5 | 0.2 |
| Salary | 0.5 | 1.0 | 0.7 |
| Score | 0.2 | 0.7 | 1.0 |

This table is called a:

**Correlation Matrix**

---

# Correlation Heatmap

Code:

```python
sns.heatmap(corr, annot=True, cmap="coolwarm")
```

---

## Code Explanation

### `sns.heatmap()`

Converts the correlation matrix into a visual chart.

### `annot=True`

Displays correlation values inside each cell.

### `cmap="coolwarm"`

Defines the color scheme of the heatmap.

---

## Why Use a Correlation Heatmap?

A heatmap helps us quickly find:

- Strong positive relationships.
- Strong negative relationships.
- Features that may be useful for Machine Learning models.

Example:

```
Age and Income = 0.85
```

This means they have a strong positive relationship.

---

# Correlation Does Not Mean Causation

One of the most important concepts in Data Analysis:

> Correlation does not mean causation.

Two variables moving together does not mean one variable causes the other.

Example:

Ice cream sales and drowning incidents may increase together.

Does ice cream cause drowning?

No.

The real reason is:

```
Hot Weather
```

Hot weather increases:

- Ice cream consumption
- Swimming activities

EDA can discover relationships, but it cannot prove cause and effect.

---

# 5.3 Pairplot

## What is Pairplot?

`pairplot()` creates multiple plots showing relationships between every numerical variable in a dataset.

It contains:

- Scatter plots between variables.
- Distribution plots on the diagonal.

Code:

```python
sns.pairplot(df, hue="target")
```

---

# Why Use Pairplot?

Pairplot is useful during the early stages of EDA because it helps us:

- Quickly inspect all relationships.
- Find possible patterns.
- Decide which relationships need deeper analysis.

---

# Understanding `hue`

Example:

```python
sns.pairplot(df, hue="target")
```

The `hue` parameter separates data points based on a categorical column.

Example:

```
Target:

0 → Class 0
1 → Class 1
```

The plot will display each class with a different color.

---

# 5.4 Data Storytelling

## What is Data Storytelling?

**Data Storytelling** means transforming analysis results into a clear and understandable explanation.

EDA is not only about creating charts.

The final goal is:

- Understanding the dataset.
- Finding important patterns.
- Explaining what those patterns mean.

---

# Data Storytelling Structure

A good EDA story should answer:

## 1. What Does the Data Contain?

Example:

"The dataset contains information about passengers including age, gender, and ticket class."

---

## 2. What Patterns Were Found?

Example:

"Passengers in higher classes had a higher survival rate."

---

## 3. What Problems Were Found?

Example:

"The Age column contains missing values that need preprocessing."

---

## 4. How Can Findings Help Modeling?

Example:

"Passenger class may be an important feature for predicting survival."

---

# 5.5 Full EDA Notebook

A complete EDA notebook combines all previous steps.

Structure:

```
EDA Notebook

│
├── Data Overview
│
├── Descriptive Statistics
│
├── Univariate Analysis
│
├── Outlier Detection
│
├── Bivariate Analysis
│
├── Correlation Analysis
│
└── Final Insights
```

---

# Complete EDA Workflow

## 1. Descriptive Statistics

Understanding:

- Mean
- Median
- Standard deviation
- Minimum values
- Maximum values

---

## 2. Univariate Analysis

Analyzing one variable using:

- Histograms
- Count plots
- Box plots

---

## 3. Outlier Detection

Finding unusual values using:

- Box plots
- IQR method

---

## 4. Bivariate Analysis

Analyzing relationships using:

- Scatter plots
- Grouped box plots

---

## 5. Correlation Analysis

Finding relationships between numerical variables using:

- Correlation Matrix
- Heatmap

---

# Hands-On Lab: Complete EDA Notebook

## Step 1: Create Scatter Plots and Grouped Box Plots

Analyze the most important variable relationships.

Example:

```python
sns.scatterplot(data=df, x="feature1", y="feature2")
```

```python
sns.boxplot(data=df, x="category", y="value")
```

---

## Step 2: Create Correlation Matrix and Heatmap

Code:

```python
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)
```

---

## Step 3: Identify Strong Relationships

Find:

- Strong positive correlations.
- Strong negative correlations.

Then explain what these relationships might mean for future Machine Learning models.

Example:

"Feature A has a strong relationship with the target, so it may be an important predictor."

---

## Step 4: Assemble Full EDA Notebook

The notebook should contain:

```
1. Dataset Overview

2. Descriptive Statistics

3. Missing Values Analysis

4. Univariate Analysis

5. Outlier Detection

6. Bivariate Analysis

7. Correlation Analysis

8. Final Conclusions
```

---

## Step 5: Commit EDA Notebook to GitHub

Example:

```bash
git add .

git commit -m "Complete EDA analysis with correlation and storytelling"

git push
```

---

# Key Takeaways

- Bivariate Analysis helps discover relationships between variables.
- Scatter plots visualize relationships between numerical variables.
- Box plots compare numerical values across categories.
- Correlation measures the strength of linear relationships.
- Correlation does not prove causation.
- Pairplots provide a quick overview of relationships in the dataset.
- Data Storytelling converts analysis results into meaningful conclusions.
- A complete EDA notebook is the foundation for future Machine Learning projects.