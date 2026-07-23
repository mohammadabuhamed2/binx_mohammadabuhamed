# Titanic & Iris Data Analysis using Python

## Project Overview

This project demonstrates basic data analysis and visualization using Python.

The project uses:
- NumPy
- Pandas
- Matplotlib

The analysis includes:
- Cleaning missing data.
- Creating a new feature.
- Performing simple statistics.
- Visualizing data using different chart types.

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

# Part 1 - Titanic Dataset

## Load Dataset

```python
pd.read_csv()
```

The Titanic dataset is loaded from a CSV file.

---

## Data Cleaning

The following preprocessing steps were applied:

- Removed the **Cabin** column because most of its values were missing.
- Filled missing values in **Age** using the column mean.
- Filled missing values in **Fare** using the column median.

Functions used:

- `drop()`
- `fillna()`
- `mean()`
- `median()`
- `isnull().sum()`

---

## Feature Engineering

A new column called **have_family** was created.

A passenger is considered to have family if:

- SibSp > 0
- OR
- Parch > 0

Example:

```python
have_family = (titanic.SibSp > 0) | (titanic.Parch > 0)
```

The total number of passengers with family members was calculated using:

```python
np.array()
sum()
```

---

# Data Visualization

## Histogram

Shows the distribution of passenger ages.

```python
plt.hist()
```

---

## Scatter Plot

Displays the relationship between:

- Age
- Fare

```python
plt.scatter()
```

---

## Bar Chart

Shows the number of:

- Survived passengers
- Non-survived passengers

Functions used:

```python
value_counts()
plt.bar()
```

---

## Figure Layout

Multiple charts were displayed in one figure using:

```python
plt.subplots()
plt.tight_layout()
```

---

# Part 2 - Fruit Growth Example

A simple line chart example comparing:

- Apples
- Oranges

Functions used:

```python
plt.plot()
plt.xlabel()
plt.ylabel()
plt.title()
plt.legend()
```

---

# Part 3 - Iris Dataset

The Iris dataset was loaded using Pandas.

Different flower species were separated into individual DataFrames.

Species:

- Setosa
- Versicolor
- Virginica

Example:

```python
flow[flow.species == 'setosa']
```

---

## Histogram Comparison

A histogram was drawn for the sepal width of each species.

Functions used:

```python
plt.hist()
plt.legend()
plt.show()
```

---

# Python Concepts Practiced

- Reading CSV files
- Data cleaning
- Handling missing values
- Creating new columns
- Boolean indexing
- Filtering data
- Counting values
- NumPy arrays
- Histograms
- Scatter plots
- Bar charts
- Figure customization
- Labels and titles
- Legends

---

# Charts Used

- Histogram
- Scatter Plot
- Bar Chart
- Line Plot

---

# Skills Learned

✔ Data Cleaning

✔ Missing Value Handling

✔ Feature Engineering

✔ Data Filtering

✔ Data Visualization

✔ NumPy Basics

✔ Pandas Basics

✔ Matplotlib Basics

---

# Files Used

- tested.csv
- iris_148.csv

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib