# Pandas Data Analysis Lab

## Overview
This lab focuses on learning the basics of **Pandas** for data analysis using two real datasets:

- Titanic Dataset
- COVID-19 Dataset

The goal was to practice loading datasets, exploring their structure, cleaning data, filtering records, and performing basic aggregations.

---

# Learning Objectives

- Load CSV files into Pandas DataFrames.
- Inspect dataset structure.
- Select columns and filter rows.
- Handle missing values.
- Group and aggregate data.
- Work with date columns.
- Perform basic data exploration.

---

# Technologies Used

- Python 3
- Pandas

---

# Dataset 1: Titanic

## Tasks Completed

### 1. Load the dataset

Loaded the Titanic dataset using:

```python
pd.read_csv()
```

Displayed:

- Dataset
- Shape
- Columns
- Data types

---

### 2. Handle Missing Values

Checked missing values using:

```python
isnull().sum()
```

Cleaning steps:

- Removed the **Cabin** column because most values were missing and it was not important for the analysis.
- Filled missing **Age** values using the mean.
- Filled missing **Fare** values using the median.

---

### 3. Filtering

Filtered passengers younger than 18 years old.

Example:

```python
Age < 18
```

---

### 4. Grouping

Grouped passengers by passenger class (**Pclass**) and calculated the total number of survivors.

Used:

```python
groupby()
sum()
```

---

# Dataset 2: COVID-19

## Tasks Completed

### 1. Load Dataset

Loaded the CSV file and explored:

- describe()
- info()
- columns
- shape
- dtypes

---

### 2. Data Inspection

Displayed:

- One column as a Series
- Specific row using loc
- First rows
- Last rows
- Random sample

---

### 3. Filtering

Selected rows where:

```python
new_cases > 2000
```

---

### 4. Sorting

Sorted the dataset by:

```python
new_cases
```

---

### 5. Row Selection

Selected rows between indices:

```python
180 : 188
```

Calculated the average number of new cases for a selected range.

---

### 6. Date Conversion

Converted the **date** column from string to datetime.

```python
pd.to_datetime()
```

Extracted:

- Year
- Month
- Day

---

### 7. Grouping

Grouped the data by month and calculated the total number of new cases.

Used:

```python
groupby()
sum()
```

---

### 8. Missing Values

Checked missing values in every column using:

```python
isnull().sum()
```

---

### 9. Duplicate Handling

Attempted to remove duplicate values from the **new_tests** column.

---

# Pandas Functions Used
```python
- read_csv()
- head()
- tail()
- info()
- describe()
- shape
- columns
- dtypes
- isnull()
- fillna()
- drop()
- groupby()
- sum()
- mean()
- median()
- sample()
- sort_values()
- loc[]
- to_datetime()
- DatetimeIndex()
```
---

# What I Learned

During this lab I learned how to:

- Load and inspect real-world datasets.
- Understand the difference between a **Series** and a **DataFrame**.
- Clean missing data using different techniques.
- Filter rows based on conditions.
- Group data and calculate summary statistics.
- Convert text dates into datetime objects.
- Extract useful information such as year, month, and day.
- Explore datasets before starting any machine learning or data analysis task.

---

# Conclusion

This lab introduced the essential features of **Pandas** for data analysis. Working with the Titanic and COVID-19 datasets provided practical experience in loading, cleaning, filtering, grouping, and exploring real datasets, which are fundamental skills in data science and machine learning.