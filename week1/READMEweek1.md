# Week 1 – Python for Data Analysis & Machine Learning

## 👨‍💻 Student Information

**Name:** Mohammad Abu Hamed

**Training:** Data Analysis & Machine Learning

**Week:** 1

---

# Overview

This repository contains all the practical work, exercises, and mini-projects completed during the **first week** of my **Data Analysis & Machine Learning** training.

Throughout this week, I learned the foundations of Python programming and the essential libraries used in data analysis. The training focused on writing clean Python code, working with numerical data using NumPy, analyzing datasets using Pandas, and creating visualizations with Matplotlib.

The exercises included working with real datasets such as the **Titanic**, **COVID-19**, and **Iris** datasets to practice real-world data analysis techniques.

---

# Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib

---

# Topics Covered During Week 1

## 1. Python Fundamentals

### Lists

Learned how to work with Python lists by:

- Creating lists
- Accessing elements
- Updating values
- Sorting
- Reversing
- Removing elements
- Adding new elements

Methods practiced:

```python
append()
insert()
remove()
pop()
sort()
reverse()
len()
```

Example:

```python
numbers = [1,3,2,5]
numbers.sort()
print(numbers)
```

---

## String Formatting

Learned different ways to format strings.

Methods:

- f-strings
- format()

Example:

```python
name = "Mohammad"
print(f"Welcome {name}")
```

---

## Tuples

Worked with immutable collections.

Topics:

- Creating tuples
- Accessing elements
- Tuple unpacking

Example:

```python
point = (5,10)
```

---

## Dictionaries

Learned how to store data using key-value pairs.

Operations:

- Add values
- Update values
- Delete values
- Access values

Example:

```python
student = {
    "name":"Mohammad",
    "age":21
}
```

---

## Sets

Learned about unique collections.

Operations practiced:

```python
union (|)
intersection (&)
difference (-)
```

---

## Control Flow

Implemented decision-making using:

```python
if
elif
else
```

Example:

```python
if age >= 18:
    print("Adult")
```

---

## Loops

Practiced:

```python
for
while
```

Example:

```python
for i in range(5):
    print(i)
```

---

## List Comprehensions

Created lists using Python's concise syntax.

Example:

```python
even_numbers = [x for x in range(20) if x % 2 == 0]
```

---

## Object-Oriented Programming (OOP)

Created simple classes.

Topics:

- class
- constructor (__init__)
- objects
- attributes

Example:

```python
class DataPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
```

---

# 2. NumPy

NumPy was introduced as the foundation of numerical computing in Python.

---

## Array Creation

Learned to create arrays using:

```python
np.array()
np.zeros()
np.ones()
np.arange()
np.linspace()
np.random
```

---

## Array Attributes

Worked with:

```python
shape
dtype
size
ndim
```

---

## Indexing & Slicing

Practiced:

- One-dimensional arrays
- Two-dimensional arrays
- Row slicing
- Column slicing

---

## Vectorized Operations

Performed arithmetic without loops.

Examples:

```python
+
-
*
/
**
```

---

## Boolean Masking

Filtered arrays using logical conditions.

Example:

```python
arr[arr > 5]
```

---

## Matrix Multiplication

Learned the difference between:

```python
*
np.dot()
@
np.matmul()
```

---

## Broadcasting

Performed operations between arrays with compatible shapes.

---

## Loading External Data

Used:

```python
np.genfromtxt()
```

to load datasets.

---

## Reshaping Arrays

Practiced:

```python
reshape()
concatenate()
```

---

## NumPy Skills Learned

✔ Numerical Computing

✔ Matrix Operations

✔ Broadcasting

✔ Dot Product

✔ Vectorization

✔ Array Manipulation

---

# 3. Pandas

Learned how to analyze and manipulate tabular datasets.

---

## Reading Data

Loaded CSV files using:

```python
pd.read_csv()
```

---

## Dataset Inspection

Practiced:

```python
head()
tail()
info()
describe()
shape
columns
dtypes
sample()
```

---

## Missing Values

Handled missing data using:

```python
isnull()
fillna()
drop()
```

Techniques:

- Mean
- Median
- Column removal

---

## Filtering Data

Filtered rows using conditions.

Examples:

```python
Age < 18
new_cases > 2000
```

---

## Selecting Data

Used:

```python
loc[]
```

to select rows and columns.

---

## Sorting

Sorted datasets using:

```python
sort_values()
```

---

## Date Handling

Converted strings into datetime objects.

Functions:

```python
pd.to_datetime()
DatetimeIndex()
```

Extracted:

- Year
- Month
- Day

---

## Grouping

Used:

```python
groupby()
sum()
mean()
```

for aggregation.

---

## Datasets Used

### Titanic Dataset

Tasks completed:

- Load dataset
- Explore data
- Handle missing values
- Filter passengers
- Group survivors by class

---

### COVID-19 Dataset

Tasks completed:

- Explore data
- Sort values
- Filter records
- Convert dates
- Group monthly statistics
- Handle duplicates
- Check missing values

---

# 4. Data Visualization with Matplotlib

Learned how to visualize data effectively.

---

## Charts Created

### Histogram

Used for:

- Age distribution
- Iris sepal width comparison

Function:

```python
plt.hist()
```

---

### Scatter Plot

Visualized the relationship between:

- Age
- Fare

Function:

```python
plt.scatter()
```

---

### Bar Chart

Displayed:

- Number of survivors
- Number of non-survivors

Function:

```python
plt.bar()
```

---

### Line Plot

Compared fruit growth over several years.

Functions:

```python
plt.plot()
plt.legend()
```

---

## Figure Customization

Learned how to use:

```python
plt.subplots()
plt.title()
plt.xlabel()
plt.ylabel()
plt.legend()
plt.tight_layout()
plt.show()
```

---

# 5. Feature Engineering

Created new features from existing data.

Example:

```python
have_family = (SibSp > 0) | (Parch > 0)
```

This identifies whether a passenger traveled with family members.

---

# 6. Real Datasets Used

Throughout this week, I worked with several real datasets:

- Titanic Dataset
- COVID-19 Dataset
- Iris Dataset
- Climate Dataset

These datasets helped me apply data cleaning, exploration, analysis, and visualization techniques in realistic scenarios.

---

# Libraries Learned

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

# Functions & Methods Practiced

## Python

- append()
- insert()
- remove()
- pop()
- sort()
- reverse()
- len()
- range()
- print()

---

## NumPy

- array()
- zeros()
- ones()
- arange()
- linspace()
- dot()
- reshape()
- concatenate()
- genfromtxt()

---

## Pandas

- read_csv()
- head()
- tail()
- info()
- describe()
- shape
- columns
- dtypes
- fillna()
- drop()
- groupby()
- mean()
- median()
- sum()
- sample()
- sort_values()
- loc[]
- to_datetime()

---

## Matplotlib

- plot()
- scatter()
- hist()
- bar()
- subplots()
- xlabel()
- ylabel()
- title()
- legend()
- tight_layout()
- show()

---

# Skills Gained

During the first week of training, I learned how to:

- Write clean Python code.
- Work with Python data structures.
- Apply Object-Oriented Programming concepts.
- Perform numerical computing using NumPy.
- Manipulate arrays efficiently.
- Analyze real datasets using Pandas.
- Handle missing values.
- Filter and group data.
- Work with date and time data.
- Perform basic feature engineering.
- Create informative visualizations.
- Explore datasets before machine learning.
- Understand the complete basic workflow of data analysis.

---

# Conclusion

The first week provided a strong foundation in Python programming and data analysis. By combining Python fundamentals with NumPy, Pandas, and Matplotlib, I gained practical experience in loading, cleaning, transforming, analyzing, and visualizing real-world datasets.

This repository represents my progress during **Week 1** and serves as the starting point of my journey toward becoming a **Data Analyst and Machine Learning Engineer**. More advanced topics and projects will be added in the upcoming weeks as I continue my training.