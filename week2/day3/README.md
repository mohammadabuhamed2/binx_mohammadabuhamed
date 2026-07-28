# Linear Algebra for Machine Learning

This repository contains my practical work and notes on the fundamental Linear Algebra concepts used in Machine Learning.

---

# Learning Objectives

After completing this lesson, I learned how to:

- Represent a single data sample as a vector.
- Represent an entire dataset as a matrix.
- Compute the dot product manually and using NumPy.
- Understand why the dot product is the foundation of model prediction.
- Perform matrix multiplication using the `@` operator.
- Understand matrix shapes and how to resolve shape mismatch errors.

---

# Topics Covered

## 1. Why Linear Algebra is the Language of Machine Learning

Machine Learning models work entirely with numbers.

- A dataset is represented as a matrix.
- Each row represents one sample.
- Each column represents one feature.
- Model parameters (weights) are stored as vectors or matrices.
- During training and prediction, models perform matrix operations continuously.

---

## 2. Vectors

A vector is an ordered list of numbers that usually represents a single data sample.

Example:

```python
import numpy as np

v = np.array([25, 50000, 3])
```

Where:

- 25 → Age
- 50000 → Salary
- 3 → Years of Experience

Shape:

```python
v.shape
```

Output:

```text
(3,)
```

Meaning:

- One vector
- Three features

---

## 3. Matrices

A matrix represents an entire dataset.

Example:

```python
X = np.array([
    [25, 50000, 3],
    [40, 80000, 10],
    [33, 62000, 5]
])
```

Shape:

```python
X.shape
```

Output:

```text
(3,3)
```

Meaning:

- 3 samples (rows)
- 3 features (columns)

Each row represents one data sample.

Each column represents one feature.

---

## 4. Dot Product

The dot product multiplies corresponding elements of two vectors and sums the results to produce a single value.

Example:

```python
features = np.array([25,50000,3])

weights = np.array([0.1,0.0002,1.5])

prediction = np.dot(features, weights)
```

Manual calculation:

```
25 × 0.1 = 2.5

50000 × 0.0002 = 10

3 × 1.5 = 4.5

Prediction = 2.5 + 10 + 4.5 = 17
```

The dot product is the core operation used by linear models to generate predictions.

---

## 5. Matrix Multiplication

Matrix multiplication allows a model to generate predictions for multiple samples at once.

Example:

```python
X = np.array([
    [25,50000,3],
    [40,80000,10],
    [33,62000,5]
])

w = np.array([0.1,0.0002,1.5])

predictions = X @ w
```

Instead of calculating the dot product for each sample separately, matrix multiplication computes all predictions in one operation.

---

# Shape Rule

For matrix multiplication:

```
(m × n) × (n × p) → (m × p)
```

The inner dimensions must be equal.

Example:

```
(3 × 3)

×

(3 × 1)

↓

(3 × 1)
```

If the inner dimensions do not match, NumPy raises a Shape Mismatch Error.

---

# Common Shape Mistake

Incorrect:

```python
X.shape
```

```
(3,3)
```

```python
weights.shape
```

```
(2,)
```

Trying:

```python
X @ weights
```

Produces:

```
ValueError: matmul: Input operand has a mismatch in its core dimension
```

Reason:

```
(3×3)

×

(2,)

3 ≠ 2
```

Solution:

Provide one weight for every feature.

Correct:

```python
weights = np.array([1,2,3])
```

Shape:

```
(3,)
```

Now the multiplication succeeds.

---

# Hands-On Lab

During this lab, I completed the following tasks:

- Represented three data samples as a NumPy matrix.
- Computed the dot product manually.
- Verified the result using `np.dot()`.
- Used matrix multiplication (`@`) to generate predictions for all samples.
- Created a shape mismatch error intentionally.
- Identified the cause of the error.
- Fixed the error by matching the matrix dimensions.

---

# Key Takeaways

- A vector represents one data sample.
- A matrix represents an entire dataset.
- Every feature requires one corresponding weight.
- The dot product computes a prediction for one sample.
- Matrix multiplication computes predictions for multiple samples simultaneously.
- Understanding matrix shapes is essential to avoid shape mismatch errors.

---

# Technologies Used

- Python
- NumPy
- Jupyter Notebook