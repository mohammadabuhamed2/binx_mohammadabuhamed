### Day 3 — NumPy: Numerical Computing 🚀

This repository contains practical code examples and exercises completed during Day 3 of the Data Analysis & Machine Learning training, focusing on **NumPy (Numerical Python)**, which serves as the fundamental numerical backbone for the entire AI/ML stack.

---

### 📚 Learning Objectives & Key Topics Covered

* **Why NumPy:** Understanding `ndarray`, memory efficiency, homogeneity, and optimized C-code performance over native Python lists.
* **Array Creation:** Utilizing `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, and `np.random`.
* **Attributes & Indexing/Slicing:** Inspecting `shape`, `dtype`, and performing multi-dimensional slicing (including 2D arrays).
* **Vectorized Operations & Boolean Masking:** Executing element-wise math without explicit loops and filtering data conditionally.
* **Broadcasting & Matrix Multiplication:** Combining shapes compatibly and performing dot products (`np.dot` vs `*` and `np.matmul`).
* **Data Integration:** Loading external datasets (`np.genfromtxt`), computing weighted sums, and restructuring arrays (`np.concatenate`, `reshape`).

---

### 💻 Code Implementation

Here is the complete Python script combining all the concepts and experiments implemented during the session:

```python
import numpy as np

# 1. Basic Vector Operations and Dot Products
l1 = np.array([10, 2, 15])
l2 = np.array([15, 2, 4])

print(np.dot(l1, l2))       # Dot product calculation
print((l1 * l2).sum())    # Element-wise multiplication followed by sum (equivalent to dot)
print(l1.shape)           # Inspecting dimensions

# 2. Array Creation Functions
z = np.zeros((2, 3))      # 2x3 matrix of zeros
o = np.ones((4, 4))       # 4x4 matrix of ones
r = np.arange(0, 15, 5)   # Range sequence with a step of 5 [0, 5, 10]
li = np.linspace(0, 1, 3) # 3 evenly spaced values between 0 and 1

print(f"The zeros matrix is:\n {z}")
print(f"The ones matrix is:\n {o}")
print(f"The range matrix is:\n {r}")
print(f"The linspace matrix is:\n {li}")

# 3. Matrix Multiplication (Matmul)
l3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0, 1, 2]
])
print("Matrix l3:\n", l3)
print("Vector l1:", l1)

# Multiplying (4,3) matrix by (3,) vector using the @ operator
l4 = l3 @ l1
print("*************")
print("Matrix Multiplication result (l3 @ l1):\n", l4)

# 4. Working with External Climate Data & Vectorized Weights
# Loading CSV dataset
climate_data = np.genfromtxt("week1/day3/climate.csv", delimiter=",", skip_header=1)
print("Climate Data Shape:", climate_data.shape)

# Applying weights using dot product (Weighted Sum / Linear Model representation)
wh = np.array([0.3, 0.7, 0.5])
res = np.dot(climate_data, wh)
print("Resulting Predictions Shape:", res.shape)

# 5. Concatenation and Reshaping
# Appending the result column back to the original dataset
fres = np.concatenate((climate_data, res.reshape(10000, 1)), axis=1)
print("*********************************")
print("Final Concatenated Array Preview:\n", fres)