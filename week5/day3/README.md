# Day 3 — Dimensionality Reduction with PCA

## 📌 Overview

In Day 3, I learned how to reduce the dimensionality of a dataset using **Principal Component Analysis (PCA)**.

The main goal of PCA is to reduce the number of dimensions while preserving as much of the original data variance as possible.

PCA was applied to the Cardiac Patients dataset to:

- Understand the Curse of Dimensionality.
- Reduce the number of dimensions.
- Analyze the Explained Variance.
- Determine how many Principal Components are needed to preserve approximately 95% of the variance.
- Reduce the dataset to two dimensions for visualization.
- Understand the trade-off between dimensionality reduction and information loss.

---

# 🎯 Learning Objectives

By the end of this lesson, I was able to:

1. Explain the Curse of Dimensionality.
2. Understand why dimensionality reduction can be useful.
3. Understand how PCA creates Principal Components.
4. Apply PCA to scaled numerical data.
5. Interpret the Explained Variance Ratio.
6. Calculate the Cumulative Explained Variance.
7. Choose the number of components needed to preserve approximately 95% of the variance.
8. Reduce high-dimensional data to two dimensions for visualization.
9. Understand the advantages and limitations of PCA.

---

# 📐 The Curse of Dimensionality

Real-world datasets can contain many features.

Each feature can be considered a dimension of the dataset.

For example:

```text
2 Features  → 2 Dimensions
10 Features → 10 Dimensions
22 Features → 22 Dimensions
```

As the number of dimensions increases, several problems can appear:

- Data becomes more sparse.
- Distance between observations becomes less informative.
- Some models may become more prone to overfitting.
- Visualization becomes difficult beyond two or three dimensions.

Dimensionality Reduction helps reduce these problems by representing the dataset using fewer dimensions while trying to preserve the important information.

---

# 🧠 Principal Component Analysis

**Principal Component Analysis (PCA)** is a dimensionality reduction technique.

Instead of simply selecting some of the original features, PCA creates new features called:

**Principal Components**

For example:

```text
Original Features

age
bmi
daily_steps
cholesterol
heart_rate
...

        ↓
       PCA
        ↓

PC1
PC2
PC3
...
```

Each Principal Component is a combination of the original features.

The components are ordered according to how much variance they capture.

```text
PC1 → captures the most variance
PC2 → captures the second most variance
PC3 → captures the next most variance
...
```

Therefore, the first few Principal Components usually contain more information than the later components.

---

# ⚖️ Scaling Before PCA

PCA is based on variance.

Features with larger numerical ranges could dominate the PCA calculation if the data is not scaled.

Therefore, the numerical features were standardized before applying PCA.

```python
from sklearn.preprocessing import StandardScaler

X = df.drop("has_heart_disease", axis=1)

X_numeric = X.select_dtypes(include="number")

X_scaled = StandardScaler().fit_transform(X_numeric)
```

After preprocessing, the dataset contained:

```text
22 dimensions
```

These scaled features were used as the input for PCA.

---

# Step 1 — Scale the Dataset

The numerical features were standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_numeric)
```

Scaling ensures that features with larger numerical ranges do not receive artificial importance during PCA.

---

# Step 2 — PCA and Cumulative Explained Variance

## Fit PCA to the Scaled Data

PCA was first fitted without specifying the number of components.

```python
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X_scaled)
```

No `n_components` value was specified because all available Principal Components were needed to analyze the amount of variance preserved by each component.

---

## Explained Variance Ratio for Each Principal Component

The Explained Variance Ratio was obtained using:

```python
print(pca.explained_variance_ratio_)
```

The result was:

```text
[0.16165568 0.09072450 0.07925986 0.07393808 0.06255891
 0.05516533 0.05047894 0.04779952 0.04554910 0.04486229
 0.04124949 0.03955390 0.03763937 0.03554665 0.03520220
 0.02458641 0.02402902 0.02235896 0.01113668 0.00976609
 0.00553991 0.00139910]
```

The first value represents the variance captured by PC1, the second represents PC2, and so on.

For example:

```text
PC1 → 16.17%
PC2 → 9.07%
PC3 → 7.93%
PC4 → 7.39%
```

PC1 captures the largest amount of variance.

---

# 📊 Cumulative Explained Variance

The individual Explained Variance Ratios were converted into a cumulative sum.

```python
import numpy as np

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)
```

The result was:

```text
[0.16165568 0.25238018 0.33164004 0.40557813 0.46813704
 0.52330237 0.57378131 0.62158083 0.66712993 0.71199222
 0.75324171 0.79279561 0.83043498 0.86598163 0.90118383
 0.92577024 0.94979926 0.97215822 0.98329490 0.99306099
 0.99860090 1.00000000]
```

The cumulative values show how much total variance is preserved when multiple components are used together.

For example:

```text
PC1                     → 16.17%
PC1 + PC2               → 25.24%
PC1 + PC2 + PC3         → 33.16%
...
First 17 Components     → 94.98%
```

---

# 📈 Plot the Cumulative Explained Variance

The Cumulative Explained Variance was plotted against the number of Principal Components.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")

plt.show()
```

The X-axis represents the number of Principal Components.

The Y-axis represents the cumulative amount of variance preserved.

The graph shows that increasing the number of components gradually increases the amount of preserved variance.

---

# Step 3 — Choosing the Number of Principal Components

The goal was to preserve approximately **95% of the total variance**.

The results showed:

```text
15 Components → 90.12%
16 Components → 92.58%
17 Components → 94.98%
18 Components → 97.22%
```

Since the objective was to preserve approximately 95% of the variance, **17 Principal Components** were selected.

---

## Reduce the Data to 17 Principal Components

```python
pca_17 = PCA(n_components=17)

X_pca_17 = pca_17.fit_transform(X_scaled)

print(pca_17.explained_variance_ratio_.sum())
```

The 17 Principal Components preserve approximately:

```text
94.98% ≈ 95%
```

of the total variance.

---

## Interpretation

The original dataset contained:

```text
22 Dimensions
```

After PCA:

```text
22 Dimensions
      ↓
     PCA
      ↓
17 Principal Components
      ↓
≈ 95% of the variance preserved
```

This reduces the dimensionality of the dataset while retaining most of its original variation.

---

# Step 4 — PCA 2D Visualization

Although 17 components were selected to preserve approximately 95% of the variance, 17 dimensions cannot be directly visualized in a normal 2D scatter plot.

Therefore, PCA was applied again using only two components for visualization.

---

## Reduce the Data to Two Principal Components

```python
pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(X_scaled)
```

The original 22-dimensional data was transformed into:

```text
PC1
PC2
```

Each patient can now be represented as one point in a two-dimensional space.

---

## Explained Variance of the First Two Components

The Explained Variance Ratio was calculated:

```python
print(pca_2.explained_variance_ratio_)
print(pca_2.explained_variance_ratio_.sum())
```

The result was:

```text
[0.16165568 0.0907245]

Total = 0.25238018094986897
```

Therefore:

```text
PC1 → 16.17%
PC2 → 9.07%

PC1 + PC2 → 25.24%
```

The first two Principal Components preserve approximately **25.24% of the total variance**.

---

# 📊 Visualize the Data Using the First Two Principal Components

The two Principal Components were visualized using a scatter plot.

```python
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca_2[:, 0],
    X_pca_2[:, 1],
    c=df["has_heart_disease"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - 2D Visualization")

plt.colorbar(
    scatter,
    label="Heart Disease"
)

plt.show()
```

In this visualization:

```text
X_pca_2[:, 0] → PC1
X_pca_2[:, 1] → PC2
```

The points were colored using the known `has_heart_disease` group.

```text
0 → No Heart Disease
1 → Heart Disease
```

The target was used only to color and interpret the visualization. It was not included as an input feature when fitting PCA.

---

# 🔍 PCA 2D Visualization Interpretation

The first two Principal Components preserve approximately **25.24% of the total variance**.

The visualization shows some separation between patients with and without heart disease, mainly along PC1.

However, there is still considerable overlap between the two groups.

This indicates that PC1 and PC2 capture some useful structure in the dataset, but two components alone do not preserve enough variance to fully represent the original 22-dimensional data.

---

# Step 5 — PCA Reduction Analysis

PCA reduced the original dataset from **22 features to 17 Principal Components** while preserving approximately **95% of the total variance**.

This means that most of the variation in the original data was retained using fewer dimensions.

For visualization, the dataset was further reduced to **2 Principal Components**.

These two components preserved approximately **25.24% of the total variance**, allowing the high-dimensional dataset to be visualized in a 2D scatter plot.

---

# ✅ What PCA Preserved

Using 17 Principal Components:

- Approximately **95% of the total variance** was preserved.
- Most of the variation in the original dataset was retained.
- The number of dimensions was reduced from 22 to 17.

Using two Principal Components:

- Some of the major structure in the dataset remained visible.
- The dataset became possible to visualize using a normal 2D scatter plot.

---

# ⚠️ What PCA Cost

PCA also introduces some trade-offs.

### Information Loss

Reducing:

```text
22 Dimensions
      ↓
17 Components
```

means that approximately 5% of the variance is not preserved.

Reducing further to:

```text
2 Components
```

preserves only approximately:

```text
25.24%
```

of the total variance.

Therefore, the 2D visualization does not contain all the information from the original dataset.

### Interpretability

The original features have clear meanings:

```text
Age
BMI
Daily Steps
Cholesterol
...
```

Principal Components do not have the same direct interpretation.

For example:

```text
PC1
PC2
```

are combinations of multiple original features.

Therefore, PCA improves dimensionality reduction and visualization at the cost of making the transformed features harder to interpret directly.

---

# 🧠 When to Use PCA

PCA can be useful when:

- A dataset contains many features.
- Some features contain redundant or correlated information.
- Dimensionality needs to be reduced.
- High-dimensional data needs to be visualized in 2D or 3D.
- A lower-dimensional representation is useful for downstream modeling.

---

# ⚠️ When PCA May Not Be Ideal

PCA may not be ideal when:

- Direct interpretation of the original features is very important.
- The first few components preserve only a small amount of the total variance.
- Reducing dimensions would remove too much useful information.

---

# 🏆 Final Conclusion

PCA was successfully applied to the Cardiac Patients dataset to explore dimensionality reduction.

The original data contained **22 numerical dimensions**.

To preserve approximately 95% of the variance:

```text
22 Dimensions
      ↓
     PCA
      ↓
17 Principal Components
      ↓
≈ 94.98% Variance Preserved
```

For visualization:

```text
22 Dimensions
      ↓
     PCA
      ↓
2 Principal Components
      ↓
≈ 25.24% Variance Preserved
```

Using 17 components provides a lower-dimensional representation while preserving most of the original variance.

Using only two components loses much more information, but makes it possible to visualize the high-dimensional dataset in a 2D scatter plot.

Therefore, PCA provides a trade-off between **dimensionality reduction, information retention, visualization, and interpretability**.

---

# 🔑 Key Takeaways

- Each feature represents a dimension in the dataset.
- High-dimensional data can create problems known as the Curse of Dimensionality.
- PCA is used to reduce the number of dimensions.
- PCA creates new features called Principal Components.
- Principal Components are combinations of the original features.
- PC1 captures the largest amount of variance.
- Later components capture progressively smaller amounts of variance.
- PCA should generally be applied after feature scaling.
- Explained Variance Ratio shows how much variance each component preserves.
- Cumulative Explained Variance shows how much variance multiple components preserve together.
- 17 Principal Components preserved approximately 95% of the variance in this dataset.
- The first two Principal Components preserved approximately 25.24% of the variance.
- Two components are useful for 2D visualization but do not represent all the original information.
- PCA reduces dimensionality at the cost of some information and direct feature interpretability.

---
