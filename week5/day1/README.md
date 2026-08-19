# Day 1 — Unsupervised Learning & K-Means Clustering

## 📌 Overview

In Day 1, I learned the fundamentals of **Unsupervised Learning** and applied **K-Means Clustering** to the Cardiac Patients dataset.

Unlike Supervised Learning, Unsupervised Learning does not use a target variable (`y`). Instead, the goal is to discover hidden patterns and natural groups in the data.

In this lesson, I learned how to:

- Understand the difference between Supervised and Unsupervised Learning.
- Understand what Clustering does.
- Apply K-Means Clustering.
- Understand clusters and centroids.
- Scale features before clustering.
- Choose the number of clusters using the Elbow Method.
- Evaluate candidate cluster numbers using the Silhouette Score.
- Assign each patient to a cluster.
- Visualize the resulting clusters.
- Interpret the discovered clusters.

---

# 🎯 Learning Objectives

By the end of this lesson, I was able to:

1. Explain how Unsupervised Learning differs from Supervised Learning.
2. Explain the purpose of Clustering.
3. Apply K-Means to numerical data.
4. Understand how K-Means assigns observations to clusters.
5. Choose an appropriate value of `k`.
6. Use the Elbow Method and Silhouette Score.
7. Interpret the resulting clusters.

---

# 🔍 Supervised vs. Unsupervised Learning

In **Supervised Learning**, the dataset contains features `X` and a known target `y`.

The model learns the relationship:

```text
X → y
```

Examples include:

- Regression
- Classification

In **Unsupervised Learning**, there is no target `y`.

The algorithm receives only:

```text
X
```

and tries to discover hidden structures or patterns in the data.

Examples include:

- Clustering
- Dimensionality Reduction
- Anomaly Detection

---

# 🧩 Clustering

Clustering is an Unsupervised Learning technique that groups similar observations together.

The main idea is:

```text
Similar observations
        ↓
Same Cluster

Different observations
        ↓
Different Clusters
```

In this project, clustering was used to discover natural groups of patients based on their numerical health and lifestyle features.

---

# 🎯 Removing the Target

The target variable was removed before clustering:

```python
X = df.drop("has_heart_disease", axis=1)
```

The `has_heart_disease` column was not used because K-Means is an Unsupervised Learning algorithm and should discover groups without knowing the correct heart disease label.

---

# 🔢 Selecting Numerical Features

For this exercise, K-Means was applied to the numerical features only.

```python
X_numeric = X.select_dtypes(include="number")
```

Categorical features were excluded from this initial K-Means analysis.

---

# ⚖️ Feature Scaling

K-Means is a distance-based algorithm.

Some features in the dataset have very different numerical ranges.

For example:

```text
Age         → relatively small values
Daily Steps → values in the thousands
BMI         → relatively small values
```

Without scaling, features with larger numerical ranges could have too much influence on the distance calculations.

Therefore, `StandardScaler` was applied before K-Means:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_numeric)
```

This places the numerical features on comparable scales before calculating distances.

---

# 🤖 K-Means Clustering

K-Means divides observations into a predefined number of clusters called `k`.

The algorithm repeatedly performs the following process:

1. Initialize `k` centroids.
2. Assign every observation to its nearest centroid.
3. Move each centroid to the mean position of the observations assigned to it.
4. Repeat the assignment and centroid update steps until the clusters stabilize.

Example:

```python
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

labels = km.fit_predict(X_scaled)
```

`fit_predict()` performs two main operations:

- Fits K-Means to the data.
- Returns the cluster assigned to each observation.

The returned labels may look like:

```text
[0, 1, 1, 0, 0, 1, ...]
```

These numbers represent cluster IDs and are not target labels.

For example:

```text
Cluster 0 ≠ No Heart Disease
Cluster 1 ≠ Heart Disease
```

The cluster numbers are simply identifiers created by K-Means.

---

# 📍 Centroids

A **Centroid** represents the center of a cluster.

K-Means assigns observations to the nearest centroid and continuously updates the centroid positions during training.

The final centroid positions can be accessed using:

```python
km.cluster_centers_
```

Each cluster has its own centroid.

---

# 📉 Choosing k Using the Elbow Method

K-Means requires the number of clusters `k` to be selected before training.

To help choose `k`, the Elbow Method was used.

K-Means was tested using values from:

```text
k = 1
to
k = 10
```

```python
inertias = []

for k in range(1, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    inertias.append(km.inertia_)
```

`inertia_` measures how close observations are to the centroids of their assigned clusters.

Lower inertia means observations are generally closer to their cluster centers.

The results were visualized:

```python
plt.plot(
    range(1, 11),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()
```

The goal is to look for the **elbow**, where increasing `k` starts producing smaller improvements in inertia.

---

# 📏 Silhouette Score

The Silhouette Score was used as an additional method for evaluating candidate values of `k`.

```python
from sklearn.metrics import silhouette_score

labels = km.fit_predict(X_scaled)

score = silhouette_score(X_scaled, labels)
```

The Silhouette Score evaluates how well observations fit inside their assigned clusters compared with other clusters.

Its values range approximately from:

```text
-1 → Poor clustering
 0 → Overlapping clusters
+1 → Well-separated clusters
```

A higher Silhouette Score generally indicates better-defined clusters.

The Elbow Method and Silhouette Score were used together to choose the final number of clusters.

---

# 🏁 Final K-Means Model

After comparing candidate values of `k`, the final K-Means model was trained using:

```text
k = 2
```

The final model assigned every patient to one of two clusters.

The cluster labels were added to the dataset:

```python
df["cluster"] = cluster_labels
```

The cluster distribution was:

```text
Cluster 1 → 4631 patients
Cluster 0 → 4369 patients
```

Total:

```text
9000 patients
```

---

# 📊 Cluster Visualization

The clusters were visualized using a 2D Scatter Plot with:

```text
X-axis → Age
Y-axis → Daily Steps
Color  → Cluster
```

Example:

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    df["age"],
    df["daily_steps"],
    c=df["cluster"]
)

plt.xlabel("Age")
plt.ylabel("Daily Steps")
plt.title("K-Means Clusters")

plt.show()
```

The following parameter:

```python
c=df["cluster"]
```

tells Matplotlib to color each observation according to its assigned cluster.

The colors themselves are selected automatically by Matplotlib.

---

# 🧠 Cluster Interpretation

The K-Means model divided the 9,000 patients into two groups:

- **Cluster 0:** 4,369 patients.
- **Cluster 1:** 4,631 patients.

The 2D visualization using Age and Daily Steps showed noticeable differences between the groups, but also showed overlap between them.

It is important to remember that the K-Means model was trained using all selected numerical features, while the 2D Scatter Plot displays only **Age and Daily Steps**.

Therefore, the visualization provides only one view of the discovered cluster structure.

The cluster IDs `0` and `1` do not represent predefined medical classes. They are groups discovered automatically by K-Means based on similarities between patients.

---

# 🔑 Key Takeaways

- Unsupervised Learning works without a target variable `y`.
- Clustering discovers natural groups within data.
- K-Means groups observations based on distance.
- `k` represents the number of clusters.
- Centroids represent the centers of the clusters.
- Feature Scaling is important because K-Means depends on distance.
- The Elbow Method helps identify candidate values for `k`.
- Silhouette Score helps evaluate how well the clusters are separated.
- `fit()` trains K-Means.
- `fit_predict()` trains K-Means and returns the cluster of each observation.
- Cluster labels such as `0` and `1` are identifiers, not true target labels.
- A 2D Scatter Plot can visualize clusters using two features, even when the clustering algorithm was trained using more features.

---