أكيد. هذا **README لـ Day 2 فقط**، ومبني على اللي درسناه وطبقناه فعليًا في المشروع، وجاهز للنسخ كـMarkdown واحد:

````markdown
# Day 2 — DBSCAN & Hierarchical Clustering

## 📌 Overview

In Day 2, I explored alternative clustering methods to overcome some of the limitations of **K-Means**.

I applied two new clustering techniques to the Cardiac Patients dataset:

- **DBSCAN**
- **Hierarchical Clustering**

The main goal was to understand how different clustering algorithms discover structure in the same dataset and compare their results.

In this lesson, I learned how to:

- Understand the limitations of K-Means.
- Apply DBSCAN clustering.
- Understand the `eps` and `min_samples` parameters.
- Identify clusters and noise points using DBSCAN.
- Visualize DBSCAN clusters.
- Apply Hierarchical Clustering.
- Build and interpret a Dendrogram.
- Understand cluster merging.
- Choose a Cut Height.
- Compare K-Means, DBSCAN, and Hierarchical Clustering.
- Select the most suitable clustering method for the dataset.

---

# 🎯 Learning Objectives

By the end of this lesson, I was able to:

1. Explain the main limitations of K-Means.
2. Understand when DBSCAN can be preferred over K-Means.
3. Run DBSCAN and interpret its clusters and noise points.
4. Understand how `eps` and `min_samples` affect DBSCAN.
5. Understand how Hierarchical Clustering builds clusters.
6. Build and interpret a Dendrogram.
7. Choose a Cut Height from a Dendrogram.
8. Compare different clustering methods on the same dataset.

---

# ⚠️ Limitations of K-Means

K-Means is useful for clustering, but it has several limitations.

### 1. K-Means Requires `k`

The number of clusters must be selected before training.

For example:

```python
KMeans(n_clusters=2)
```

Methods such as the Elbow Method and Silhouette Score can help choose `k`, but K-Means still requires the number of clusters in advance.

### 2. K-Means Works Best with Compact Clusters

K-Means generally works best when clusters are relatively compact and similarly sized.

It may produce misleading results when the natural clusters have irregular shapes.

### 3. K-Means Forces Every Point into a Cluster

K-Means does not have a special label for noise or outliers.

Every observation must belong to one of the available clusters.

This can be a problem when the dataset contains unusual or isolated observations.

---

# 🔍 DBSCAN

**DBSCAN — Density-Based Spatial Clustering of Applications with Noise** is a clustering algorithm based on the density of observations.

Instead of using centroids like K-Means, DBSCAN searches for regions where observations are densely packed.

DBSCAN has two important advantages:

- It does not require the number of clusters to be specified in advance.
- It can identify observations that do not belong to any dense cluster as noise.

---

# ⚙️ Preparing the Data

The target variable was removed because clustering is an Unsupervised Learning task.

```python
X = df.drop("has_heart_disease", axis=1)
```

Only numerical features were selected:

```python
X_numeric = X.select_dtypes(include="number")
```

The numerical features were then scaled using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_numeric)
```

Scaling is important because DBSCAN uses distances between observations.

---

# 🔧 DBSCAN Parameters

DBSCAN mainly depends on two parameters:

## `eps`

`eps` controls the maximum distance between observations for them to be considered neighbors.

A smaller `eps` creates smaller neighborhoods and can result in more noise points.

A larger `eps` creates larger neighborhoods and allows more observations to become connected.

---

## `min_samples`

`min_samples` controls how many nearby observations are required to form a dense region.

For example:

```python
min_samples=6
```

requires enough nearby observations for DBSCAN to consider an area sufficiently dense.

Together:

```text
eps
↓
How close should observations be?

min_samples
↓
How many nearby observations are required?
```

---

# 🤖 Applying DBSCAN

DBSCAN was applied to the scaled numerical data:

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=2.9,
    min_samples=6
)

labels = db.fit_predict(X_scaled)
```

`fit_predict()` performs the clustering and returns the cluster label assigned to every observation.

The cluster distribution was inspected using:

```python
pd.Series(labels).value_counts()
```

The result was:

```text
-1    5655
 0    3147
 3     113
 1      43
 9       7
10       6
 8       6
 6       6
 7       6
 4       5
 2       3
 5       3
```

---

# 🔊 DBSCAN Noise

DBSCAN uses:

```text
-1
```

to represent **Noise Points**.

Therefore:

```text
-1 → Noise
0  → Cluster
1  → Cluster
2  → Cluster
...
```

The `-1` label is not another cluster.

In this experiment, DBSCAN identified:

```text
11 Clusters
5655 Noise Points
```

The largest cluster was:

```text
Cluster 0 → 3147 patients
```

---

# 🧠 DBSCAN Result Interpretation

Using `eps=2.9` and `min_samples=6`, DBSCAN identified **11 clusters** and **5,655 noise points**.

A large portion of the dataset was classified as noise, while Cluster 0 was the largest cluster with 3,147 patients.

This indicates that with these DBSCAN parameters, the dataset does not form a small number of clearly dense clusters.

---

# 📊 Visualizing DBSCAN

The DBSCAN results can be visualized using two numerical features.

For example:

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    df["age"],
    df["daily_steps"],
    c=labels
)

plt.xlabel("Age")
plt.ylabel("Daily Steps")
plt.title("DBSCAN Clusters")

plt.show()
```

The color of each observation represents its DBSCAN cluster label.

The visualization uses only two features for display, while DBSCAN itself was trained using all selected numerical features in `X_scaled`.

---

# 🌳 Hierarchical Clustering

**Hierarchical Clustering** builds clusters in a hierarchical structure.

It starts with individual observations and gradually merges the closest observations or groups.

The process can be thought of as:

```text
Individual observations
        ↓
Merge closest observations
        ↓
Small clusters
        ↓
Merge closest clusters
        ↓
Larger clusters
        ↓
One complete hierarchy
```

Unlike K-Means, Hierarchical Clustering does not require choosing the final number of clusters before building the hierarchy.

---

# 📦 Sampling the Dataset

The dataset contains 9,000 patients.

A Dendrogram containing all observations would be very large and difficult to interpret.

Therefore, a sample of 500 observations was used:

```python
X_sample = X_scaled[:500]
```

---

# 🔗 Building the Hierarchy

The hierarchical structure was created using `linkage`:

```python
from scipy.cluster.hierarchy import linkage

Z = linkage(
    X_sample,
    method="ward"
)
```

The `ward` method determines which clusters should be merged while trying to keep the resulting groups compact.

---

# 🌲 Hierarchical Clustering Dendrogram

The hierarchical structure was visualized using a Dendrogram:

```python
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Patients")
plt.ylabel("Distance")

plt.show()
```

The Dendrogram shows how observations and smaller clusters are gradually merged based on similarity.

The height of each merge represents the distance between the groups being combined.

A higher merge indicates that the groups being combined are more different from each other.

---

# 🔍 Dendrogram Interpretation

The Dendrogram showed a noticeable gap between two major merging levels:

```text
Approximately 30
        ↓
Large vertical gap
        ↓
Approximately 43
```

This gap suggests that a natural separation may exist before the final major merge.

---

# ✂️ Choosing a Cut Height

A Cut Height was selected inside the large gap between the major merges.

The selected Cut Height was:

```text
35
```

It was visualized using:

```python
plt.figure(figsize=(12, 6))

dendrogram(Z)

plt.axhline(
    y=35,
    color="red",
    linestyle="--"
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Patients")
plt.ylabel("Distance")

plt.show()
```

The value `35` is not a special fixed value.

It was selected because it lies between the major merges at approximately `30` and `43`.

Other values clearly inside the same gap could produce the same cluster structure.

---

# 🧠 Hierarchical Clustering Interpretation

A cut height of approximately **35** divides the Dendrogram into **2 main clusters**.

This suggests that a two-cluster structure may be a reasonable representation of the sampled data.

It is important to note that the Dendrogram was created using a sample of **500 patients**, not the complete dataset of 9,000 patients.

---

# 🔄 Comparing the Clustering Methods

Three clustering approaches were compared:

| Method | Result |
|---|---|
| K-Means | 2 clusters |
| DBSCAN | 11 clusters + 5,655 noise points |
| Hierarchical Clustering | 2 main clusters on the 500-patient sample |

---

## K-Means

K-Means divided the complete dataset into **2 relatively balanced clusters**:

```text
Cluster 0 → 4369 patients
Cluster 1 → 4631 patients
```

Every patient was assigned to one of the two clusters.

---

## DBSCAN

Using:

```text
eps = 2.9
min_samples = 6
```

DBSCAN produced:

```text
11 Clusters
5655 Noise Points
```

A large proportion of the dataset was classified as noise, and several of the discovered clusters contained only a small number of observations.

---

## Hierarchical Clustering

The Dendrogram created from the sample of 500 patients showed a large separation between major merges.

Using a Cut Height of approximately `35` resulted in:

```text
2 Main Clusters
```

This result supports the possibility of a two-cluster structure in the sampled data.

---

# 🏆 Final Conclusion

Based on the clustering results, **K-Means appears to be the most suitable method among the three methods tested for this dataset**.

K-Means produced two relatively balanced clusters:

```text
Cluster 0 → 4369 patients
Cluster 1 → 4631 patients
```

Hierarchical Clustering also suggested a possible **two-cluster structure** in the sampled data.

DBSCAN, using `eps=2.9` and `min_samples=6`, classified **5,655 patients as noise** and produced several small clusters.

Therefore, based on the experiments performed in this lesson, K-Means provided the clearest and most practical clustering result.

This does not mean that DBSCAN is generally worse than K-Means. Its performance depends strongly on the structure of the data and the selected `eps` and `min_samples` values.

---

# 📊 When to Use Each Method

| Method | Best When | Limitation |
|---|---|---|
| **K-Means** | Clusters are compact and `k` is roughly known | Requires `k` and forces every observation into a cluster |
| **DBSCAN** | Data contains noise or irregularly shaped clusters | Sensitive to `eps` and `min_samples` |
| **Hierarchical Clustering** | We want to explore nested cluster structure using a Dendrogram | Can become slow and difficult to visualize with large datasets |

---

# 🔑 Key Takeaways

- K-Means is not suitable for every clustering problem.
- K-Means requires the number of clusters `k` in advance.
- K-Means forces every observation into a cluster.
- DBSCAN groups observations based on density.
- DBSCAN does not require specifying the number of clusters.
- DBSCAN can identify Noise Points using the label `-1`.
- `eps` controls the neighborhood distance in DBSCAN.
- `min_samples` controls the minimum number of observations required for a dense region.
- Hierarchical Clustering gradually merges similar observations and clusters.
- A Dendrogram visualizes the hierarchical merging process.
- Horizontal connections in a Dendrogram represent cluster merges.
- The height of a merge represents the distance at which the groups were merged.
- A Cut Height can be used to select a final cluster structure.
- K-Means and Hierarchical Clustering suggested a two-cluster structure in this experiment.
- DBSCAN produced many noise points with the parameters tested.
- Different clustering algorithms can produce very different results on the same dataset.

---

