# Day 4 — t-SNE & Anomaly Detection

## 📌 Overview

In Day 4, I explored two important Unsupervised Learning techniques:

- **t-SNE (t-distributed Stochastic Neighbor Embedding)** for high-dimensional data visualization.
- **Isolation Forest** for Anomaly Detection.

The main goal was to visualize the local structure of the Cardiac Patients dataset using t-SNE and detect unusual observations using Isolation Forest.

In this lesson, I learned how to:

- Apply t-SNE to high-dimensional data.
- Visualize data in two dimensions using t-SNE.
- Understand the `perplexity` parameter.
- Compare PCA and t-SNE.
- Understand what Anomaly Detection is.
- Understand why Anomaly Detection is often unsupervised.
- Apply Isolation Forest.
- Understand the `contamination` parameter.
- Identify normal and anomalous observations.
- Inspect flagged anomalies and hypothesize why they were detected.
- Understand the relationship between anomaly detection and clustering.

---

# 🎯 Learning Objectives

By the end of this lesson, I was able to:

1. Use t-SNE to visualize high-dimensional data.
2. Understand how t-SNE preserves local neighborhoods.
3. Distinguish between PCA and t-SNE.
4. Explain what Anomaly Detection is.
5. Explain why Anomaly Detection is often an Unsupervised Learning problem.
6. Apply Isolation Forest to a dataset.
7. Understand the `contamination` parameter.
8. Interpret the output of Isolation Forest.
9. Inspect anomalous observations and compare them with the overall dataset.
10. Understand the connection between DBSCAN noise points and anomaly detection.

---

# 🧠 t-SNE

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a dimensionality reduction technique mainly used for visualization.

Like PCA, t-SNE can transform high-dimensional data into two dimensions.

For example:

```text
22 Dimensions
      ↓
    t-SNE
      ↓
t-SNE 1
t-SNE 2
```

However, PCA and t-SNE have different goals.

PCA focuses on preserving the overall variance of the data.

t-SNE focuses on preserving **Local Neighborhoods**, meaning that observations that were similar in the original high-dimensional space should remain relatively close to each other in the 2D visualization.

This makes t-SNE useful for visually discovering groups and local patterns in complex datasets.

---

# 🔍 PCA vs. t-SNE

| PCA | t-SNE |
|---|---|
| Preserves global variance | Preserves local neighborhoods |
| Used for dimensionality reduction and visualization | Mainly used for visualization |
| Fast | Usually slower |
| Components represent directions of variance | Axes do not have direct meaning |
| Can be useful for downstream modeling | Mainly used for exploring and visualizing data |

The values of `t-SNE 1` and `t-SNE 2` should not be interpreted as meaningful original features.

The important information in a t-SNE visualization is the **relative position of the observations**.

---

# Step 1 — t-SNE 2D Visualization

## Apply t-SNE to the Scaled Data

The scaled numerical data prepared in the previous lessons was used as the input for t-SNE.

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)
```

---

## Understanding `n_components`

The parameter:

```python
n_components=2
```

instructs t-SNE to transform the high-dimensional dataset into two dimensions.

```text
Original Dimensions
        ↓
      t-SNE
        ↓
t-SNE Dimension 1
t-SNE Dimension 2
```

Two dimensions were selected because the goal was to visualize the data using a 2D scatter plot.

---

## Understanding `perplexity`

The parameter:

```python
perplexity=30
```

controls the scale of the local neighborhoods considered by t-SNE.

It influences how t-SNE balances very local relationships with somewhat broader structure when arranging observations in the 2D space.

The value `30` was used in this experiment.

---

## Understanding `random_state`

t-SNE involves randomness during its optimization process.

```python
random_state=42
```

was used to make the experiment more reproducible when using the same data and settings.

---

## Check the Shape of the t-SNE Data

After applying t-SNE, the shape of the transformed dataset was checked:

```python
print(X_tsne.shape)
```

The result was:

```text
(9000, 2)
```

This means that the original dataset containing 9,000 patients was transformed so that every patient is now represented by two t-SNE coordinates.

```text
9000 Patients
      ×
2 t-SNE Dimensions
```

---

## Visualize t-SNE Using K-Means Clusters

The t-SNE representation was visualized using a scatter plot.

The points were colored according to the K-Means clusters obtained in Day 1.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=df["cluster"]
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of K-Means Clusters")

plt.colorbar(scatter, label="Cluster")

plt.show()
```

In the plot:

```text
X_tsne[:, 0] → t-SNE Dimension 1
X_tsne[:, 1] → t-SNE Dimension 2
```

The color of each point represents its K-Means cluster.

It is important to note that **t-SNE did not create these clusters**.

The cluster labels were produced by K-Means, while t-SNE was only used to visualize their structure in two dimensions.

---

# 📊 t-SNE Visualization Interpretation

The t-SNE visualization revealed local structure in the dataset and showed how the K-Means clusters were distributed in two dimensions.

The two clusters showed noticeable separation in several regions, although some overlap between them was still present.

Unlike PCA, the t-SNE axes do not have a direct interpretation.

The important information is the relative position of the points and which observations appear close to each other.

---

# Step 2 — Compare PCA and t-SNE

## PCA vs. t-SNE Visualization

In Day 3, PCA was used to reduce the Cardiac Patients dataset to two Principal Components.

The first two PCA components preserved approximately:

```text
25.24% of the Total Variance
```

The PCA visualization showed the overall structure of the dataset, but considerable overlap was visible between the groups.

The t-SNE visualization revealed clearer local patterns and groupings because t-SNE focuses on preserving local neighborhoods.

---

## Comparison and Interpretation

The main difference observed between the two techniques was:

```text
PCA
↓
Focuses on Global Variance
↓
Useful for Dimensionality Reduction + Visualization

t-SNE
↓
Focuses on Local Neighborhoods
↓
Mainly Useful for Visualization
```

PCA is useful when dimensionality reduction is needed while preserving as much global variance as possible.

t-SNE is useful when the goal is to visually explore local relationships and possible group structures in high-dimensional data.

The t-SNE axes themselves do not have direct feature meanings.

Therefore, t-SNE should mainly be interpreted by looking at the relative positions and neighborhoods of the observations.

---

# 🚨 Anomaly Detection

**Anomaly Detection** is the process of identifying observations that differ significantly from the general pattern of the dataset.

Anomalies may represent unusual observations, rare events, errors, failures, fraud, or other uncommon patterns depending on the application.

Anomaly Detection is often an Unsupervised Learning problem because datasets frequently do not contain predefined labels such as:

```text
Normal
Anomaly
```

Instead, the algorithm learns the general structure of the data and identifies observations that appear unusual compared with the majority.

---

# 🌲 Isolation Forest

**Isolation Forest** is an algorithm designed specifically for Anomaly Detection.

The main idea is that unusual observations are generally easier to isolate from the rest of the dataset.

Consider:

```text
● ● ● ● ●
 ● ● ● ●
● ● ● ● ●


                         ★
```

The points represented by `●` form a dense group.

The point represented by `★` is far from the majority.

Isolation Forest repeatedly partitions the data.

An unusual observation often requires fewer partitions to become isolated than an observation located inside a dense region.

Therefore:

```text
Easy to isolate
      ↓
Potential Anomaly
```

---

# Step 3 — Anomaly Detection with Isolation Forest

## Apply Isolation Forest to the Dataset

Isolation Forest was applied to the scaled dataset.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

anomaly_labels = iso.fit_predict(X_scaled)
```

---

# ⚙️ Understanding `contamination`

The parameter:

```python
contamination=0.05
```

specifies the expected proportion of anomalous observations.

A value of:

```text
0.05 = 5%
```

was used.

Since the dataset contains:

```text
9000 Patients
```

5% corresponds to:

```text
9000 × 0.05 = 450
```

Therefore, approximately 450 observations were expected to be classified as anomalies.

---

# 🏷️ Isolation Forest Labels

Isolation Forest returns two possible labels:

```text
 1 → Normal
-1 → Anomaly
```

A label of `-1` does not automatically mean that the observation is incorrect or medically abnormal.

It means that the observation has an unusual feature pattern compared with most observations in the dataset.

---

## Count Normal and Anomalous Points

The number of observations in each category was calculated using:

```python
pd.Series(anomaly_labels).value_counts()
```

The result was:

```text
 1    8550
-1     450
```

Therefore, Isolation Forest identified:

```text
Normal Points  → 8550
Anomalies      → 450
```

---

# 📊 Isolation Forest Results

Isolation Forest identified:

- **8,550 normal points**
- **450 anomalous points**

The model used `contamination=0.05`, meaning approximately 5% of the dataset was expected to be anomalous.

The flagged points are observations that Isolation Forest considers unusual compared with the overall structure of the dataset.

They are not necessarily errors or incorrect records.

---

# Step 4 — Inspect Two Flagged Anomalies

To understand the detected anomalies better, the Isolation Forest labels were added to the original dataset.

---

## Add the Anomaly Labels to the Dataset

```python
df["anomaly"] = anomaly_labels
```

The new column contains:

```text
 1 → Normal
-1 → Anomaly
```

for every patient.

---

## Extract the Anomalous Points

Only observations classified as anomalies were selected:

```python
anomalies = df[df["anomaly"] == -1]
```

The anomalous subset contains:

```text
450 Patients
```

---

## Inspect Two Flagged Anomalies

Two anomalous observations were inspected:

```python
anomalies.head(2)
```

The first two flagged observations were patients with DataFrame indices:

```text
Patient 17
Patient 19
```

To understand why these observations might have been flagged, their feature values were compared with the overall dataset statistics.

---

## Compare the Flagged Points with the Dataset Statistics

The descriptive statistics of the dataset were examined using:

```python
df.describe()
```

This provided values such as:

```text
Mean
Standard Deviation
Minimum
25th Percentile
Median
75th Percentile
Maximum
```

These statistics were used to identify features where the flagged observations differed substantially from the general dataset.

---

# 🔎 Analysis of Two Flagged Anomalies

## Anomaly 1 — Patient 17

Patient 17 contains several values that are unusual compared with the overall dataset.

For example:

- LDL is **191**, compared with a dataset mean of approximately **103.28**.
- Total cholesterol is **272**, compared with a mean of approximately **188.75**.
- Fasting blood sugar is **155**, compared with a mean of approximately **119.47**.
- HbA1c is **7.2**, compared with a mean of approximately **5.79**.
- BMI is **31.3**, compared with a mean of approximately **25.26**.

The combination of several relatively high values may make this patient different from the majority of observations, which could explain why Isolation Forest flagged this point as an anomaly.

---

## Anomaly 2 — Patient 19

Patient 19 shows a different pattern of unusual values.

For example:

- BMI is **15.9**, compared with a dataset mean of approximately **25.26**.
- Maximum heart rate achieved is **209**, compared with a mean of approximately **164.84** and a maximum of **210**.
- HDL is **86**, compared with a mean of approximately **55.22**.
- Fasting blood sugar is **64**, compared with a mean of approximately **119.47**.
- HbA1c is **4.4**, compared with a mean of approximately **5.79**.

The combination of very low and very high feature values makes this observation unusual compared with the general pattern of the dataset, which may explain why Isolation Forest flagged it as an anomaly.

---

# 🧠 Anomaly Interpretation

Isolation Forest does not necessarily flag an observation because of one individual feature.

It considers the overall combination of features.

Therefore, the explanations for Patient 17 and Patient 19 are **hypotheses** based on comparing their values with the overall dataset statistics.

Being classified as an anomaly does not necessarily mean that:

```text
The observation is incorrect
OR
The patient is medically abnormal
```

It means that:

```text
The combination of feature values is unusual
compared with most observations in this dataset.
```

---

# 🔗 Anomaly Detection and Clustering

Anomaly Detection and Clustering are related Unsupervised Learning tasks.

In Day 2, DBSCAN used:

```text
-1 → Noise
```

for observations that did not belong to sufficiently dense regions.

In Day 4, Isolation Forest also used:

```text
-1 → Anomaly
```

for observations considered unusual compared with the overall data.

However, the methods identify unusual observations differently.

```text
DBSCAN
↓
Looks at Density
↓
Points outside dense regions may become Noise

Isolation Forest
↓
Attempts to isolate observations
↓
Points that are easier to isolate may become Anomalies
```

Therefore, DBSCAN noise points can be viewed as one form of detecting unusual observations during clustering, while Isolation Forest is specifically designed for anomaly detection.

---

# 🏆 Final Conclusion

Day 4 introduced two different ways of exploring hidden structure in high-dimensional data.

### t-SNE

t-SNE transformed the high-dimensional Cardiac Patients dataset into two dimensions for visualization.

The visualization revealed local patterns and showed the distribution of the K-Means clusters more clearly.

Unlike PCA, t-SNE focuses primarily on preserving local neighborhoods rather than global variance.

### Isolation Forest

Isolation Forest was used to identify unusual observations.

Using:

```text
contamination = 0.05
```

the model classified:

```text
8550 → Normal
450  → Anomalies
```

Two flagged observations were inspected and compared with the overall dataset statistics.

Both contained combinations of feature values that differed from the general pattern of the dataset.

This demonstrated that anomaly detection can identify unusual observations without requiring predefined anomaly labels.

---

# 🔑 Key Takeaways

- t-SNE is mainly a visualization technique for high-dimensional data.
- t-SNE attempts to preserve local neighborhoods.
- Observations that are similar in the original feature space should remain relatively close in the t-SNE visualization.
- The t-SNE axes do not have direct feature meanings.
- PCA and t-SNE have different goals.
- PCA focuses on global variance.
- t-SNE focuses on local neighborhoods.
- PCA can be used for dimensionality reduction and visualization.
- t-SNE is mainly used for visualization.
- Anomaly Detection identifies observations that differ from the general pattern of the data.
- Anomaly Detection is often unsupervised because anomaly labels may not be available.
- Isolation Forest isolates unusual observations using random partitions.
- `contamination` specifies the expected proportion of anomalies.
- Isolation Forest uses `1` for normal observations and `-1` for anomalies.
- With `contamination=0.05`, 450 of the 9,000 observations were flagged as anomalies.
- Anomaly labels do not automatically mean that the observations are incorrect or medically abnormal.
- Multiple unusual feature values together can cause an observation to appear anomalous.
- DBSCAN noise detection and Isolation Forest anomaly detection are related but use different approaches.

---