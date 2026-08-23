# Week 5 — Unsupervised Learning & Phase 3 Capstone Planning

## 📌 Overview

During Week 5, I explored several important **Unsupervised Learning** techniques and applied them to the **Cardiac Patients** dataset.

The week covered:

- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- t-SNE
- Anomaly Detection with Isolation Forest
- Phase 3 Capstone Project Planning
- Sprint 1 Planning

The main goal was to understand how Machine Learning can discover hidden patterns in data without relying on predefined target labels.

At the end of the week, the Cardiac Patients project was also selected and structured as the Phase 3 Capstone Project.

---

# 📅 Week Structure

| Day | Topic |
|---|---|
| Day 1 | Unsupervised Learning & K-Means |
| Day 2 | DBSCAN & Hierarchical Clustering |
| Day 3 | Dimensionality Reduction with PCA |
| Day 4 | t-SNE & Anomaly Detection |
| Day 5 | Phase 3 Project Selection & Sprint 1 Planning |

---

# Day 1 — Unsupervised Learning & K-Means

## 📌 Overview

Day 1 introduced **Unsupervised Learning** and how it differs from Supervised Learning.

In Supervised Learning, the model learns from labeled data:

```text
X + y
```

In Unsupervised Learning, there is no target variable used to guide the algorithm:

```text
X only
```

The goal is not to predict a known answer, but to discover hidden structures and patterns in the data.

---

## Supervised vs. Unsupervised Learning

### Supervised Learning

```text
Data → X + y
Goal → Predict a known target
Examples → Regression, Classification
```

### Unsupervised Learning

```text
Data → X only
Goal → Discover hidden structure
Examples → Clustering, PCA, Anomaly Detection
```

---

## 🔵 Clustering

**Clustering** groups observations according to their similarity.

The objective is:

```text
Similar observations
        ↓
Same Cluster

Different observations
        ↓
Different Clusters
```

No predefined cluster labels are required.

The algorithm discovers the groups from the structure of the data.

---

# K-Means Clustering

K-Means divides observations into a predefined number of clusters called `k`.

The algorithm repeatedly performs the following process:

```text
Choose k Centroids
        ↓
Assign Each Point to the Nearest Centroid
        ↓
Calculate New Centroids
        ↓
Repeat Until Stable
```

---

## Scaling Before Clustering

K-Means relies on distance calculations.

Therefore, numerical features were standardized before clustering.

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_numeric)
```

Scaling prevents features with large numerical ranges from dominating the distance calculations.

---

## Choosing K — Elbow Method

K-Means requires the number of clusters to be specified in advance.

The **Elbow Method** was used to evaluate different values of `k`.

```python
from sklearn.cluster import KMeans

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

The inertia values were plotted:

```python
import matplotlib.pyplot as plt

plt.plot(
    range(1, 11),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()
```

The elbow in the graph helps identify a reasonable number of clusters.

---

## Silhouette Score

The **Silhouette Score** was also used to evaluate clustering quality.

It ranges approximately from:

```text
-1 → Poor clustering
 0 → Overlapping clusters
+1 → Well-separated clusters
```

A higher score generally indicates better-defined clusters.

---

## Final K-Means Model

The final K-Means model divided the dataset into two clusters.

The cluster distribution was:

```text
Cluster 1 → 4631 patients
Cluster 0 → 4369 patients
```

The cluster labels were added to the dataset:

```python
df["cluster"] = labels
```

---

## K-Means Interpretation

The two clusters represent groups of patients with different combinations of health and lifestyle characteristics.

The cluster number itself does not have an inherent meaning.

For example:

```text
Cluster 0
```

does not automatically mean healthy or unhealthy.

The clusters must be interpreted by examining the feature values of the observations within each group.

---

# Day 2 — DBSCAN & Hierarchical Clustering

## 📌 Overview

Day 2 explored alternatives to K-Means.

K-Means has several limitations:

- The number of clusters must be selected in advance.
- It works best with roughly round clusters.
- It forces every observation into a cluster.
- Outliers are not explicitly identified.

Two alternative clustering techniques were explored:

- DBSCAN
- Hierarchical Clustering

---

# DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups observations according to density.

Unlike K-Means, DBSCAN can:

- Discover clusters without specifying `k`.
- Detect irregularly shaped clusters.
- Identify noise points.

DBSCAN returns:

```text
0, 1, 2, ... → Clusters
-1            → Noise
```

---

## DBSCAN Parameters

Two important parameters control DBSCAN.

### eps

```text
eps
```

controls how close observations must be to be considered neighbors.

### min_samples

```text
min_samples
```

controls how many nearby observations are required to form a dense region.

---

## Applying DBSCAN

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=2.9,
    min_samples=6
)

labels = db.fit_predict(X_scaled)

pd.Series(labels).value_counts()
```

The experiment produced multiple small clusters and a large number of noise observations.

Example result:

```text
-1    5655
 0    3147
 3     113
 1      43
 ...
```

This demonstrated that DBSCAN behaved very differently from K-Means on the Cardiac Patients dataset.

---

## DBSCAN Interpretation

A label of:

```text
-1
```

means DBSCAN considered the observation to be **Noise** because it was not located in a sufficiently dense region according to the selected `eps` and `min_samples`.

DBSCAN was highly sensitive to the selected parameter values.

Changing `eps` or `min_samples` produced significantly different clustering results.

---

# Hierarchical Clustering

**Hierarchical Clustering** creates a hierarchy of clusters.

The process starts with observations separated and progressively merges the closest groups.

```text
Individual Observations
        ↓
Small Groups
        ↓
Larger Groups
        ↓
One Complete Hierarchy
```

---

## Creating the Hierarchy

```python
from scipy.cluster.hierarchy import linkage

Z = linkage(
    X_scaled,
    method="ward"
)
```

The `linkage` function calculates how observations or clusters should be merged.

---

## Dendrogram

The hierarchy was visualized using a **Dendrogram**.

```python
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(12, 6))

dendrogram(Z)

plt.xlabel("Samples")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")

plt.show()
```

The vertical axis represents the distance at which clusters are merged.

A horizontal cut can be placed at a selected height.

The number of vertical branches crossed by the cut represents the resulting number of clusters.

---

# Comparing Clustering Methods

| Method | Main Strength | Limitation |
|---|---|---|
| K-Means | Simple and efficient | Requires `k` |
| DBSCAN | Detects noise and irregular shapes | Sensitive to parameters |
| Hierarchical | Shows nested cluster structure | Expensive for large datasets |

Each clustering algorithm analyzes the structure of the data differently.

---

# Day 3 — Dimensionality Reduction with PCA

## 📌 Overview

Day 3 introduced **Dimensionality Reduction** and **Principal Component Analysis (PCA)**.

The Cardiac Patients dataset contained:

```text
22 Numerical Dimensions
```

PCA was used to represent the dataset using fewer dimensions while preserving as much variance as possible.

---

# Curse of Dimensionality

As the number of features increases, datasets become increasingly high-dimensional.

High dimensionality can cause problems such as:

- Sparse data.
- Less meaningful distances.
- Increased model complexity.
- Higher risk of overfitting.
- Difficulty visualizing the data.

Dimensionality reduction attempts to represent the data using fewer dimensions.

---

# Principal Component Analysis

PCA creates new dimensions called:

**Principal Components**

Instead of selecting original features directly:

```text
Age
BMI
Cholesterol
Blood Pressure
...
```

PCA creates:

```text
PC1
PC2
PC3
...
```

Each Principal Component is a combination of the original features.

The components are ordered according to the amount of variance they capture.

---

## Scaling Before PCA

PCA is variance-based.

Therefore, the data was standardized before applying PCA.

```python
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_numeric)
```

---

## Fit PCA

```python
from sklearn.decomposition import PCA

pca = PCA()

X_pca = pca.fit_transform(X_scaled)
```

---

# Explained Variance Ratio

The amount of variance captured by each component was inspected:

```python
print(pca.explained_variance_ratio_)
```

The first components produced:

```text
PC1 → 16.17%
PC2 → 9.07%
PC3 → 7.93%
...
```

PC1 captured the largest amount of variance.

---

# Cumulative Explained Variance

The cumulative variance was calculated:

```python
import numpy as np

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)
```

The analysis showed:

```text
15 Components → 90.12%
16 Components → 92.58%
17 Components → 94.98%
18 Components → 97.22%
```

Since the goal was to retain approximately 95% of the variance, **17 Principal Components** were selected.

---

## Reduce to 17 Components

```python
pca_17 = PCA(n_components=17)

X_pca_17 = pca_17.fit_transform(X_scaled)

print(
    pca_17.explained_variance_ratio_.sum()
)
```

The result was approximately:

```text
94.98%
```

Therefore:

```text
22 Dimensions
      ↓
     PCA
      ↓
17 Principal Components
      ↓
≈ 95% Variance Preserved
```

---

# PCA 2D Visualization

PCA was also applied using only two components for visualization.

```python
pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(X_scaled)
```

The first two components preserved:

```text
PC1 → 16.17%
PC2 → 9.07%

Total → 25.24%
```

---

## Visualization

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

The visualization showed some separation between patients with and without heart disease, especially along PC1.

However, considerable overlap remained.

Since the first two components preserved only approximately **25.24% of the total variance**, the visualization does not represent all information from the original dataset.

---

# PCA Trade-Off

Using 17 components:

```text
22 Dimensions
      ↓
17 Components
      ↓
≈ 95% Variance Preserved
```

Using two components:

```text
22 Dimensions
      ↓
2 Components
      ↓
≈ 25.24% Variance Preserved
```

PCA therefore provides a trade-off between:

- Dimensionality
- Information retention
- Visualization
- Interpretability

---

# Day 4 — t-SNE & Anomaly Detection

## 📌 Overview

Day 4 introduced:

- t-SNE for high-dimensional visualization.
- Anomaly Detection.
- Isolation Forest.

---

# t-SNE

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a dimensionality reduction technique mainly designed for visualization.

Unlike PCA, which focuses on global variance, t-SNE focuses on preserving **Local Neighborhoods**.

In simple terms:

```text
Points that are similar
in high-dimensional space
        ↓
t-SNE tries to keep them
close together in 2D
```

---

# Applying t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)
```

The resulting shape was:

```text
(9000, 2)
```

Therefore:

```text
9000 Patients
      ↓
t-SNE
      ↓
2 Dimensions
```

---

# t-SNE Visualization

The t-SNE points were colored using the K-Means clusters created in Day 1.

```python
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=df["cluster"]
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of K-Means Clusters")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.show()
```

t-SNE did not create these clusters.

K-Means created the cluster labels, while t-SNE was used only to visualize their structure.

---

# PCA vs. t-SNE

```text
PCA
↓
Global Variance
↓
Dimensionality Reduction + Visualization


t-SNE
↓
Local Neighborhoods
↓
Visualization
```

The PCA visualization showed the general structure of the dataset but contained considerable overlap.

The t-SNE visualization revealed clearer local structures and groupings.

The axes of t-SNE do not have direct meanings.

The important information is the relative position of observations.

---

# Anomaly Detection

**Anomaly Detection** identifies observations that differ significantly from the general pattern of the dataset.

Anomaly detection is often unsupervised because datasets may not contain labels specifying:

```text
Normal
Anomaly
```

Instead, the algorithm analyzes the structure of the data and attempts to identify unusual observations.

---

# Isolation Forest

**Isolation Forest** detects anomalies by attempting to isolate observations.

The main idea is:

```text
Normal Observation
→ Located near many observations
→ Harder to isolate

Anomaly
→ Different from most observations
→ Easier to isolate
```

---

## Applying Isolation Forest

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

anomaly_labels = iso.fit_predict(X_scaled)
```

Isolation Forest returns:

```text
 1 → Normal
-1 → Anomaly
```

---

# Contamination

The model used:

```python
contamination=0.05
```

This means approximately:

```text
5%
```

of the observations were expected to be anomalies.

With 9,000 observations:

```text
9000 × 0.05 = 450
```

---

# Isolation Forest Results

```python
pd.Series(anomaly_labels).value_counts()
```

Result:

```text
 1    8550
-1     450
```

Therefore:

```text
Normal Observations → 8550
Anomalies           → 450
```

---

# Inspecting Flagged Anomalies

The labels were added to the dataset:

```python
df["anomaly"] = anomaly_labels
```

Then the anomalous observations were extracted:

```python
anomalies = df[
    df["anomaly"] == -1
]
```

Two observations were inspected:

```python
anomalies.head(2)
```

The selected observations were:

```text
Patient 17
Patient 19
```

---

## Patient 17

Several values were noticeably different from the overall dataset averages.

Examples:

```text
LDL
Patient → 191
Mean    → 103.28

Total Cholesterol
Patient → 272
Mean    → 188.75

Fasting Blood Sugar
Patient → 155
Mean    → 119.47

HbA1c
Patient → 7.2
Mean    → 5.79

BMI
Patient → 31.3
Mean    → 25.26
```

The combination of several relatively high values may have contributed to this observation being flagged as unusual.

---

## Patient 19

This observation contained a different combination of unusual values.

Examples:

```text
BMI
Patient → 15.9
Mean    → 25.26

Maximum Heart Rate
Patient → 209
Mean    → 164.84
Maximum → 210

HDL
Patient → 86
Mean    → 55.22

Fasting Blood Sugar
Patient → 64
Mean    → 119.47

HbA1c
Patient → 4.4
Mean    → 5.79
```

The combination of unusually low and high values may have made the observation easier for Isolation Forest to isolate.

These explanations are hypotheses.

Being classified as an anomaly does not automatically mean that the observation is incorrect or medically abnormal.

It means that its feature pattern is unusual compared with most observations in the dataset.

---

# DBSCAN Noise vs. Isolation Forest Anomalies

Both methods can identify unusual observations, but they work differently.

```text
DBSCAN
↓
Density-Based
↓
Sparse Observations → Noise


Isolation Forest
↓
Isolation-Based
↓
Easy-to-Isolate Observations → Anomalies
```

---

# Day 5 — Phase 3 Project Selection & Sprint 1 Planning

## 📌 Overview

Day 5 focused on transitioning from the learning phase into the Phase 3 Capstone Project.

The Cardiac Patients project was selected as the capstone project and structured as an end-to-end Machine Learning project.

---

# Selected Capstone Project

## Cardiac Disease Risk Prediction and Patient Data Analysis

The project will focus on building an end-to-end Machine Learning system for analyzing cardiac patient data.

Patient health and lifestyle features will be used to predict heart disease risk using Supervised Learning.

The project will also include Unsupervised Learning analysis to explore:

- Hidden patterns
- Patient groups
- Dimensional structure
- Unusual observations

---

# Project Workflow

```text
Cardiac Patient Dataset
          ↓
         EDA
          ↓
Data Cleaning & Preprocessing
          ↓
Supervised Learning
          ↓
Model Evaluation
          ↓
Unsupervised Learning
          ↓
Final Model Selection
          ↓
Deployment
          ↓
Documentation
```

---

# Problem Statement

Heart disease risk can be associated with multiple health and lifestyle factors, making patient data complex to analyze manually.

The goal of this project is to build an end-to-end Machine Learning system that analyzes cardiac patient data and predicts whether a patient is at risk of heart disease.

The project will also use Unsupervised Learning techniques to explore hidden patterns, patient groups, dimensional structure, and unusual observations within the dataset.

The final system should provide reliable predictions, clear model evaluation, useful data analysis, and an accessible deployed application.

---

# Definition of Done

The project will be considered complete when the following requirements are satisfied:

- A clean and documented Jupyter Notebook covers the complete Machine Learning pipeline.
- The dataset is properly cleaned and preprocessed.
- Supervised Learning models are trained and evaluated.
- A baseline model is established.
- Improved models are compared against the baseline.
- Unsupervised Learning analysis is completed and documented.
- Final model evaluation metrics are clearly reported.
- A working application is deployed using Streamlit or FastAPI.
- The application is accessible through a public URL.
- The GitHub repository contains a clean project structure.
- A complete README is provided.
- A `requirements.txt` file is included.
- Model artifacts are included or properly managed.
- A short technical write-up explains the methodology, results, and conclusions.

---

# Sprint 1 Backlog

The goal of Sprint 1 is to understand the dataset and establish a baseline model that can be improved in later sprints.

---

## Task 1 — Dataset Selection and Understanding

**Estimated Effort:** 2 hours

### Acceptance Criteria

- Dataset is selected and loaded successfully.
- Dataset shape and columns are inspected.
- Feature meanings are understood.
- Target variable is identified.
- Missing values are inspected.
- Data types are inspected.
- Results are documented in Markdown.
- Notebook runs without errors.
- Work is committed to the correct feature branch.

---

## Task 2 — Exploratory Data Analysis

**Estimated Effort:** 3 hours

### Acceptance Criteria

- Numerical features are summarized.
- Important feature distributions are visualized.
- Target distribution is analyzed.
- Important relationships are explored.
- Potential outliers are inspected.
- Findings are documented in Markdown.
- Notebook runs without errors.
- Work is committed with a clear commit message.

---

## Task 3 — Data Preprocessing

**Estimated Effort:** 2 hours

### Acceptance Criteria

- Missing values are handled.
- Numerical and categorical features are identified.
- Categorical features are encoded when required.
- Numerical features are scaled when required.
- Target and input features are separated.
- Data leakage is avoided.
- Preprocessing is documented.
- Notebook runs without errors.
- Changes are committed to the correct feature branch.

---

## Task 4 — Baseline Model

**Estimated Effort:** 2 hours

### Acceptance Criteria

- Dataset is split appropriately.
- A baseline classification model is trained.
- Predictions are generated on unseen data.
- Appropriate classification metrics are calculated.
- Baseline results are documented.
- Metrics are stored for future comparisons.
- Notebook runs without errors.
- Changes are committed with a clear message.
- A Pull Request is opened before merging.

---

# Sprint 1 Goal

The goal of Sprint 1 is to understand and prepare the cardiac patient dataset and establish a baseline Machine Learning model for heart disease prediction.

By the end of the sprint, the dataset should be explored, cleaned, and preprocessed, and a baseline classification model should be trained and evaluated using appropriate metrics.

The baseline results will serve as a reference point for evaluating and improving the models developed in later sprints.

---

# Mentor Sign-Off

The project scope, backlog, acceptance criteria, and Sprint 1 goal should be reviewed and approved by the mentor before Phase 3 development begins.

```text
Status: Pending Mentor Approval
```

---

# GitHub Repository Structure

The planned project structure is:

```text
Cardiac-Patients/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── modeling.ipynb
│   ├── evaluation.ipynb
│   └── unsupervised_analysis.ipynb
│
├── src/
│   └── preprocessing.py
│
├── models/
│
├── app/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# GitHub Feature Branch Workflow

Development tasks will use a Feature Branch Workflow instead of working directly on `main`.

The workflow is:

```text
main
 ↓
Create Feature Branch
 ↓
Complete the Task
 ↓
Test the Code
 ↓
git add
 ↓
git commit
 ↓
git push
 ↓
Open Pull Request
 ↓
Mentor Review
 ↓
Merge into main
```

Example:

```bash
git checkout -b feature/eda
```

After completing the work:

```bash
git add .
git commit -m "Complete exploratory data analysis"
git push origin feature/eda
```

A Pull Request is then opened for review before merging the changes into `main`.

---

# 📊 Week 5 Summary

During these five days, several Unsupervised Learning techniques were explored and applied to the Cardiac Patients dataset.

### Day 1

```text
Unsupervised Learning
        ↓
K-Means
        ↓
Elbow Method
        ↓
Silhouette Score
        ↓
Cluster Interpretation
```

### Day 2

```text
K-Means Limitations
        ↓
DBSCAN
        ↓
Noise Detection
        ↓
Hierarchical Clustering
        ↓
Dendrogram
        ↓
Clustering Method Comparison
```

### Day 3

```text
High-Dimensional Data
        ↓
StandardScaler
        ↓
PCA
        ↓
Explained Variance
        ↓
17 Components ≈ 95%
        ↓
2D PCA Visualization
```

### Day 4

```text
High-Dimensional Data
        ↓
t-SNE Visualization
        ↓
PCA vs. t-SNE
        ↓
Anomaly Detection
        ↓
Isolation Forest
        ↓
450 Flagged Anomalies
        ↓
Anomaly Interpretation
```

### Day 5

```text
Capstone Selection
        ↓
Problem Statement
        ↓
Definition of Done
        ↓
Sprint 1 Backlog
        ↓
Acceptance Criteria
        ↓
Sprint Goal
        ↓
GitHub Workflow
```

---

# 🧠 Key Concepts Learned

## Clustering

Clustering discovers groups of similar observations without predefined target labels.

The clustering algorithms explored were:

- K-Means
- DBSCAN
- Hierarchical Clustering

---

## Dimensionality Reduction

Dimensionality Reduction represents high-dimensional data using fewer dimensions.

PCA reduced the Cardiac Patients dataset from:

```text
22 Dimensions
      ↓
17 Principal Components
      ↓
≈ 94.98% Variance Preserved
```

---

## Visualization

PCA and t-SNE were both used to visualize high-dimensional data.

```text
PCA
→ Global Variance

t-SNE
→ Local Neighborhoods
```

---

## Anomaly Detection

Isolation Forest was used to identify unusual observations.

Using:

```text
contamination = 0.05
```

the results were:

```text
8550 Normal Observations
450 Anomalies
```

---

## Project Planning

The Cardiac Patients project was structured as a Phase 3 Capstone Project.

The planned final workflow includes:

```text
EDA
↓
Preprocessing
↓
Supervised Learning
↓
Evaluation
↓
Unsupervised Learning
↓
Final Model
↓
Deployment
↓
Documentation
```

---

# 🏆 Final Conclusion

Week 5 expanded the Machine Learning workflow beyond prediction by introducing techniques for discovering hidden structures, reducing dimensionality, visualizing complex datasets, and identifying unusual observations.

K-Means was used to divide patients into groups, while DBSCAN and Hierarchical Clustering provided alternative approaches to discovering structure.

PCA demonstrated how high-dimensional data can be represented using fewer dimensions while preserving most of its variance. In this dataset, 17 Principal Components preserved approximately **94.98%** of the total variance.

t-SNE provided another perspective by focusing on local relationships between observations and producing a two-dimensional visualization of the patient data.

Isolation Forest identified **450 anomalous observations** when using a contamination value of 5%, demonstrating how unusual patterns can be detected without predefined anomaly labels.

Finally, the Cardiac Patients project was selected and structured as the Phase 3 Capstone Project, with a clear problem statement, Definition of Done, Sprint 1 backlog, acceptance criteria, sprint goal, and GitHub development workflow.

This week connected individual Machine Learning techniques into a broader end-to-end project workflow that will continue through the Phase 3 capstone.

---

# 🛠️ Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- StandardScaler
- K-Means
- Silhouette Score
- DBSCAN
- Hierarchical Clustering
- Dendrogram
- PCA
- t-SNE
- Isolation Forest
- Jupyter Notebook
- Git
- GitHub

---

# 🚀 Next Phase

The next stage is **Phase 3**, where the Cardiac Patients project will be developed across multiple sprints into a complete end-to-end Machine Learning application.

The planned direction is:

```text
Sprint 1
→ Understand the Data + Establish Baseline

Sprint 2
→ Improve Models + Evaluate Performance

Sprint 3
→ Advanced Analysis + Unsupervised Learning

Sprint 4
→ Final Model + Deployment + Documentation
```