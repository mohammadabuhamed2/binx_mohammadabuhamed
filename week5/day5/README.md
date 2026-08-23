# Phase 3 — Capstone Project Planning

## Step 1 — Project Selection

### Selected Project

**Cardiac Disease Risk Prediction and Patient Data Analysis**

The project will focus on building an end-to-end Machine Learning system for analyzing cardiac patient data.

The project will use patient health and lifestyle features to predict heart disease risk using Supervised Learning techniques.

The project will also include Unsupervised Learning analysis to explore hidden patterns, patient groups, dimensional structure, and unusual observations in the dataset.

The complete project workflow will include:

- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
- Supervised Learning
- Model Evaluation
- Unsupervised Learning Analysis
- Project Documentation

## Step 2 — Problem Statement & Definition of Done

### Problem Statement

Heart disease risk can be associated with multiple health and lifestyle factors, making patient data complex to analyze manually.

The goal of this project is to build an end-to-end Machine Learning system that analyzes cardiac patient data and predicts whether a patient is at risk of heart disease.

The project will also use Unsupervised Learning techniques to explore hidden patterns, patient groups, dimensional structure, and unusual observations within the dataset.

The final system should provide reliable predictions, clear model evaluation, useful data analysis, and an accessible deployed application.

### Definition of Done

The project will be considered complete when the following requirements are satisfied:

- A clean and documented Jupyter Notebook covers the complete Machine Learning pipeline from EDA to preprocessing, modeling, and evaluation.
- The dataset is properly cleaned and preprocessed.
- Supervised Learning models are trained and evaluated using appropriate metrics.
- A baseline model is established and compared with improved models.
- Unsupervised Learning analysis is completed and documented.
- The final trained model has clearly reported evaluation metrics.
- The GitHub repository contains a clear project structure and README.
- A `requirements.txt` file contains the required project dependencies.
- The trained model artifacts are included or properly managed.
- A short technical write-up explains the problem, methodology, results, and conclusions.

## Step 3 — Sprint 1 Backlog & Acceptance Criteria

### Sprint 1 Backlog

The first sprint will focus on understanding the cardiac patient dataset, preparing the data for Machine Learning, and establishing a baseline model that can be improved in later sprints.

---

### Task 1 — Dataset Selection and Understanding

**Description:**

Select the cardiac patient dataset and understand its structure, features, target variable, and overall quality.

**Estimated Effort:** 2 hours

**Acceptance Criteria:**

- The dataset is selected and loaded successfully.
- The dataset shape and column names are inspected.
- Feature meanings are understood and documented.
- The target variable is clearly identified.
- Missing values and data types are inspected.
- The work is documented in Markdown.
- The notebook runs without errors.
- The work is committed to the correct GitHub feature branch with a clear commit message.

---

### Task 2 — Exploratory Data Analysis (EDA)

**Description:**

Explore the dataset to understand feature distributions, relationships, possible outliers, and the target distribution.

**Estimated Effort:** 3 hours

**Acceptance Criteria:**

- Numerical features are summarized using descriptive statistics.
- Important feature distributions are visualized.
- The target variable distribution is analyzed.
- Relationships between important features are explored.
- Potential outliers and unusual values are inspected.
- Important findings are documented in Markdown.
- All notebook cells run without errors.
- The work is committed to the correct GitHub feature branch with a clear commit message.

---

### Task 3 — Data Preprocessing

**Description:**

Prepare the dataset for Machine Learning by cleaning and transforming the required features.

**Estimated Effort:** 2 hours

**Acceptance Criteria:**

- Missing values are handled appropriately.
- Numerical and categorical features are identified.
- Required categorical features are encoded.
- Required numerical features are scaled when appropriate.
- The target is separated from the input features.
- Data leakage is avoided during preprocessing.
- The preprocessing steps are documented in Markdown.
- The notebook runs without errors.
- The work is committed to the correct GitHub feature branch.

---

### Task 4 — Baseline Model

**Description:**

Build an initial supervised Machine Learning model to establish a performance baseline for heart disease prediction.

**Estimated Effort:** 2 hours

**Acceptance Criteria:**

- The dataset is split appropriately for training and evaluation.
- A baseline classification model is trained successfully.
- Predictions are generated on unseen data.
- Appropriate classification metrics are calculated and reported.
- The baseline results are documented clearly.
- The baseline metrics are saved for comparison with future models.
- The notebook runs without errors.
- The work is committed to the correct GitHub feature branch with a clear commit message.
- A pull request is opened for review before merging.
## Step 4 — Sprint 1 Goal

### Sprint Goal

The goal of Sprint 1 is to understand and prepare the cardiac patient dataset and establish a baseline Machine Learning model for heart disease prediction.

By the end of the sprint, the dataset should be explored, cleaned, and preprocessed, and a baseline classification model should be trained and evaluated using appropriate metrics.

The baseline results will serve as a reference point for evaluating and improving the models developed in later sprints.

### Mentor Sign-Off

Sprint 1 should begin after the project scope, backlog, acceptance criteria, and sprint goal have been reviewed and approved by the mentor.

**Status:** Approved

## Step 5 — GitHub Repository & Feature Branch Workflow

The project repository will follow a clear structure separating data, notebooks, reusable source code, trained models, and deployment files.

Development tasks will be completed using a feature-branch workflow instead of working directly on the main branch.

For each major task:

1. Create a dedicated feature branch.
2. Complete and test the required work.
3. Commit the changes with a clear commit message.
4. Push the feature branch to GitHub.
5. Open a Pull Request for review.
6. Merge the branch into `main` after approval.

This workflow keeps the main branch stable and ensures that project changes are reviewed before being merged.