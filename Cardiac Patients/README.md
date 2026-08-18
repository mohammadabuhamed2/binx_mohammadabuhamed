# Cardiac Patient Heart Disease Prediction

## 📌 Project Overview

This project focuses on analyzing cardiac patient data and building Machine Learning classification models to predict whether a patient has heart disease.

The project includes:

- Exploratory Data Analysis (EDA)
- Data Preparation
- Data Preprocessing
- Machine Learning Pipelines
- Multiple Classification Models
- Hyperparameter Tuning using GridSearchCV
- Cross-Validation
- Model Evaluation
- ROC-AUC Analysis
- Final Model Selection

The main goal is to build and compare multiple classification models and select a model that can effectively identify patients with heart disease.

---

# 📊 Dataset

The dataset contains **9,000 patient records** with demographic, clinical, lifestyle, and wearable-related features.

The target variable is:

`has_heart_disease`

Where:

- `0` → No Heart Disease
- `1` → Heart Disease

The target distribution is approximately:

- **69.7%** No Heart Disease
- **30.3%** Heart Disease

This indicates a moderate class imbalance.

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Checking dataset shape and data types
- Checking missing values
- Checking duplicated rows
- Analyzing the target distribution
- Analyzing numerical feature distributions
- Analyzing categorical features
- Detecting potential outliers
- Examining relationships between important features
- Correlation analysis

The `patient_id` column was removed because it represents a unique identifier and does not provide useful predictive information.

---

# 🧹 Data Preparation

The dataset contains different types of features:

### Numerical Features

Examples include:

- Age
- Blood Pressure
- Cholesterol
- HDL
- LDL
- Triglycerides
- Fasting Blood Sugar
- HbA1c
- BMI
- Heart Rate
- ST Depression
- Exercise Minutes
- Sleep Hours
- Stress Score
- Daily Steps
- Diet Quality Score

### Categorical Features

Categorical features include:

- `sex`
- `chest_pain_type`
- `smoker_status`

### Boolean Features

Boolean features were converted to numerical values:

- `True` → `1`
- `False` → `0`

These include features such as:

- `exercise_induced_angina`
- `family_history`
- `wearable_owner`

---

# ⚙️ Preprocessing Pipeline

A Scikit-learn `Pipeline` and `ColumnTransformer` were used to combine preprocessing and model training.

The preprocessing pipeline performs:

### Numerical Features

`StandardScaler` is applied to numerical features to standardize their scales.

### Categorical Features

`OneHotEncoder` is applied to categorical features to convert categorical values into numerical representations.

Using a Pipeline ensures that preprocessing is learned from the training data and then consistently applied to unseen data.

---

# ✂️ Train/Test Split

The dataset was divided into:

- Training Set: **80%**
- Test Set: **20%**

The Test Set contains **1,800 patients** and was kept unseen during model training and hyperparameter tuning.

---

# 🤖 Machine Learning Models

Five classification algorithms were trained and compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbors (KNN)

Each model was combined with the preprocessing steps inside a Scikit-learn Pipeline.

---

# 🔧 Hyperparameter Tuning

`GridSearchCV` was used to search for suitable hyperparameter combinations for each model.

Cross-Validation was used during the search to obtain a more reliable estimate of model performance.

The primary scoring metric during hyperparameter tuning was:

`F1-score`

F1-score was selected because the dataset contains a moderate class imbalance and the metric balances Precision and Recall.

---

# 📈 Cross-Validation Results

The best Cross-Validation F1-scores were:

| Model | Best CV F1-score |
|---|---:|
| Logistic Regression | **0.8251** |
| SVM | **0.8233** |
| Random Forest | **0.8038** |
| Decision Tree | **0.7330** |
| KNN | **0.7106** |

Logistic Regression achieved the highest Cross-Validation F1-score, followed closely by SVM.

---

# 🧪 Model Evaluation

The tuned models were evaluated on the unseen Test Set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Special attention was given to the performance of the positive class:

`Heart Disease = 1`

---

# 📊 Final Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | False Negatives |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.91** | 0.87 | 0.83 | **0.85** | **0.9604** | 97 |
| SVM | **0.91** | **0.88** | 0.81 | **0.85** | 0.9599 | 103 |
| Decision Tree | 0.86 | 0.84 | 0.68 | 0.75 | 0.9110 | 180 |
| Random Forest | 0.88 | 0.79 | **0.85** | 0.82 | 0.9464 | **84** |
| KNN | 0.85 | 0.87 | 0.62 | 0.72 | 0.9214 | 214 |

---

# 🌲 Final Model — Random Forest

Random Forest was selected as the final model.

Although Logistic Regression and SVM achieved higher overall Accuracy, F1-score, and ROC-AUC, Random Forest achieved the:

- **Highest Recall: 0.85**
- **Lowest number of False Negatives: 84**

This was considered important for the objective of this project.

A **False Negative** represents a patient who actually has heart disease but is predicted as not having heart disease.

A **False Positive** represents a patient who does not have heart disease but is predicted as having heart disease.

For a heart disease screening-oriented prediction task, missing a patient who actually has heart disease can be more concerning than flagging a patient who does not have the disease for further evaluation.

Therefore, greater importance was given to **Recall and reducing False Negatives** when selecting the final model.

---

# 📉 Random Forest Test Results

The selected Random Forest model achieved:

- **Accuracy:** 0.88
- **Precision:** 0.79
- **Recall:** 0.85
- **F1-score:** 0.82
- **ROC-AUC:** 0.9464

Confusion Matrix results:

- **True Negatives:** 1119
- **False Positives:** 125
- **False Negatives:** 84
- **True Positives:** 472

The model correctly detected **472 out of 556** patients with heart disease.

---

# 📈 ROC-AUC Results

The ROC-AUC scores were:

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | **0.9604** |
| SVM | 0.9599 |
| Random Forest | 0.9464 |
| KNN | 0.9214 |
| Decision Tree | 0.9110 |

Logistic Regression and SVM achieved the highest ROC-AUC scores.

Random Forest also demonstrated strong discrimination performance with a ROC-AUC of approximately **0.946**.

---

# 📁 Project Structure

```text
Cardiac Patients/
│
├── data/
│   ├── original_data.csv
│   ├── processed_heart_data.csv
│   └── test_data.csv
│
├── notebooks/
│   ├── 01_eda_preparation.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
│
├── models/
│   ├── logistic_regression_pipeline.pkl
│   ├── svm_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   └── knn_pipeline.pkl
│
├── outputs/
│   ├── model_comparison.csv
│   ├── roc_curve_comparison.png
│   └── random_forest_confusion_matrix.png
│
├── README.md
│
└── requirements.txt
```

---

# 📓 Notebooks

### `01_eda_preparation.ipynb`

Contains:

- Dataset exploration
- Data quality checks
- Feature analysis
- Target distribution analysis
- Data preparation

### `02_modeling.ipynb`

Contains:

- Train/Test Split
- Preprocessing Pipeline
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- KNN
- GridSearchCV
- Cross-Validation
- Hyperparameter Tuning
- Saving trained pipelines

### `03_evaluation.ipynb`

Contains:

- Loading the unseen Test Set
- Loading trained pipelines
- Predictions
- Confusion Matrices
- Classification Reports
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- ROC Curve comparison
- Final model comparison
- Final model selection

---

# 💾 Saved Models

The trained Scikit-learn Pipelines are stored inside the `models/` directory.

Each saved Pipeline contains both:

- Data preprocessing
- Trained classification model

This allows raw patient features to be passed directly to the Pipeline without manually repeating Scaling or One-Hot Encoding.

---

# ⚠️ Limitations

Several limitations should be considered:

- The dataset has a moderate class imbalance.
- Model performance depends on the quality and representativeness of the available dataset.
- Different classification thresholds can change the balance between False Positives and False Negatives.
- The selected model still produces False Negatives and False Positives.
- High predictive performance on this dataset does not automatically imply equivalent performance on patients from different populations or clinical environments.
- This project is a Machine Learning classification project and should not be treated as a real-world medical diagnostic system without appropriate clinical validation.

---

# 🎯 Conclusion

This project developed and compared five Machine Learning classification models for predicting heart disease.

Logistic Regression and SVM achieved the strongest overall Accuracy, F1-score, and ROC-AUC performance.

However, Random Forest achieved the highest Recall for patients with heart disease and produced the lowest number of False Negatives.

Because reducing missed positive cases was prioritized in this project, **Random Forest was selected as the final model**.

The project demonstrates the importance of evaluating Machine Learning models using multiple metrics instead of selecting a model based only on Accuracy.