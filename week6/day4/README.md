# Day 4 — Building & Training a Network in Keras

## 📌 Overview

In this lesson, we learned how to build, train, evaluate, and improve a Neural Network using TensorFlow and Keras.

We used the Keras Sequential API to build a binary classification model for the Cardiac Disease Prediction project.

We also learned how to:

- Build Dense layers.
- Compile a Neural Network.
- Train the model using training and validation data.
- Read the training history.
- Detect overfitting using Loss, Accuracy, and Recall curves.
- Use Batch Normalization and Dropout.
- Evaluate the final models on the test set.
- Compare the Neural Network with the Day 1 Random Forest baseline.

---

# 🎯 Learning Objectives

By the end of this lesson, we should be able to:

- Build a Neural Network using the Keras Sequential API.
- Use Dense layers with appropriate activation functions.
- Compile a model using an optimizer, loss function, and metrics.
- Train a Neural Network using `fit()`.
- Evaluate a trained model using `evaluate()`.
- Read and visualize the training history.
- Detect overfitting from training and validation curves.
- Apply Batch Normalization and Dropout.
- Compare a regularized Neural Network with the original model.
- Compare the Neural Network with the classical Machine Learning baseline.

---

# 1. TensorFlow and Keras

TensorFlow is a framework used to build and train Neural Networks.

Keras provides a high-level API inside TensorFlow that makes building Neural Networks easier.

Instead of manually implementing:

```text
Forward Propagation
↓
Loss Calculation
↓
Backpropagation
↓
Gradient Calculation
↓
Weight Updates
```

TensorFlow and Keras handle these calculations automatically.

We can import TensorFlow using:

```python
import tensorflow as tf
```

---

# 2. Prepare the Cardiac Patients Dataset

The processed Cardiac Patients dataset was loaded before building the Neural Network.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv(
    r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\Cardiac Patients\data\dataprocessed_heart_data.csv"
)
```

---

## 2.1 Separate Features and Target

The target variable is:

```text
has_heart_disease
```

where:

```text
0 → No Heart Disease
1 → Heart Disease
```

The features and target were separated using:

```python
X = df.drop("has_heart_disease", axis=1)
y = df["has_heart_disease"]
```

---

## 2.2 Split the Data

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test set is kept separate and is only used for final evaluation.

---

# 3. Identify Numerical and Categorical Features

The dataset contains both numerical and categorical features.

```python
catcols = X.select_dtypes(include="object").columns
numcols = X.select_dtypes(exclude="object").columns
```

Categorical features require encoding before they can be used by the Neural Network.

Numerical features are scaled to place them on a similar numerical scale.

---

# 4. Preprocess the Features

A `ColumnTransformer` was used to apply different preprocessing techniques to numerical and categorical columns.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocceser = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numcols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), catcols)
    ]
)
```

The numerical features are transformed using `StandardScaler`.

The categorical features are transformed using `OneHotEncoder`.

---

## 4.1 Transform the Training Data

The preprocessor is fitted only on the training data.

```python
X_train_pre = preprocceser.fit_transform(X_train)
```

`fit_transform()` performs two operations:

```text
fit
→ Learn the scaling values and categories from the training data

transform
→ Apply the learned transformations
```

---

## 4.2 Transform the Test Data

The test data is transformed using the already fitted preprocessor.

```python
X_test_pre = preprocceser.transform(X_test)
```

Only `transform()` is used on the test data to avoid Data Leakage.

---

# 5. Step 1 — Build a Keras Sequential Neural Network

The number of input features is determined after preprocessing.

```python
n_features = X_train_pre.shape[1]
```

This is important because One-Hot Encoding may increase the number of input columns.

---

## 5.1 Initial Neural Network Architecture

The initial Neural Network was built using the Keras Sequential API.

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The architecture is:

```text
Processed Input Features
↓
64 Neurons + ReLU
↓
32 Neurons + ReLU
↓
1 Output Neuron + Sigmoid
↓
Heart Disease Probability
```

The two hidden layers with 64 and 32 neurons are initial design choices and are not assumed to be the optimal architecture.

---

# 6. Activation Functions

The hidden layers use:

```text
ReLU
```

because ReLU introduces non-linearity and is commonly used in hidden layers.

The output layer uses:

```text
Sigmoid
```

because this project is a Binary Classification problem.

Sigmoid produces a value between 0 and 1 that can be interpreted as the probability of the positive class.

---

# 7. Step 2 — Compile and Train the Neural Network

Before training, the model must be compiled.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall")
    ]
)
```

The model uses:

```text
Optimizer → Adam
Loss → Binary Cross-Entropy
Metrics → Accuracy and Recall
```

Binary Cross-Entropy is used because the project is a Binary Classification task.

Recall was also monitored because detecting positive Heart Disease cases is important for this project.

---

## 7.1 Train the Neural Network

The model was trained using:

```python
history = model.fit(
    X_train_pre,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)
```

The settings mean:

```text
validation_split = 0.2
→ 20% of the training data is used for validation

epochs = 30
→ The model passes through the training data 30 times

batch_size = 32
→ 32 samples are processed before each weight update
```

The `history` object stores the training results from every Epoch.

---

# 8. Training History

The `history.history` property contains values recorded during training.

For example:

```python
history.history["loss"]
```

contains the Training Loss for every Epoch.

Other available values include:

```text
loss
val_loss
accuracy
val_accuracy
recall
val_recall
```

These values can be plotted to understand how the Neural Network behaves during training.

---

# 9. Step 3 — Training and Validation Loss

The Training Loss and Validation Loss were plotted.

```python
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()
```

---

## 9.1 Loss Curve Interpretation

The Training Loss continued to decrease throughout the training process.

However, the Validation Loss decreased only during the first few Epochs and then started to increase.

This indicates Overfitting.

The Neural Network continued improving on the training data, while its performance on unseen validation data became worse.

The best validation performance occurred during the early Epochs before the Validation Loss started increasing.

---

# 10. Training and Validation Accuracy

The Training Accuracy and Validation Accuracy were also compared.

```python
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()
```

---

## 10.1 Accuracy Curve Interpretation

The Training Accuracy continued to increase and reached approximately 95%.

The Validation Accuracy improved during the early Epochs but then remained almost stable and slightly decreased.

The increasing gap between Training Accuracy and Validation Accuracy supports the conclusion that the original Neural Network was overfitting.

The model was learning the training data increasingly well without improving its ability to generalize to unseen validation data.

---

# 11. Training and Validation Recall

Recall was also monitored during training.

```python
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["recall"],
    label="Training Recall"
)

plt.plot(
    history.history["val_recall"],
    label="Validation Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Training vs Validation Recall")
plt.legend()

plt.show()
```

---

## 11.1 Recall Curve Interpretation

The Training Recall continued to increase throughout training and reached approximately 0.89.

The Validation Recall improved slightly during the early Epochs but then remained relatively stable around 0.75–0.78.

The growing gap between Training Recall and Validation Recall was another indication of Overfitting.

The model became better at detecting positive cases in the training data, but this improvement did not transfer equally to unseen validation data.

---

# 12. Step 4 — Add Batch Normalization and Dropout

The original Neural Network showed clear signs of Overfitting.

A second Neural Network was created using Batch Normalization and Dropout.

```python
model_regularized = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

---

# 13. Batch Normalization

Batch Normalization normalizes values moving between layers during training.

Its goal is to make training more stable and efficient.

It can help the Neural Network train more consistently.

---

# 14. Dropout

Dropout randomly disables a percentage of neurons during each training step.

In this experiment:

```python
tf.keras.layers.Dropout(0.3)
```

means that approximately 30% of the neurons are temporarily disabled during each training step.

The neurons are not permanently removed.

Dropout prevents the model from relying too heavily on specific neurons and can reduce Overfitting.

---

# 15. Compile the Regularized Model

The regularized model was compiled using the same optimizer, loss function, and metrics as the original model.

```python
model_regularized.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall")
    ]
)
```

Using similar training settings makes the comparison between the two models more meaningful.

---

# 16. Train the Regularized Model

The regularized model was trained using:

```python
history_regularized = model_regularized.fit(
    X_train_pre,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)
```

The results were stored in:

```text
history_regularized
```

while the original model results remained stored in:

```text
history
```

---

# 17. Compare Original and Regularized Validation Loss

The Validation Loss curves of the two models were compared.

```python
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["val_loss"],
    label="Original Validation Loss"
)

plt.plot(
    history_regularized.history["val_loss"],
    label="Regularized Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Original vs Regularized Validation Loss")
plt.legend()

plt.show()
```

---

## 17.1 Regularization Comparison

The regularized model produced a lower and more stable Validation Loss than the original model.

The original model started to overfit after the first few Epochs because its Validation Loss continued increasing.

After adding Dropout and Batch Normalization, the Validation Loss decreased further and remained relatively stable during most of the training process.

This indicates that the regularization techniques reduced Overfitting and improved generalization.

---

# 18. Compare Original and Regularized Validation Recall

The Validation Recall of the two models was also compared.

```python
plt.figure(figsize=(8, 6))

plt.plot(
    history_regularized.history["val_recall"],
    label="Regularized Validation Recall"
)

plt.plot(
    history.history["val_recall"],
    label="Original Validation Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Original vs Regularized Validation Recall")
plt.legend()

plt.show()
```

---

## 18.1 Validation Recall Comparison

The original model achieved slightly higher Validation Recall than the regularized model.

After adding Dropout and Batch Normalization, Validation Recall became slightly lower even though the Validation Loss became more stable.

This shows that regularization reduced Overfitting but slightly reduced the model's ability to detect positive cases in the validation data.

Therefore, both generalization and Recall should be considered when selecting the final model.

---

# 19. Step 5 — Evaluate on the Test Set

The original Neural Network was evaluated on the unseen test set.

```python
original_loss, original_accuracy, original_recall = model.evaluate(
    X_test_pre,
    y_test
)
```

The regularized model was also evaluated.

```python
regularized_loss, regularized_accuracy, regularized_recall = model_regularized.evaluate(
    X_test_pre,
    y_test
)
```

---

# 20. Test Set Results

The original Neural Network achieved:

```text
Loss     = 0.3300
Accuracy = 0.8811
Recall   = 0.7896
```

The regularized Neural Network achieved:

```text
Loss     = 0.2541
Accuracy = 0.8933
Recall   = 0.7932
```

The regularized Neural Network performed better than the original model.

It achieved:

- Lower Loss
- Higher Accuracy
- Slightly higher Recall

This confirms that Batch Normalization and Dropout improved the overall Neural Network performance.

---

# 21. Compare with the Day 1 Baseline

The Day 1 reference model was the previously trained Random Forest.

The results were:

| Model | Accuracy | Recall |
|---|---:|---:|
| Random Forest Baseline | 0.88 | **0.85** |
| Original Neural Network | 0.881 | 0.790 |
| Regularized Neural Network | **0.893** | 0.793 |

The regularized Neural Network achieved the highest Accuracy.

However, the Random Forest achieved the highest Recall.

Since Recall is especially important for detecting positive Heart Disease cases in this project, the Random Forest remains stronger according to the selected priority metric.

The Neural Network still showed that regularization can improve generalization and overall performance compared with the original architecture.

---

# 22. Complete Keras Workflow

The complete workflow used in this lesson was:

```text
Load Dataset
↓
Separate X and y
↓
Train-Test Split
↓
Identify Numerical and Categorical Features
↓
StandardScaler + OneHotEncoder
↓
Build Sequential Model
↓
Compile
↓
Train with fit()
↓
Store Training History
↓
Plot Loss, Accuracy, and Recall
↓
Detect Overfitting
↓
Add Batch Normalization + Dropout
↓
Retrain
↓
Compare Validation Curves
↓
Evaluate on Test Set
↓
Compare with Random Forest Baseline
```

---

# ✅ Key Takeaways

- TensorFlow and Keras handle Forward Propagation, Backpropagation, and optimization automatically.
- The Sequential API builds Neural Networks by stacking layers in order.
- Dense layers connect every input from the previous layer to every neuron.
- ReLU is commonly used in hidden layers.
- Sigmoid is appropriate for Binary Classification output.
- Binary Cross-Entropy is appropriate for the Cardiac Disease Prediction task.
- `compile()` defines how the model will be trained.
- `fit()` performs the training process.
- `evaluate()` measures performance on unseen test data.
- The `history` object stores training and validation results for every Epoch.
- Training and validation curves can reveal Overfitting.
- The original Neural Network showed clear signs of Overfitting.
- Batch Normalization and Dropout reduced Overfitting.
- The regularized Neural Network achieved better test performance than the original Neural Network.
- The regularized Neural Network achieved higher Accuracy than the Random Forest baseline.
- The Random Forest still achieved higher Recall and remained stronger according to the project's priority metric.