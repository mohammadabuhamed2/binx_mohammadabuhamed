# Week 6 — Neural Networks & Sprint 1

## Cardiac Disease Risk Prediction and Patient Data Analysis

## Overview

During this week, the project moved from a classical Machine Learning baseline to building, training, regularizing, tuning, and evaluating a Neural Network.

The main objective was to understand how Neural Networks work internally and then apply these concepts to the Cardiac Disease Risk Prediction project.

The week covered:

- Neural Network architecture
- Neurons, weights, biases, and layers
- Activation functions
- Forward propagation
- Loss functions
- Backpropagation
- Gradient Descent
- Optimizers
- Epochs and batches
- Building Neural Networks with TensorFlow/Keras
- Batch Normalization
- Dropout
- Hyperparameter tuning
- EarlyStopping
- ModelCheckpoint
- Final model evaluation
- Sprint Review and Retrospective

---

# Day 1 — Sprint 1 Planning & Neural Network Architecture

## Learning Objectives

By the end of Day 1, we were able to:

- Define the Sprint 1 goal.
- Establish a classical Machine Learning baseline.
- Understand the basic structure of a Neural Network.
- Explain how a single neuron works.
- Understand Input, Hidden, and Output layers.
- Understand the role of weights and biases.

---

## Sprint 1 Goal

The main Sprint 1 goal was to build a Neural Network for the Cardiac Disease Risk Prediction project and compare it with the best classical Machine Learning model.

A previously trained Random Forest model was used as the baseline.

### Random Forest Baseline

| Metric | Result |
|---|---:|
| Accuracy | 0.88 |
| Recall | 0.85 |
| F1-score | 0.82 |
| ROC-AUC | 0.946 |

Because the project focuses on identifying positive Heart Disease cases, Recall was considered an important metric.

---

## Why Neural Networks?

Neural Networks are powerful models that can learn complex and non-linear relationships between features.

Instead of manually defining relationships between variables, a Neural Network learns useful patterns through its weights during training.

---

## A Single Neuron

A neuron receives input features, multiplies them by weights, adds a bias, and then applies an activation function.

The basic calculation is:

\[
z = XW + b
\]

Where:

- `X` = input features
- `W` = weights
- `b` = bias
- `z` = weighted sum

After calculating `z`, an Activation Function is applied.

\[
a = f(z)
\]

---

## Neural Network Layers

A Neural Network usually contains three main types of layers.

### Input Layer

The Input Layer receives the features from the dataset.

Examples from the cardiac dataset include:

- Age
- Blood Pressure
- Cholesterol
- BMI
- Heart Rate
- Exercise
- Smoking Status

---

### Hidden Layers

Hidden Layers learn intermediate patterns from the input features.

Each hidden layer contains neurons that transform the information received from the previous layer.

Adding more hidden layers can allow a Neural Network to learn more complex relationships.

---

### Output Layer

The Output Layer produces the final prediction.

Because this project is a Binary Classification problem:

```text
0 = No Heart Disease
1 = Heart Disease
```

the Output Layer contains one neuron.

---

## Weights and Biases

Weights determine how important each input is to a neuron.

Bias allows the neuron to shift its decision boundary.

Both weights and biases are learned automatically during Neural Network training.

---

# Day 2 — Activations, Forward Propagation & Loss

## Learning Objectives

By the end of Day 2, we were able to:

- Understand why Activation Functions are necessary.
- Understand ReLU, Sigmoid, and Tanh.
- Choose the correct activation for Hidden and Output layers.
- Understand Forward Propagation.
- Understand Loss Functions.
- Select the correct loss function for Binary Classification.

---

# Activation Functions

Without Activation Functions, multiple Neural Network layers would behave like one large linear model.

Activation Functions introduce non-linearity, allowing the network to learn complex patterns.

---

## ReLU

ReLU stands for Rectified Linear Unit.

\[
ReLU(x) = \max(0,x)
\]

Behavior:

- Negative values become `0`.
- Positive values remain unchanged.

ReLU is commonly used in Hidden Layers.

For this project:

```python
activation="relu"
```

was used in the Hidden Layers.

---

## Sigmoid

The Sigmoid function converts any input into a value between `0` and `1`.

\[
Sigmoid(x)=\frac{1}{1+e^{-x}}
\]

It is useful for Binary Classification because the output can be interpreted as a probability.

For this project:

```python
activation="sigmoid"
```

was used in the Output Layer.

---

## Tanh

Tanh produces values between `-1` and `1`.

\[
tanh(x)=
\frac{e^x-e^{-x}}
{e^x+e^{-x}}
\]

Tanh is another non-linear activation function, although ReLU is more commonly used in modern Hidden Layers.

---

# Forward Propagation

Forward Propagation is the process of moving information from the Input Layer through the Hidden Layers to the Output Layer.

Example:

\[
z_1 = XW_1 + b_1
\]

\[
a_1 = ReLU(z_1)
\]

Then:

\[
z_2 = a_1W_2 + b_2
\]

Finally:

\[
\hat{y} = Sigmoid(z_2)
\]

The final value represents the predicted probability.

---

## Simple Forward Propagation Example

A small example was created manually using:

```python
X_sample = [2.0, 1.0]
```

After applying the weights, biases, ReLU, and Sigmoid, the model produced approximately:

```text
Prediction Probability ≈ 0.74
```

This example was only used to understand Forward Propagation conceptually.

---

# Loss Functions

The Loss Function measures how wrong the model prediction is.

Training attempts to minimize this value.

---

## Binary Cross-Entropy

For Binary Classification, Binary Cross-Entropy is commonly used.

\[
L =
-[y\log(\hat{y}) +
(1-y)\log(1-\hat{y})]
\]

For the Cardiac Disease project:

```python
loss="binary_crossentropy"
```

was selected because the target contains two classes.

---

## Final Activation and Loss Selection

For this project:

| Layer | Activation |
|---|---|
| Hidden Layers | ReLU |
| Output Layer | Sigmoid |

And:

```text
Loss Function = Binary Cross-Entropy
```

---

# Day 3 — Backpropagation, Gradient Descent & Optimizers

## Learning Objectives

By the end of Day 3, we were able to:

- Understand the Neural Network training loop.
- Understand Gradient Descent.
- Understand the Learning Rate.
- Understand Backpropagation conceptually.
- Understand the Chain Rule.
- Understand Optimizers.
- Understand Epochs and Batches.

---

# Neural Network Training Loop

Training a Neural Network follows four main steps:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Weights
```

This cycle repeats many times until the model improves.

---

# Gradient Descent

Gradient Descent is the process used to update model weights in order to reduce the Loss.

The Gradient tells us how changing each weight affects the Loss.

Weights are updated in the opposite direction of the Gradient.

Conceptually:

\[
W_{new}
=
W_{old}
-
LearningRate \times Gradient
\]

---

# Learning Rate

The Learning Rate controls how large each weight update is.

A very small Learning Rate can make training slow.

A very large Learning Rate can make training unstable.

Different Learning Rates were tested using a small Neural Network.

The experiment used:

```text
0.00001
0.001
0.1
```

In the experiment:

- `0.00001` learned very slowly.
- `0.001` improved gradually.
- `0.1` learned much faster and remained stable in that specific experiment.

The experiment demonstrated how strongly Learning Rate can affect training speed.

---

# Backpropagation

Backpropagation determines how much each weight contributed to the final error.

The process works backward from the Loss toward earlier layers.

It calculates Gradients for the weights and biases.

These Gradients are then used by the Optimizer to update the parameters.

---

## Chain Rule

A Neural Network contains connected calculations.

For example:

```text
Input
 ↓
Weighted Sum
 ↓
Activation
 ↓
Next Layer
 ↓
Prediction
 ↓
Loss
```

Because the final Loss depends indirectly on earlier weights, the Chain Rule is used to calculate how changing an earlier weight affects the final Loss.

---

# Optimizers

Optimizers determine how the calculated Gradients are used to update the weights.

Common Optimizers include:

- SGD
- Adam

For this project, Adam was used.

```python
optimizer="adam"
```

Adam automatically adjusts the updates during training and is commonly used for Neural Networks.

---

# Epochs and Batches

## Epoch

One Epoch means the Neural Network has processed the entire Training Set once.

For example, with approximately:

```text
7200 Training Samples
```

one Epoch means all `7200` samples were used once.

---

## Batch

A Batch is a smaller group of Training Samples processed before updating the model weights.

For example:

```text
Batch Size = 32
```

With approximately `7200` samples:

\[
7200 / 32 \approx 225
\]

Therefore, the model performs approximately `225` weight updates during one Epoch.

---

# Day 4 — Building & Training a Neural Network in Keras

## Learning Objectives

By the end of Day 4, we were able to:

- Build a Neural Network using Keras Sequential API.
- Compile and train the model.
- Evaluate model performance.
- Read the training history.
- Detect Overfitting.
- Use Batch Normalization.
- Use Dropout.
- Compare a Neural Network with a classical Machine Learning baseline.

---

# Dataset Preparation

The target variable was:

```python
has_heart_disease
```

The data was separated into:

```python
X = df.drop("has_heart_disease", axis=1)
y = df["has_heart_disease"]
```

Then the data was divided into Training and Test sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Feature Preprocessing

Categorical and Numerical features were identified separately.

```python
catcols = X.select_dtypes(include="object").columns
numcols = X.select_dtypes(exclude="object").columns
```

Numerical features were standardized using:

```python
StandardScaler()
```

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing pipeline was:

```python
preprocceser = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numcols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), catcols)
    ]
)
```

Training data was fitted and transformed:

```python
X_train_pre = preprocceser.fit_transform(X_train)
```

Test data was only transformed:

```python
X_test_pre = preprocceser.transform(X_test)
```

---

# Original Neural Network

The initial Neural Network architecture was:

```text
Input
 ↓
Dense 64 — ReLU
 ↓
Dense 32 — ReLU
 ↓
Dense 1 — Sigmoid
```

The model was built using:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The model was compiled using:

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

---

# Training History

Keras stores training results inside:

```python
history.history
```

Important recorded metrics included:

```text
loss
val_loss
accuracy
val_accuracy
recall
val_recall
```

These values were used to create Training Curves.

---

# Detecting Overfitting

The original model showed signs of Overfitting.

Training Loss continued decreasing while Validation Loss began increasing.

Training Accuracy and Recall continued improving while Validation performance became more stable.

This created a growing gap between Training and Validation performance.

---

# Batch Normalization

Batch Normalization was added to help stabilize training.

Example:

```python
tf.keras.layers.BatchNormalization()
```

It normalizes intermediate layer outputs during training.

---

# Dropout

Dropout randomly disables some neurons during training.

Example:

```python
tf.keras.layers.Dropout(0.3)
```

This prevents the network from relying too heavily on specific neurons and can reduce Overfitting.

---

# Regularized Neural Network

The improved architecture became:

```python
regularized_model = tf.keras.Sequential([
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

# Day 4 Model Results

| Model | Test Loss | Accuracy | Recall |
|---|---:|---:|---:|
| Original Neural Network | 0.3300 | 0.8811 | 0.7896 |
| Regularized Neural Network | 0.2541 | 0.8933 | 0.7932 |

The Regularized Neural Network achieved:

- Lower Test Loss
- Higher Accuracy
- Slightly higher Recall

This showed that Batch Normalization and Dropout helped improve Generalization.

---

# Comparison with Random Forest Baseline

| Model | Accuracy | Recall |
|---|---:|---:|
| Random Forest | 0.880 | **0.850** |
| Regularized Neural Network | **0.893** | 0.793 |

The Neural Network achieved better Accuracy.

However, Random Forest achieved better Recall.

Because Recall is important for identifying positive Heart Disease cases, Random Forest remained the stronger baseline according to the project priority.

---

# Day 5 — Tuning, Evaluation & Sprint Review

## Learning Objectives

By the end of Day 5, we were able to:

- Tune a Neural Network systematically.
- Change one Hyperparameter at a time.
- Use EarlyStopping.
- Use ModelCheckpoint.
- Select the best Neural Network configuration.
- Evaluate the tuned model.
- Compare the final Neural Network with the baseline.
- Prepare Sprint Review evidence.
- Complete the Sprint Retrospective.

---

# Systematic Hyperparameter Tuning

The following Hyperparameters were tuned:

```text
Learning Rate
Network Size
Dropout Rate
Batch Size
```

Only one variable was changed at a time.

This makes it easier to understand how each Hyperparameter affects model performance.

---

# Learning Rate Tuning

| Learning Rate | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 0.0001 | **0.2587** | **0.7746** | 61 |
| 0.001 | 0.2596 | 0.7700 | 18 |
| 0.01 | 0.2700 | 0.7653 | 13 |

Selected:

```text
Learning Rate = 0.0001
```

---

# Network Size Tuning

| Network Size | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 16 → 8 | 0.2671 | 0.7559 | 146 |
| 32 → 16 | **0.2576** | **0.7793** | 103 |
| 64 → 32 | 0.2619 | 0.7676 | 70 |

Selected:

```text
Network Size = 32 → 16
```

---

# Dropout Rate Tuning

| Dropout Rate | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 0.1 | 0.2721 | 0.7582 | 72 |
| 0.3 | 0.2640 | 0.7606 | 87 |
| 0.5 | **0.2608** | **0.7770** | 111 |

Selected:

```text
Dropout Rate = 0.5
```

---

# Batch Size Tuning

| Batch Size | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 16 | 0.2650 | 0.7746 | 69 |
| 32 | **0.2617** | **0.7770** | 115 |
| 64 | 0.2635 | 0.7582 | 163 |

Selected:

```text
Batch Size = 32
```

---

# Final Selected Hyperparameters

| Hyperparameter | Selected Value |
|---|---:|
| Learning Rate | 0.0001 |
| Network Size | 32 → 16 |
| Dropout Rate | 0.5 |
| Batch Size | 32 |

---

# Final Tuned Neural Network

The final Neural Network architecture was:

```python
final_model = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The Adam Optimizer was configured with:

```python
learning_rate = 0.0001
```

The model used:

```python
loss="binary_crossentropy"
```

and monitored:

```text
Accuracy
Recall
```

---

# EarlyStopping

EarlyStopping automatically stops training when Validation Loss stops improving.

```python
final_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

The option:

```python
restore_best_weights=True
```

restores the model weights from the best Epoch.

---

# ModelCheckpoint

ModelCheckpoint was used to save the best version of the model.

```python
final_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "final_tuned_model.keras",
    monitor="val_loss",
    save_best_only=True
)
```

This prevents losing the best model if later Epochs perform worse.

---

# Final Model Evaluation

The final tuned Neural Network was evaluated using the Test Set.

| Metric | Result |
|---|---:|
| Test Loss | 0.2397 |
| Test Accuracy | 0.9122 |
| Test Recall | 0.8309 |

The tuned Neural Network achieved strong performance on unseen data.

---

# Final Model Comparison

| Model | Test Loss | Accuracy | Recall |
|---|---:|---:|---:|
| Random Forest Baseline | — | 0.880 | **0.850** |
| Original Neural Network | 0.3300 | 0.881 | 0.790 |
| Regularized Neural Network | 0.2541 | 0.893 | 0.793 |
| Final Tuned Neural Network | **0.2397** | **0.912** | 0.831 |

The Final Tuned Neural Network achieved the highest Accuracy.

The Random Forest achieved the highest Recall.

---

# Training Curves

The final Neural Network was analyzed using three Training Curves:

```text
Loss Curve
Accuracy Curve
Recall Curve
```

---

## Loss Curve Findings

Both Training Loss and Validation Loss decreased steadily.

Validation Loss remained stable without increasing significantly.

This indicates that the tuned model controlled Overfitting effectively.

Validation Loss was slightly lower than Training Loss for most Epochs.

This can happen because Dropout is enabled during training but disabled during validation.

Overall, the model showed stable learning and good Generalization.

---

## Accuracy Curve Findings

Both Training Accuracy and Validation Accuracy increased during training.

Validation Accuracy reached approximately:

```text
0.89 – 0.90
```

and remained stable.

The gap between Training and Validation Accuracy remained small.

This indicates that the model did not show severe Overfitting.

---

## Recall Curve Findings

Training Recall gradually improved during training.

Validation Recall initially fluctuated but became more stable later.

Toward the end of training, Training Recall and Validation Recall became close.

This indicates that the model generalized reasonably well when identifying positive Heart Disease cases.

---

# Sprint Review

During Sprint 1, the project successfully moved from a classical Machine Learning baseline to a complete Neural Network workflow.

The Neural Network was:

- Designed
- Built
- Trained
- Evaluated
- Regularized
- Tuned
- Compared with the baseline

The Final Tuned Neural Network achieved:

```text
Test Loss     = 0.2397
Test Accuracy = 0.9122
Test Recall   = 0.8309
```

Compared with the original Neural Network:

```text
Accuracy: 0.881 → 0.912
Recall:   0.790 → 0.831
Loss:     0.330 → 0.240
```

Therefore, systematic tuning significantly improved the Neural Network.

---

# Sprint Retrospective

## What Went Well

The Neural Network improved significantly after systematic tuning.

Changing one Hyperparameter at a time made it easier to understand the effect of each tuning decision.

EarlyStopping prevented unnecessary training.

ModelCheckpoint ensured that the best model was preserved.

Training Curves provided clear evidence of model behavior and Generalization.

---

## What Could Be Improved

Some tuning experiments required many Epochs.

The Neural Network still did not exceed the Random Forest model in Recall.

More work can be done to improve the detection of positive Heart Disease cases.

---

## Concrete Change for the Next Sprint

The next Sprint should focus more directly on:

```text
Improving Recall
Reducing False Negatives
Maintaining strong overall model performance
```

This is especially important because missing a positive Heart Disease case is more costly for the objective of this project than incorrectly flagging some negative cases.

---

# Week 6 Final Results

## Best Neural Network Configuration

| Hyperparameter | Value |
|---|---:|
| Hidden Layer 1 | 32 Neurons |
| Hidden Layer 2 | 16 Neurons |
| Hidden Activation | ReLU |
| Output Activation | Sigmoid |
| Learning Rate | 0.0001 |
| Dropout | 0.5 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| EarlyStopping | Enabled |
| ModelCheckpoint | Enabled |

---

## Final Performance

| Model | Accuracy | Recall |
|---|---:|---:|
| Random Forest | 0.880 | **0.850** |
| Original Neural Network | 0.881 | 0.790 |
| Regularized Neural Network | 0.893 | 0.793 |
| Final Tuned Neural Network | **0.912** | 0.831 |

---

# Final Conclusion

This week provided a complete introduction to Neural Networks, starting from the mathematical idea of a single neuron and progressing to a fully trained and tuned Keras model.

The project demonstrated how:

```text
Inputs
   ↓
Weights + Biases
   ↓
Activation Functions
   ↓
Forward Propagation
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradient Descent
   ↓
Weight Updates
```

work together during Neural Network training.

The original Neural Network achieved:

```text
Accuracy = 88.1%
Recall   = 79.0%
```

After applying Batch Normalization and Dropout:

```text
Accuracy = 89.3%
Recall   = 79.3%
```

After systematic Hyperparameter Tuning:

```text
Accuracy = 91.22%
Recall   = 83.09%
Loss     = 0.2397
```

The tuning process therefore produced a clear improvement in Neural Network performance.

However, the Random Forest baseline still achieved a slightly higher Recall of:

```text
85%
```

Therefore:

- The **Final Tuned Neural Network** achieved the best overall Accuracy.
- The **Random Forest** remained the strongest model according to Recall.
- The next stage should focus on reducing False Negatives and further improving Recall.