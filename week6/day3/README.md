# Day 3 — Backpropagation, Gradient Descent & Optimizers

## 📌 Overview

In this lesson, we learned how a Neural Network actually learns during training.

The training process follows a repeated cycle:

```text
Forward Pass
↓
Compute Loss
↓
Backpropagation
↓
Update Weights
↓
Repeat
```

We also learned how Gradient Descent, Learning Rate, Backpropagation, Optimizers, Epochs, and Batches work together during Neural Network training.

---

# 🎯 Learning Objectives

By the end of this lesson, we should be able to:

- Describe the four-step Neural Network training loop.
- Explain how Gradient Descent updates the weights.
- Understand the role of the Learning Rate.
- Explain the main idea of Backpropagation.
- Understand why the Chain Rule is used.
- Explain the role of Optimizers such as Adam and SGD.
- Understand Epochs and Batches.
- Compare different Learning Rates using Loss curves.

---

# 1. Neural Network Training Loop

Training a Neural Network follows four main steps.

## 1.1 Forward Pass

The input data moves through the Neural Network layer by layer until the network produces a prediction.

```text
Input
↓
Hidden Layers
↓
Output
↓
Prediction
```

---

## 1.2 Compute Loss

The prediction is compared with the true answer using a Loss Function.

The Loss tells us how wrong the prediction is.

```text
Prediction
+
True Value
↓
Loss Function
↓
Loss
```

A high Loss means that the prediction is poor.

A low Loss means that the prediction is closer to the correct answer.

---

## 1.3 Backpropagation

Backpropagation works backward through the Neural Network to calculate how much each weight contributed to the error.

```text
Loss
↓
Backpropagation
↓
Gradients
```

The gradients tell us how each weight should change in order to reduce the Loss.

---

## 1.4 Update Weights

The Optimizer uses the calculated gradients to update the weights.

After the weights are updated, the Neural Network performs another Forward Pass.

The process repeats many times during training.

```text
Forward Pass
↓
Loss
↓
Backpropagation
↓
Update Weights
↓
Repeat
```

---

# 2. Gradient Descent

Gradient Descent is the process used to move the weights in a direction that reduces the Loss.

The Gradient tells us the direction in which the Loss increases the fastest.

Because we want to reduce the Loss, Gradient Descent moves in the opposite direction.

The idea can be imagined as standing on a hill and moving downhill until reaching a lower point.

```text
High Loss
   ↓
   ↓
   ↓
Minimum Loss
```

The goal of training is to find weights that produce a small Loss.

---

# 3. Learning Rate

The Learning Rate controls the size of each weight update.

It determines how large the steps are during Gradient Descent.

```text
Small Learning Rate
→ Small weight updates
→ Slow learning

Suitable Learning Rate
→ Reasonable updates
→ Stable learning

Large Learning Rate
→ Large weight updates
→ Faster learning, but it may become unstable
```

The Learning Rate is an important Hyperparameter in Neural Network training.

A commonly used starting value with the Adam optimizer is:

```text
0.001
```

---

# 4. Learning Rate Experiment

A small Neural Network was trained using three different Learning Rates:

```python
learning_rates = [
    0.00001,
    0.001,
    0.1
]
```

The same Neural Network architecture was used for each Learning Rate.

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The network contains:

- 2 input values
- 8 neurons in the hidden layer
- ReLU activation in the hidden layer
- 1 output neuron
- Sigmoid activation in the output layer

---

## 4.1 Adam Optimizer

The Adam Optimizer was used during training.

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr
)
```

The Learning Rate changes during each experiment, while the rest of the Neural Network remains the same.

---

## 4.2 Compile the Model

The model was compiled using Adam and Binary Cross-Entropy.

```python
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy"
)
```

Binary Cross-Entropy was used because the example is a Binary Classification problem.

---

## 4.3 Train the Model

The Neural Network was trained for 100 Epochs.

```python
training_result = model.fit(
    X_train_small,
    y_train_small,
    epochs=100
)
```

The training result stores information about the training process.

For example:

```python
training_result.history["loss"]
```

returns the Loss value from every Epoch.

The Loss values were stored for each Learning Rate:

```python
histories[lr] = training_result.history["loss"]
```

---

## 4.4 Learning Rate Results

The three Learning Rates produced different training behaviors.

A Learning Rate of `0.00001` was very small. The Loss remained almost constant, which means that the model was learning very slowly.

A Learning Rate of `0.001` allowed the model to learn gradually, and the Loss decreased steadily.

A Learning Rate of `0.1` produced the fastest decrease in Loss in this experiment and approached zero without showing instability.

This experiment shows that the Learning Rate strongly affects how quickly a Neural Network learns.

---

# 5. Backpropagation

Backpropagation is the process the Neural Network uses to understand which weights caused the error.

It starts from the Loss and moves backward through the network.

```text
Loss
↑
Output Layer
↑
Hidden Layer
↑
Weights
```

Backpropagation calculates Gradients for the weights.

These gradients are later used by the Optimizer to update the weights and reduce the Loss.

---

# 6. Chain Rule

The Chain Rule is used during Backpropagation because the final Loss depends on several connected calculations.

A weight does not affect the Loss directly.

Instead, the process may look like:

```text
Weight
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

The Chain Rule allows Backpropagation to calculate how a change in an earlier weight affects the final Loss.

The main idea is:

```text
How much did the Weight affect the next calculation?
×
How much did that calculation affect the next one?
×
...
↓
Effect of the Weight on the Loss
```

This allows the Neural Network to calculate the Gradient for every weight.

---

# 7. Optimizers

The Optimizer uses the gradients calculated by Backpropagation to update the weights.

The process is:

```text
Backpropagation
↓
Gradients
↓
Optimizer
↓
Update Weights
```

Two common Optimizers are:

## SGD

SGD is a basic Gradient Descent optimizer.

It updates the weights using the calculated gradients and the Learning Rate.

## Adam

Adam is a commonly used Optimizer that adapts the update size for different weights.

It is often used as a strong default Optimizer for Neural Networks.

---

# 8. Epochs

An Epoch means one complete pass through the entire Training Dataset.

For example:

```text
Training Dataset = 9000 samples
```

One Epoch means that the Neural Network has processed all 9000 samples once.

```text
Epoch 1
→ All training samples

Epoch 2
→ All training samples again

Epoch 3
→ All training samples again
```

Training usually contains many Epochs.

---

# 9. Batches

A Batch is a smaller subset of the Training Dataset.

Instead of processing all the training data at once, the data can be divided into smaller groups.

For example:

```text
Training Samples = 9000
Batch Size = 32
```

The network processes approximately:

```text
Batch 1 → 32 samples
Batch 2 → 32 samples
Batch 3 → 32 samples
...
```

until all 9000 samples have been processed.

After that, one Epoch is complete.

The next Epoch then begins.

For each Batch, the Neural Network performs:

```text
Forward Pass
↓
Compute Loss
↓
Backpropagation
↓
Update Weights
```

Therefore, the weights can be updated many times during a single Epoch.

---

# 10. Relationship Between Epoch and Batch

The important difference is:

```text
Batch
= A small part of the Training Dataset

Epoch
= One complete pass through the entire Training Dataset
```

For example:

```text
9000 Training Samples
Batch Size = 32
```

One Epoch contains approximately:

```text
9000 ÷ 32 ≈ 282 Batches
```

After all batches are processed, the Epoch ends and the next Epoch begins.

---

# 11. Complete Neural Network Training Process

The complete training process can be summarized as:

```text
Training Data
↓
Split into Batches
↓
Forward Pass
↓
Prediction
↓
Compute Loss
↓
Backpropagation
↓
Chain Rule
↓
Calculate Gradients
↓
Optimizer
↓
Learning Rate controls update size
↓
Update Weights
↓
Next Batch
↓
Finish all Batches
↓
One Epoch Complete
↓
Start Next Epoch
```


---

# ✅ Key Takeaways

- Neural Network training follows the cycle: Forward Pass → Loss → Backpropagation → Weight Update.
- Forward Propagation produces the prediction.
- The Loss Function measures the prediction error.
- Backpropagation calculates the gradients for the weights.
- The Chain Rule allows Backpropagation to calculate how each weight affects the final Loss.
- Gradient Descent moves the weights in a direction that reduces the Loss.
- The Learning Rate controls the size of the weight updates.
- Adam and SGD are common Optimizers.
- An Epoch is one complete pass through the Training Dataset.
- A Batch is a smaller subset of the Training Dataset.
- Weight updates usually happen after each Batch.
- The Learning Rate has a strong effect on the speed and stability of Neural Network training.