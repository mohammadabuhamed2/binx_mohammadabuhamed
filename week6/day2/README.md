# Day 2 — Activations, Forward Propagation & Loss

## 📌 Overview

In this lesson, we learned how activation functions, forward propagation, and loss functions work inside a Neural Network.

The main idea is that a Neural Network needs non-linear activation functions to learn complex patterns. During a forward pass, the input moves through the network layer by layer until a prediction is produced. The loss function then measures how far that prediction is from the true answer.

We also applied these concepts to the Cardiac Disease Prediction project.

---

# 🎯 Learning Objectives

By the end of this lesson, we should be able to:

- Explain why non-linear activation functions are necessary.
- Understand the behavior of ReLU, Sigmoid, and Tanh.
- Choose the correct activation function for hidden and output layers.
- Explain how Forward Propagation works.
- Choose the correct loss function for regression and classification tasks.
- Compute a simple forward pass using NumPy.

---

# 1. Why Activation Functions Matter

Without an activation function, a Neural Network behaves like a linear model even if it contains multiple layers.

Activation functions introduce non-linearity, allowing the network to learn more complex relationships and patterns.

---

# 2. Common Activation Functions

## 2.1 ReLU Activation Function

The ReLU function is defined as:

$$
\text{ReLU}(x)=\max(0,x)
$$

ReLU returns 0 for negative input values and keeps positive values unchanged.

It is commonly used in hidden layers because it is simple, efficient, and introduces non-linearity.

### ReLU Plot Interpretation

The ReLU graph is flat at 0 for negative inputs.

For positive inputs, the output increases linearly and is equal to the input value.

This is why the graph remains at 0 on the negative side and rises as a straight line on the positive side.

---

## 2.2 Sigmoid Activation Function

The Sigmoid function is defined as:

$$
\text{Sigmoid}(x)=\frac{1}{1+e^{-x}}
$$

Sigmoid transforms any input value into a value between 0 and 1.

It is mainly used in the output layer for Binary Classification because the output can be interpreted as a probability.

### Sigmoid Plot Interpretation

Large negative inputs produce outputs close to 0.

Large positive inputs produce outputs close to 1.

When the input is 0, the output is 0.5.

This creates the characteristic S-shaped curve.

---

## 2.3 Tanh Activation Function

The Tanh function is defined as:

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Tanh transforms input values into a range between -1 and 1.

Unlike Sigmoid, Tanh is centered around zero.

### Tanh Plot Interpretation

Large negative values approach -1.

Large positive values approach 1.

When the input is 0, the output is also 0.

The curve has an S-like shape similar to Sigmoid but is centered around zero.

---

# 3. Choosing Activation Functions

A practical rule is:

- Hidden Layers → ReLU
- Binary Classification Output → Sigmoid
- Multi-Class Classification Output → Softmax
- Regression Output → Linear / No Activation

For the Cardiac Disease Prediction project:

- Hidden Layer Activation: ReLU
- Output Layer Activation: Sigmoid

The output activation is Sigmoid because the target variable contains two classes:

- 0 → No Heart Disease
- 1 → Heart Disease

---

# 4. Forward Propagation

Forward Propagation is the process of sending the input data through the Neural Network to produce a prediction.

The general process is:

```text
Input
↓
Weighted Sum
↓
Activation Function
↓
Hidden Layer
↓
Weighted Sum
↓
Output Activation
↓
Prediction
```

For a layer, the weighted sum is:

$$
z=XW+b
$$

where:

- `X` represents the input values.
- `W` represents the weights.
- `b` represents the bias.
- `z` represents the weighted sum before applying the activation function.

---

# 5. Forward Pass Example

A small Neural Network was created with:

- 2 input features
- 2 neurons in one hidden layer
- ReLU activation in the hidden layer
- 1 output neuron
- Sigmoid activation in the output layer

The sample input was:

```python
X_sample = np.array([2.0, 1.0])
```

The hidden layer weights were:

```python
W1 = np.array([
    [0.5, -0.2],
    [0.3, 0.8]
])
```

The hidden layer biases were:

```python
b1 = np.array([0.1, -0.1])
```

---

## 5.1 Hidden Layer Weighted Sum

The hidden layer weighted sum is calculated as:

$$
z_1=XW_1+b_1
$$

The result was:

```text
[1.4, 0.3]
```

---

## 5.2 Apply ReLU

The ReLU activation is applied to the hidden layer:

$$
a_1=\text{ReLU}(z_1)
$$

Since both values were positive, the result remained:

```text
[1.4, 0.3]
```

---

## 5.3 Output Layer Weighted Sum

The output layer weights and bias were:

```python
W2 = np.array([0.7, -0.5])
b2 = 0.2
```

The output weighted sum is:

$$
z_2=a_1W_2+b_2
$$

The result was approximately:

```text
1.03
```

---

## 5.4 Apply Sigmoid

The Sigmoid activation is applied to the output layer:

$$
\hat{y}=\text{Sigmoid}(z_2)
$$

The final output was approximately:

```text
0.74
```

This means that the small example network produced approximately a 74% probability for the positive class.

This example was created only to demonstrate how Forward Propagation works and is not a real patient prediction.

---

# 6. Loss Functions

After Forward Propagation produces a prediction, the Loss Function measures how far the prediction is from the true value.

A lower loss means that the prediction is closer to the correct answer.

The correct loss function depends on the task.

| Task | Loss Function |
|---|---|
| Regression | Mean Squared Error |
| Binary Classification | Binary Cross-Entropy |
| Multi-Class Classification | Categorical Cross-Entropy |

---

## 6.1 Mean Squared Error

Mean Squared Error is commonly used for Regression.

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

It measures the average squared difference between the true values and predicted values.

---

## 6.2 Binary Cross-Entropy

Binary Cross-Entropy is used for Binary Classification.

$$
L=-\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]
$$

It compares the predicted probability with the true binary label.

Predictions that are far from the correct answer receive a higher loss.

---

## 6.3 Categorical Cross-Entropy

Categorical Cross-Entropy is used for Multi-Class Classification.

$$
L=-\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

It compares the predicted class probabilities with the true class.

---

# 7. Cardiac Disease Project Design Choices

The Cardiac Disease Prediction project is a Binary Classification problem.

Therefore, the selected Neural Network design choices are:

- Hidden Layer Activation: **ReLU**
- Output Layer Activation: **Sigmoid**
- Loss Function: **Binary Cross-Entropy**

ReLU is selected for the hidden layers because it introduces non-linearity and is efficient during training.

Sigmoid is selected for the output layer because it produces a value between 0 and 1 that can represent the probability of Heart Disease.

Binary Cross-Entropy is selected because the target has two possible classes.

---

# 8. Hands-On Lab Summary

During the practical lab, we:

1. Plotted ReLU, Sigmoid, and Tanh activation functions.
2. Interpreted the shape and behavior of each activation function.
3. Selected Sigmoid as the output activation for the Cardiac Disease Prediction project.
4. Selected Binary Cross-Entropy as the loss function.
5. Built a small Neural Network example using NumPy.
6. Computed the hidden layer weighted sum.
7. Applied ReLU activation.
8. Computed the output layer weighted sum.
9. Applied Sigmoid activation.
10. Obtained a final forward-pass output of approximately **0.74**.

---

# ✅ Key Takeaways

- Activation functions introduce non-linearity into Neural Networks.
- ReLU is the default choice for most hidden layers.
- Sigmoid is suitable for Binary Classification output.
- Softmax is suitable for Multi-Class Classification output.
- Forward Propagation moves data through the network to produce a prediction.
- The Loss Function measures how wrong the prediction is.
- Binary Cross-Entropy is appropriate for the Cardiac Disease Prediction task.
- The Neural Network will later learn its weights and biases during training instead of using manually selected values.