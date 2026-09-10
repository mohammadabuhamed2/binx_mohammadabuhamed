# Day 2 — Building CNNs & Transfer Learning

## Overview

Day 2 extends the CNN concepts from Day 1 by building complete Convolutional Neural Networks.

The main topics include:

- Convolution layers.
- Pooling.
- Complete CNN architecture.
- Data augmentation.
- Transfer learning.
- Freezing pretrained layers.
- Fine-tuning pretrained models.

The main goal is to understand how modern image-classification systems can achieve strong results without training extremely large models from scratch.

---

# Learning Objectives

By the end of this day, I should be able to:

- Build a complete CNN using Keras.
- Understand Max Pooling.
- Combine convolution, pooling, flattening, and Dense layers.
- Explain why image models overfit.
- Apply Data Augmentation.
- Understand Transfer Learning.
- Use pretrained models such as MobileNet, ResNet, or EfficientNet.
- Freeze pretrained layers.
- Explain Fine-Tuning.

---

# 1. Pooling

After convolution, CNNs often use a Pooling layer.

Pooling reduces the spatial size of a Feature Map.

The most common type is:

**Max Pooling**

Example:

```text
2 × 2 region

[1, 5]
[2, 3]
```

Max Pooling keeps:

```text
5
```

because it is the maximum value.

---

# Why Use Pooling?

Pooling helps:

- Reduce computation.
- Reduce memory usage.
- Reduce overfitting.
- Keep important signals.
- Make the model less sensitive to small image shifts.

The common CNN pattern is:

```text
Convolution
↓
Pooling
↓
Convolution
↓
Pooling
↓
...
```

---

# 2. Complete CNN Architecture

A typical CNN has two main sections.

## Feature Extraction

```text
Conv2D
↓
MaxPooling
↓
Conv2D
↓
MaxPooling
```

These layers learn visual features.

## Classification

```text
Flatten
↓
Dense
↓
Output
```

These layers use the extracted features to predict the class.

---

# CNN Example

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

model = Sequential([
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )
])
```

---

# Understanding the Architecture

## Conv2D(32, (3,3))

```python
Conv2D(32, (3, 3), activation="relu")
```

means:

- Create 32 filters.
- Every filter is 3 × 3.
- Each filter learns a different visual pattern.

---

## MaxPooling2D

```python
MaxPooling2D((2, 2))
```

means:

- Look at 2 × 2 areas.
- Keep the maximum value from every area.
- Reduce Feature Map dimensions.

---

## Flatten

```python
Flatten()
```

converts multidimensional feature maps into a one-dimensional vector.

Example:

```text
Feature Maps
(30, 30, 64)

↓ Flatten

Vector
```

This allows the extracted features to enter Dense layers.

---

## Dense Layer

```python
Dense(64, activation="relu")
```

learns combinations of the visual features extracted by the CNN.

---

## Output Layer

For multi-class classification:

```python
Dense(num_classes, activation="softmax")
```

Softmax produces one probability for every class.

Example:

```text
Cat = 0.80
Dog = 0.15
Bird = 0.05
```

The largest probability becomes the prediction.

---

# 3. Data Augmentation

Image datasets are often too small to train large neural networks without overfitting.

**Data Augmentation** generates modified versions of existing training images.

Possible transformations include:

- Horizontal flipping.
- Rotation.
- Zoom.
- Translation.
- Cropping.

The label does not change.

Example:

```text
Original Cat Image
↓
Flip
↓
Still Cat

Original Cat Image
↓
Small Rotation
↓
Still Cat
```

---

# Keras Data Augmentation

```python
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom
)

augment = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])
```

---

# Why Data Augmentation Helps

Without augmentation:

```text
Small Dataset
↓
Model sees same images repeatedly
↓
Memorization
↓
Overfitting
```

With augmentation:

```text
Original Images
↓
Random Transformations
↓
More varied training examples
↓
Better Generalization
```

---

# 4. Transfer Learning

Training a powerful CNN from scratch can require:

- Large datasets.
- Powerful GPUs.
- Long training time.

**Transfer Learning** solves this problem by using a model that has already been trained on a large dataset.

Common pretrained image models include:

- MobileNet
- ResNet
- EfficientNet

Instead of learning basic visual features from scratch, we reuse the previously learned representations.

---

# Transfer Learning Workflow

```text
Pretrained CNN
↓
Already knows edges, shapes, textures, objects
↓
Remove / replace original classifier
↓
Add classifier for our dataset
↓
Train new classifier
```

---

# Example Using MobileNetV2

```python
from tensorflow.keras.applications import MobileNetV2

base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)
```

---

# include_top=False

```python
include_top=False
```

means:

Do not load the original final classification layers.

We only want the pretrained Feature Extractor.

---

# weights="imagenet"

```python
weights="imagenet"
```

means:

Load weights previously learned from the ImageNet dataset.

The model is therefore not starting from random weights.

---

# 5. Freezing the Pretrained Model

Initially, the pretrained backbone is usually frozen:

```python
base.trainable = False
```

This means:

```text
Pretrained Weights
→ Do not update
```

Only the newly added classification layers learn from our dataset.

Example:

```python
model = Sequential([
    base,
    Flatten(),
    Dense(
        num_classes,
        activation="softmax"
    )
])
```

---

# Why Freeze First?

The pretrained features are already useful.

If we immediately train the entire model aggressively on a small dataset, we could damage the useful pretrained representations.

So the common workflow is:

```text
Pretrained Backbone
↓
Freeze
↓
Train Classification Head
```

---

# 6. Fine-Tuning

After training the new classifier, some pretrained layers can be unfrozen.

This process is called:

**Fine-Tuning**

```text
Pretrained Weights
↓
Small Updates
↓
Adapt to our specific dataset
```

Fine-Tuning normally uses a small Learning Rate because the pretrained weights are already useful.

---

# Transfer Learning vs Fine-Tuning

| Technique | Pretrained Backbone |
|---|---|
| Feature Extraction | Frozen |
| Fine-Tuning | Some or all layers trainable |

A common strategy is:

```text
1. Load pretrained model
2. Freeze backbone
3. Train classifier
4. Unfreeze selected layers
5. Use small learning rate
6. Fine-tune
```

---

# Hands-On Lab — Building and Transferring a CNN

## Step 1 — Build CNN From Scratch

Create a CNN containing:

```text
Conv
↓
Pool
↓
Conv
↓
Pool
↓
Flatten
↓
Dense
↓
Output
```

Train the model and record:

- Training Accuracy
- Validation Accuracy
- Loss
- Training Time

---

## Step 2 — Add Data Augmentation

Add:

```text
RandomFlip
RandomRotation
RandomZoom
```

Compare the validation curves with the original CNN.

The expected benefit is improved generalization and reduced overfitting.

---

## Step 3 — Apply Transfer Learning

Load a pretrained model such as MobileNetV2.

```python
base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)

base.trainable = False
```

Add a custom output classifier and train it on the project dataset.

---

## Step 4 — Compare Approaches

Compare:

| Approach | Accuracy | Training Time | Overfitting |
|---|---:|---:|---|
| CNN From Scratch | Record Result | Record | Observe |
| CNN + Augmentation | Record Result | Record | Observe |
| Transfer Learning | Record Result | Record | Observe |

Document which model performed best and why.

---

# Key Concepts Summary

| Concept | Meaning |
|---|---|
| Pooling | Reduces feature-map dimensions |
| Max Pooling | Keeps the largest value in each region |
| Flatten | Converts feature maps into a vector |
| Data Augmentation | Creates transformed training examples |
| Transfer Learning | Reuses knowledge from another pretrained model |
| Freezing | Prevents pretrained weights from changing |
| Fine-Tuning | Makes small updates to pretrained weights |
| Pretrained Model | Model already trained on a large dataset |

---



# What I Learned

Day 2 showed that a CNN can be built from convolution and pooling blocks followed by a classifier.

It also introduced one of the most important ideas in modern Deep Learning:

**Transfer Learning**

Instead of always starting with random weights:

```text
Random Model
↓
Learn everything from scratch
```

we can use:

```text
Pretrained Model
↓
Reuse learned features
↓
Adapt to our task
```

This idea later becomes extremely important again when working with pretrained Transformers for NLP.