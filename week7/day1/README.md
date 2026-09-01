# Day 1 — Sprint 2 Planning & Convolutional Neural Networks

## 📌 Overview

In Day 1, we started Sprint 2 and introduced Convolutional Neural Networks (CNNs).

The main goal was to understand why dense neural networks are not suitable for image data and how CNNs use convolution and filters to detect visual patterns efficiently.

We also applied a manually defined edge-detection filter to a sample image and visualized the resulting feature map.

---

## 🎯 Learning Objectives

By the end of this lesson, we should be able to:

1. Complete Sprint 2 planning and define the core-model backlog.
2. Explain why dense neural networks are inefficient for images.
3. Understand convolution and how filters move across an image.
4. Explain filters, feature maps, stride, and padding.
5. Understand parameter sharing.
6. Explain why CNNs require fewer weights than dense networks.
7. Select CNNs as the appropriate architecture for image data.

---

# 1. Sprint 2 Planning

Sprint 2 focuses on developing and improving the project's core model.

## Sprint 2 Goal

Develop and improve the project's core model using the architecture that best fits the data type, evaluate different approaches, and achieve better performance than the Week 6 baseline.

## Sprint 2 Backlog

- Review the Sprint 1 baseline and identify one improvement to carry forward.
- Study CNNs and understand convolution, filters, feature maps, stride, and padding.
- Build and train a CNN on image data.
- Experiment with data augmentation.
- Apply transfer learning.
- Compare relevant architectures.
- Select the final core architecture.
- Train and tune the selected model.
- Record experiment settings and evaluation metrics.
- Compare Sprint 2 results with previous baselines.
- Prepare results for the Sprint Review.

## Sprint 1 Improvement to Carry Forward

In Sprint 2, experiments will be documented more clearly by recording the model configuration and evaluation metric for each run.

---

# 2. Why Dense Networks Are Not Ideal for Images

Images contain spatial information.

Nearby pixels are related, and visual patterns such as edges, shapes, and textures may appear anywhere in an image.

A dense neural network usually flattens the image into a long vector, which does not preserve this spatial structure effectively.

Dense networks can also require a very large number of weights.

For example:

`200 × 200 × 3 = 120,000 input values`

Connecting these inputs to a dense layer can create millions of weights.

CNNs solve this problem by using small filters that process local regions of the image.

---

# 3. Convolution

Convolution is the main operation used in CNNs.

A small filter moves across the image and performs a calculation at each position.

The basic operation is:

`Image Region × Filter → Sum → Feature Map Value`

---

## Filter / Kernel

A Filter, also called a Kernel, is a small matrix of weights used to detect a specific pattern.

Example of a vertical edge-detection filter:

```text
-1   0   1
-1   0   1
-1   0   1
```

The filter compares pixel values on the left side with pixel values on the right side.

A strong difference produces a large output value.

---

## Feature Map

A Feature Map is the output produced after applying a filter to an image.

It shows where the filter detected its pattern.

Higher values represent stronger matches with the filter.

---

## Stride

Stride controls how far the filter moves after each convolution operation.

For example:

`stride = 1`

means the filter moves one pixel at a time.

---

## Padding

Padding adds pixels around the border of an image.

Common options include:

- `VALID` → no padding is added.
- `SAME` → padding is added to help preserve the image dimensions.

---

# 4. Applying an Edge Detection Filter

## Import Libraries

```python
import tensorflow as tf
import matplotlib.pyplot as plt
```

---

## Create a Sample Image

```python
image = tf.constant([
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255],
    [0, 0, 0, 255, 255]
], dtype=tf.float32)
```

Pixel values:

- `0` → black
- `255` → white

The image contains a clear vertical edge between the dark and bright regions.

---

## Display the Original Image

```python
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()
```

---

## Reshape the Image

TensorFlow expects image data for `tf.nn.conv2d` in the following format:

`[Batch, Height, Width, Channels]`

For our sample image:

- `Batch = 1`
- `Height = 5`
- `Width = 5`
- `Channels = 1`

General format:

`[number_of_images, height, width, channels]`

```python
image_reshaped = tf.reshape(
    image,
    [1, 5, 5, 1]
)
```

---

## Define the Vertical Edge Detection Filter

```python
filter_ = tf.constant([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=tf.float32)
```

The filter has a size of:

`3 × 3`

---

## Reshape the Filter

TensorFlow expects the convolution filter in the following format:

`[Filter Height, Filter Width, Input Channels, Number of Filters]`

For our filter:

- `Filter Height = 3`
- `Filter Width = 3`
- `Input Channels = 1`
- `Number of Filters = 1`

General format:

`[filter_height, filter_width, input_channels, number_of_filters]`

```python
filter_reshaped = tf.reshape(
    filter_,
    [3, 3, 1, 1]
)
```

---

## `tf.nn.conv2d` vs `Conv2D`

- `tf.nn.conv2d` → applies a manually defined filter to an image.
- `Conv2D` → creates learnable filters inside a CNN model.

For this example, we use `tf.nn.conv2d` because the filter is defined manually.

---

## Apply the Convolution

```python
feature_map = tf.nn.conv2d(
    image_reshaped,
    filter_reshaped,
    strides=1,
    padding="VALID"
)
```

The filter moves one pixel at a time because:

`strides = 1`

No padding is added because:

`padding = "VALID"`

The basic convolution calculation is:

`Feature Map Value = Sum(Image Region × Filter)`

---

## Feature Map Output Size

Without padding, the output size is calculated using:

`Output Size = ((N - F) / S) + 1`

Where:

- `N` = input size
- `F` = filter size
- `S` = stride

For this example:

`N = 5`

`F = 3`

`S = 1`

Therefore:

`Output Size = ((5 - 3) / 1) + 1 = 3`

So the resulting feature map has a size of:

`3 × 3`

---

## Convert the Feature Map to 2D

`tf.nn.conv2d` returns the output with extra dimensions.

For this example:

`(1, 3, 3, 1)`

We use `tf.squeeze()` to remove dimensions of size 1.

```python
feature_map_2d = tf.squeeze(feature_map)
```

The result becomes:

`(3, 3)`

---

## Compare the Original Image and Feature Map

```python
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(feature_map_2d, cmap="gray")
ax[1].set_title("Feature Map")
ax[1].axis("off")

plt.show()
```

The original image and the feature map are displayed side by side.

- In the original image, black pixels have low values and white pixels have high values.
- After applying the edge-detection filter, areas with stronger edge responses appear brighter.
- White areas in the feature map represent higher values, meaning the filter detected a stronger vertical edge there.
- Darker areas represent lower values, meaning little or no edge was detected.

---

# 5. Parameter Sharing

A CNN uses the same small filter across the entire image.

For example, a 3×3 grayscale filter contains:

`3 × 3 = 9 weights`

The same 9 weights are reused at every location in the image.

This is called Parameter Sharing.

A dense layer uses many different weights for its connections, which can create a much larger number of parameters.

Because CNNs reuse the same filters, they are much more efficient for image data.

---

# 6. Feature Hierarchy in CNNs

CNNs learn features gradually through multiple layers.

Early layers may learn:

- Edges
- Lines
- Corners

Middle layers may learn:

- Shapes
- Curves
- Textures

Deeper layers may learn:

- Eyes
- Faces
- Wheels
- Complete objects

The network learns these features automatically during training.

---

# 7. Core Model Architecture Decision

The project uses image data, so a Convolutional Neural Network (CNN) is the appropriate core architecture.

CNNs are designed to preserve spatial information in images, detect visual patterns using filters, and learn features such as edges, shapes, textures, and objects.

---

# ✅ Day 1 Summary

In this lesson, we learned:

- Why dense neural networks are inefficient for image data.
- How CNNs preserve spatial information.
- What convolution means.
- What filters and kernels are.
- How a filter creates a feature map.
- What stride controls.
- What padding does.
- How to apply a manually defined edge-detection filter.
- How to calculate the feature map output size.
- Why `tf.squeeze()` is useful for displaying the feature map.
- Why CNNs use fewer weights through parameter sharing.
- How CNNs learn a hierarchy of visual features.
- Why CNNs are the correct core architecture for image-based projects.