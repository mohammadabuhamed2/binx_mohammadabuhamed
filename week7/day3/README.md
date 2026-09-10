# Day 3 — RNNs & LSTMs for Sequential Data

## Overview

Day 3 focuses on **Sequential Data** and neural-network architectures that understand order.

Unlike image data, sequential data contains relationships across time or position.

Examples include:

- Text
- ECG signals
- Time series
- Audio
- Sensor measurements

The main architectures studied are:

- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory networks (LSTMs)
- Gated Recurrent Units (GRUs)

The practical lab used ECG heartbeat sequences to compare an LSTM against a Simple RNN.

---

# Learning Objectives

By the end of this day, I should be able to:

- Explain why order matters in sequential data.
- Explain how an RNN processes sequences.
- Understand the Hidden State.
- Explain the Vanishing Gradient Problem.
- Understand how LSTM improves long-term memory.
- Understand the basic idea of GRU.
- Understand Embeddings for text.
- Build an LSTM using TensorFlow/Keras.
- Evaluate a sequence classification model.
- Compare LSTM with SimpleRNN.

---

# 1. Why Sequential Data Needs a Different Architecture

In sequential data, order carries information.

For example:

```text
"The movie was not good."
```

is different from rearranging the same words.

Similarly, an ECG signal contains measurements over time.

```text
t1 → t2 → t3 → t4 → ...
```

The relationship between these measurements is important.

A model that ignores order may lose critical information.

---

# 2. Recurrent Neural Networks

An RNN processes a sequence one step at a time.

```text
x1 → RNN → h1
          ↓
x2 → RNN → h2
          ↓
x3 → RNN → h3
```

The important component is:

**Hidden State**

The Hidden State acts like a memory containing information from previous steps.

At every step:

```text
Current Input
+
Previous Hidden State
↓
New Hidden State
```

---

# 3. RNN Memory

Consider a sentence:

```text
"The movie was surprisingly very good"
```

The RNN reads:

```text
The
↓
movie
↓
was
↓
surprisingly
↓
very
↓
good
```

Information from previous words is passed forward through the Hidden State.

This allows later words to be interpreted using earlier context.

---

# 4. The Vanishing Gradient Problem

During training, neural networks use Backpropagation.

For an RNN, gradients must travel backward through many time steps.

```text
Step 100
↓
Step 99
↓
...
↓
Step 1
```

When these gradients repeatedly become smaller, information from earlier steps receives almost no useful update.

This is called:

**Vanishing Gradient**

The result is:

```text
Long Sequence
↓
Plain RNN gradually forgets old information
```

---

# 5. LSTM

**LSTM = Long Short-Term Memory**

LSTM is a special type of RNN designed to preserve useful information over longer sequences.

It contains a memory mechanism with gates.

Conceptually, these gates determine:

```text
What should I remember?
What should I forget?
What should I output?
```

This allows LSTM to retain important context for longer periods.

---

# 6. LSTM vs SimpleRNN

| SimpleRNN | LSTM |
|---|---|
| Simpler architecture | More complex architecture |
| Fewer parameters | More parameters |
| Faster | Usually slower |
| Weak long-term memory | Better long-term memory |
| More affected by vanishing gradients | Designed to reduce this problem |

---

# 7. GRU

**GRU = Gated Recurrent Unit**

GRU follows a similar idea to LSTM but uses a simpler gating structure.

It usually:

- Has fewer parameters.
- Trains faster.
- Can achieve performance similar to LSTM.

The best choice depends on the dataset.

---

# 8. Embeddings

For text, neural networks cannot process raw words directly.

Words are first converted into IDs.

```text
movie → 15
good → 27
bad → 83
```

An Embedding Layer converts each ID into a learnable vector.

```text
Token ID
↓
Embedding
↓
Vector
```

Example:

```python
from tensorflow.keras.layers import Embedding

Embedding(
    input_dim=10000,
    output_dim=128
)
```

This means:

- Vocabulary contains up to 10,000 token IDs.
- Each token is represented using 128 numbers.

---

# Example Text LSTM Architecture

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=128
    ),

    LSTM(64),

    Dense(
        1,
        activation="sigmoid"
    )
])
```

For binary sentiment classification:

```text
Sigmoid output
↓
Probability between 0 and 1
```

---

# Hands-On Lab — ECG Sequence Classification

## Dataset

The practical sequence-modeling experiment used the:

**MIT-BIH Arrhythmia ECG Dataset**

The training dataset contained:

```text
87,554 samples
```

and the test dataset contained:

```text
21,892 samples
```

Each sample contained:

```text
187 ECG measurements
+
1 class label
```

---

# ECG Classes

| Class | Meaning |
|---:|---|
| 0 | Normal |
| 1 | Supraventricular |
| 2 | Ventricular |
| 3 | Fusion |
| 4 | Unknown |

---

# Separate Features and Target

```python
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
```

The first 187 columns represent the ECG sequence.

The last column represents the class.

---

# Reshape the ECG Data

LSTM expects data in this form:

```text
(samples, time_steps, features)
```

The ECG data contains:

```text
187 time steps
1 value at every step
```

Therefore:

```python
X_train = X_train.values.reshape(
    -1,
    187,
    1
)

X_test = X_test.values.reshape(
    -1,
    187,
    1
)
```

The final shape is:

```text
(samples, 187, 1)
```

---

# Understanding -1 in reshape()

```python
reshape(-1, 187, 1)
```

`-1` tells NumPy:

> Automatically calculate the number of samples.

We manually specify:

```text
187 time steps
1 feature
```

and NumPy determines the remaining dimension.

---

# Class Distribution

The ECG dataset was highly imbalanced.

Approximate training distribution:

| Class | Percentage |
|---:|---:|
| 0 | 82.77% |
| 1 | 2.54% |
| 2 | 6.61% |
| 3 | 0.73% |
| 4 | 7.35% |

Therefore, Accuracy alone is not sufficient for understanding model quality.

Recall and F1-score are also important.

---

# Train / Validation Split

A stratified validation split was created:

```python
from sklearn.model_selection import train_test_split

X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
```

Using:

```python
stratify=y_train
```

preserves approximately the same class distribution in Training and Validation sets.

This was especially important because the original ECG dataset ordering could create an invalid validation set if the final portion of the dataset was used directly.

---

# LSTM Model

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(
        shape=(187, 1)
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(
        5,
        activation="softmax"
    )
])
```

---

# LSTM Output

The task has five classes.

Therefore:

```python
Dense(
    5,
    activation="softmax"
)
```

produces five probabilities.

Example:

```text
[0.05, 0.02, 0.88, 0.03, 0.02]
```

The predicted class is selected using:

```python
np.argmax(...)
```

because Class 2 has the largest probability.

---

# Compile

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

`SparseCategoricalCrossentropy` is appropriate because:

- There are multiple classes.
- Labels are stored as integers such as 0, 1, 2, 3, 4.

---

# Callbacks

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

EarlyStopping:

- Monitors Validation Loss.
- Stops unnecessary training.
- Reduces overfitting.
- Restores the best model weights.

---

# LSTM Test Result

The LSTM achieved approximately:

```text
Test Accuracy = 97.54%
Test Loss = 0.0905
```

---

# LSTM Classification Report

Important results included approximately:

| Class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 0.98 | 1.00 | 0.99 |
| 1 | 0.92 | 0.59 | 0.72 |
| 2 | 0.93 | 0.92 | 0.92 |
| 3 | 0.72 | 0.73 | 0.72 |
| 4 | 0.99 | 0.96 | 0.98 |

Overall:

```text
Accuracy ≈ 0.98
Macro F1 ≈ 0.87
Weighted F1 ≈ 0.97
```

The lower Recall for Class 1 showed why class-specific metrics are important with imbalanced datasets.

---

# SimpleRNN Comparison

A SimpleRNN model was built using the same basic structure.

```python
rnn_model = tf.keras.Sequential([
    tf.keras.Input(
        shape=(187, 1)
    ),

    tf.keras.layers.SimpleRNN(64),

    tf.keras.layers.Dense(
        5,
        activation="softmax"
    )
])
```

---

# SimpleRNN Test Result

```text
Test Accuracy ≈ 82.97%
Test Loss ≈ 0.5293
```

The SimpleRNN performed well mostly on the dominant Normal class but struggled heavily with minority classes.

Approximate Recall:

| Class | LSTM | SimpleRNN |
|---:|---:|---:|
| 0 | 1.00 | 1.00 |
| 1 | 0.59 | 0.01 |
| 2 | 0.92 | 0.03 |
| 3 | 0.73 | 0.00 |
| 4 | 0.96 | 0.00 |

---

# LSTM vs SimpleRNN

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| SimpleRNN | 82.97% | 0.5293 |
| LSTM | 97.54% | 0.0905 |

The LSTM clearly performed better.

Its gated memory allowed it to preserve useful information across the ECG sequence more effectively than a plain RNN.

---

# Why Order Awareness Matters for ECG

An ECG is not simply a collection of unrelated numbers.

Its values form a waveform over time.

```text
Time Step 1
↓
Time Step 2
↓
Time Step 3
↓
...
```

Changing this order destroys the waveform structure.

Therefore, sequence-aware models such as LSTM are suitable for ECG classification.

---

# Key Concepts Summary

| Concept | Meaning |
|---|---|
| Sequential Data | Data where order matters |
| RNN | Processes sequence step-by-step |
| Hidden State | Memory passed between steps |
| Vanishing Gradient | Gradients become too small across long sequences |
| LSTM | RNN with gated long-term memory |
| GRU | Simpler gated recurrent architecture |
| Embedding | Converts token IDs into learned vectors |
| Softmax | Produces probabilities for multiple classes |
| Argmax | Selects the index of the largest probability |

---


# Main Conclusion

The practical experiment demonstrated why LSTM is useful for long sequential data.

```text
SimpleRNN
→ Step-by-step memory
→ Weak long-term retention
→ 82.97%

LSTM
→ Gated memory
→ Better long-range sequence learning
→ 97.54%
```

The most important lesson is:

**When order carries useful information, the model architecture should preserve and learn that order.**