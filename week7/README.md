# Week 7 — Advanced Deep Learning Architectures & Sprint 2

## Overview

Week 7 focused on selecting and applying neural-network architectures that match different types of data.

The week progressed from image-based deep learning with Convolutional Neural Networks (CNNs), to sequential modeling with RNNs and LSTMs, and finally to Attention, Transformers, Transfer Learning, and Fine-Tuning.

The main goal of Sprint 2 was to move beyond the initial neural-network baseline and develop a stronger core model based on the structure of the project data.

During this week, the following major architectures were studied:

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory networks (LSTMs)
- GRUs
- Attention
- Transformers
- Pretrained Transformers
- Transfer Learning
- Fine-Tuning

The week concluded with a full model comparison and Sprint 2 Review.

---

# Week Learning Objectives

By the end of Week 7, I was able to:

- Select a neural-network architecture based on the type of data.
- Explain why CNNs are suitable for image data.
- Understand convolution, filters, feature maps, stride, and padding.
- Build a complete CNN architecture.
- Understand Data Augmentation.
- Explain Transfer Learning and Fine-Tuning.
- Understand how RNNs process sequential data.
- Explain the Hidden State.
- Understand the Vanishing Gradient Problem.
- Explain how LSTMs improve long-term sequence memory.
- Build and evaluate LSTM models.
- Compare LSTM with SimpleRNN.
- Explain Attention and Self-Attention.
- Understand Query, Key, and Value.
- Understand Transformer Encoder and Decoder architectures.
- Use pretrained Transformers.
- Use Hugging Face pipelines for inference.
- Fine-tune a pretrained RoBERTa Transformer using Keras.
- Use EarlyStopping and ModelCheckpoint.
- Evaluate models using Accuracy, Precision, Recall, F1-score, and Loss.
- Compare multiple model architectures.
- Complete the Sprint Review and Retrospective.

---

# Week Roadmap

```text
Day 1
CNN Fundamentals
        ↓
Day 2
CNN Architecture + Transfer Learning
        ↓
Day 3
RNN + LSTM for Sequential Data
        ↓
Day 4
Attention + Transformers
        ↓
Day 5
Core Model Fine-Tuning + Sprint Review
```

---

# Day 1 — Sprint 2 Planning & Convolutional Neural Networks

## Main Topics

- Sprint 2 planning
- Why Dense Networks are inefficient for images
- Convolution
- Filters / Kernels
- Feature Maps
- Stride
- Padding
- Parameter Sharing
- Translation Invariance
- CNN Feature Hierarchy

---

## Sprint 2 Planning

Sprint 2 focused on improving the project's core model.

The main architecture-selection principle was:

| Data Type | Recommended Architecture |
|---|---|
| Tabular Data | Dense Network / Classical ML |
| Images | CNN |
| Text | RNN / LSTM / Transformer |
| Sequential / Time-Series Data | LSTM / GRU / Transformer |

The goal is not to choose the most complicated architecture.

The architecture should match the structure of the data.

---

## Why Dense Networks Are Not Ideal for Images

Images contain spatial relationships between nearby pixels.

For example, a color image with:

```text
200 × 200 × 3
```

contains:

```text
120,000 pixel values
```

Connecting every pixel to a large Dense layer would require a huge number of parameters.

Dense Networks also lose spatial relationships when the image is flattened.

CNNs solve this problem by processing small local regions of the image.

---

## Convolution

A convolution uses a small matrix called a:

**Filter / Kernel**

Example:

```text
3 × 3 Filter
```

The filter moves across the image and calculates a local Dot Product.

```text
Image
↓
Filter slides across image
↓
Local Dot Products
↓
Feature Map
```

Different filters can learn different patterns such as:

- Edges
- Curves
- Corners
- Textures
- Shapes

---

## Feature Map

The output of a convolution operation is called a:

**Feature Map**

A Feature Map shows where a particular learned pattern appears in the image.

```text
Image
↓
Edge Filter
↓
Edge Feature Map
```

---

## Stride

Stride controls how far the filter moves.

```text
Stride = 1
→ Move one pixel

Stride = 2
→ Move two pixels
```

Increasing stride reduces the output dimensions and computation.

---

## Padding

Padding adds additional values around image borders.

It helps:

- Preserve spatial dimensions.
- Process edge pixels.
- Prevent feature maps from shrinking too quickly.

---

## Parameter Sharing

CNNs reuse the same filter across the entire image.

```text
One Filter
↓
Used at many image locations
```

This dramatically reduces the number of parameters compared with Dense Networks.

---

## Translation Invariance

Because a filter moves across the image, it can detect a pattern regardless of where it appears.

For example:

```text
Edge on left side
Edge in center
Edge on right side
```

The same filter can detect all of them.

---

## CNN Feature Hierarchy

CNNs learn features hierarchically.

```text
Pixels
↓
Edges
↓
Shapes
↓
Textures / Parts
↓
Objects
```

Early layers usually learn simple visual patterns.

Deeper layers combine these patterns into increasingly complex representations.

---

# Day 2 — Building CNNs & Transfer Learning

## Main Topics

- Pooling
- Complete CNN architecture
- Data Augmentation
- Transfer Learning
- Pretrained CNNs
- Freezing layers
- Fine-Tuning

---

## Pooling

Pooling reduces Feature Map dimensions.

The most common form is:

**Max Pooling**

Example:

```text
[1, 5]
[2, 3]
```

Max Pooling keeps:

```text
5
```

Pooling helps:

- Reduce computation.
- Reduce memory usage.
- Reduce overfitting.
- Preserve strong visual signals.

---

## Complete CNN Architecture

A typical CNN follows:

```text
Image
↓
Conv2D
↓
MaxPooling
↓
Conv2D
↓
MaxPooling
↓
Flatten
↓
Dense
↓
Output
```

Example:

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

## Data Augmentation

Data Augmentation creates modified versions of training images.

Examples include:

- Horizontal Flip
- Rotation
- Zoom
- Translation

Example:

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

The goal is to increase training variety and reduce overfitting.

---

## Transfer Learning

Training a strong CNN from scratch can require large datasets and significant computational resources.

Transfer Learning reuses a model that has already been trained on a large dataset.

Examples:

- MobileNet
- ResNet
- EfficientNet

Workflow:

```text
Pretrained Model
↓
Reuse learned features
↓
Replace original classifier
↓
Train classifier for new task
```

---

## MobileNetV2 Example

```python
from tensorflow.keras.applications import MobileNetV2

base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)

base.trainable = False
```

---

## Freezing

```python
base.trainable = False
```

means:

> Do not update the pretrained weights.

The pretrained model works as a Feature Extractor.

---

## Fine-Tuning

Fine-Tuning means allowing some or all pretrained weights to update on the new dataset.

```text
Pretrained Weights
↓
Small Updates
↓
Adapt to New Task
```

This concept became especially important later when fine-tuning RoBERTa for text classification.

---

# Day 3 — RNNs & LSTMs for Sequential Data

## Main Topics

- Sequential Data
- RNN
- Hidden State
- Vanishing Gradient
- LSTM
- GRU
- Embeddings
- Sequence Classification

---

## Why Order Matters

Sequential data depends on order.

Examples:

- Text
- ECG
- Time-Series Data
- Audio

For example:

```text
"The movie was not good"
```

cannot be treated as unrelated individual words.

The model must understand the sequence.

---

## Recurrent Neural Network

RNNs process sequences step-by-step.

```text
Input 1
↓
Hidden State
↓
Input 2
↓
Hidden State
↓
Input 3
```

The Hidden State acts as memory.

---

## Vanishing Gradient

During Backpropagation Through Time, gradients travel through many time steps.

They may become increasingly small.

```text
Long Sequence
↓
Gradient becomes smaller
↓
Early information receives weak updates
```

This is called:

**Vanishing Gradient**

Plain RNNs therefore struggle with long-term dependencies.

---

## LSTM

**LSTM = Long Short-Term Memory**

LSTM uses gated memory to decide:

```text
What should be remembered?
What should be forgotten?
What should be output?
```

This allows LSTM to preserve important information for longer sequences.

---

## GRU

GRU is another gated recurrent architecture.

Compared with LSTM:

- It usually has fewer parameters.
- It is simpler.
- It can train faster.
- It often produces similar performance.

---

# Day 3 Practical Project — ECG Classification

The practical experiment used the MIT-BIH ECG heartbeat dataset.

Each ECG sample contained:

```text
187 ECG measurements
+
1 class label
```

---

## ECG Classes

| Class | Meaning |
|---:|---|
| 0 | Normal |
| 1 | Supraventricular |
| 2 | Ventricular |
| 3 | Fusion |
| 4 | Unknown |

---

## Prepare Features and Target

```python
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
```

---

## Reshape for LSTM

LSTM expects:

```text
(samples, time_steps, features)
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

---

## Stratified Validation Split

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

Stratification was important because the ECG classes were highly imbalanced.

---

## LSTM Architecture

```python
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

## Compile

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## LSTM ECG Results

```text
Test Accuracy = 97.54%
Test Loss = 0.0905
```

Approximate F1 results:

| Class | F1 |
|---:|---:|
| 0 | 0.99 |
| 1 | 0.72 |
| 2 | 0.92 |
| 3 | 0.72 |
| 4 | 0.98 |

---

## SimpleRNN Comparison

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

Result:

```text
Test Accuracy ≈ 82.97%
Test Loss ≈ 0.5293
```

---

## LSTM vs SimpleRNN

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| SimpleRNN | 82.97% | 0.5293 |
| LSTM | **97.54%** | **0.0905** |

The LSTM clearly handled long sequential dependencies more effectively.

---

# Day 4 — Attention & Transformers

## Main Topics

- RNN limitations
- Attention
- Self-Attention
- Query / Key / Value
- Multi-Head Attention
- Positional Encoding
- Transformer Architecture
- Encoder
- Decoder
- Pretrained Transformers
- Hugging Face

---

## Limitation of RNNs

RNN and LSTM process sequences step-by-step.

```text
Token 1
↓
Token 2
↓
Token 3
```

This limits parallel processing and can make long sequences computationally expensive.

---

## Attention

Attention allows each token to directly examine other tokens.

Example:

```text
"The movie was not good"
```

When processing:

```text
good
```

Attention can directly focus on:

```text
not
```

---

## Self-Attention

Each token calculates how relevant other tokens are.

```text
Token
↓
Looks at all tokens
↓
Calculates relevance
↓
Weighted contextual representation
```

---

## Query, Key, Value

A simple way to remember them:

```text
Query
→ What am I looking for?

Key
→ What do I represent?

Value
→ What information do I provide?
```

Conceptually:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dk)V
```

---

## Multi-Head Attention

Transformers use multiple Attention Heads.

Different Heads can learn different relationships.

```text
Head 1 → Short-distance relationship
Head 2 → Long-distance relationship
Head 3 → Semantic relationship
Head 4 → Structural relationship
```

---

## Positional Information

Transformers process sequence positions in parallel.

Therefore positional information must be included explicitly.

```text
Token Representation
+
Position Representation
↓
Transformer
```

---

## Encoder vs Decoder

### Encoder

Main purpose:

```text
Understand Input
```

Examples:

- BERT
- RoBERTa
- DistilBERT

Suitable for:

- Classification
- Sentiment Analysis
- Information Extraction

### Decoder

Main purpose:

```text
Generate Output
```

Example:

- GPT

Workflow:

```text
Previous Tokens
↓
Predict Next Token
↓
Repeat
```

---

# Pretrained Transformers

Instead of training Transformers from scratch, pretrained models can be reused.

Examples:

- BERT
- DistilBERT
- RoBERTa
- GPT

---

## Hugging Face Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)
```

Example:

```python
texts = [
    "This course is amazing!",
    "I really hate this movie.",
    "The project was difficult but very useful."
]

results = classifier(texts)
```

---

# IMDB Transformer Experiment

The pretrained Transformer was evaluated on the same IMDB test split.

```python
transformer_results = classifier(
    X_test.tolist(),
    batch_size=32,
    truncation=True
)
```

Approximate result:

```text
Accuracy ≈ 89%
F1-score ≈ 0.89
```

---

## Why It Did Not Beat the LSTM

The LSTM had been trained specifically on IMDB.

The pretrained Transformer pipeline had not been fine-tuned on our training dataset.

```text
LSTM
→ IMDB-specific training

Pretrained Transformer
→ General pretrained knowledge
```

This motivated the Fine-Tuning experiment on Day 5.

---

# Day 5 — Advancing the Core Model & Sprint Review

## Main Goal

Day 5 focused on:

- Selecting the final architecture.
- Improving the core model.
- Fine-Tuning.
- Logging experiments.
- Comparing models.
- Completing Sprint 2.
- Sprint Review.
- Retrospective.

---

# Day 5 Project — IMDB Sentiment Classification

The dataset contained:

```text
50,000 Reviews
```

Target:

```text
Negative
Positive
```

Encoded as:

```text
0 = Negative
1 = Positive
```

---

## Dataset Split

```text
Train      = 40,000
Validation = 5,000
Test       = 5,000
```

---

# LSTM Sentiment Model

Text preprocessing used:

```python
tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=400
)
```

The selected sequence length was based on review-length analysis.

---

## LSTM Architecture

```python
lstm_model = tf.keras.Sequential([
    tf.keras.Input(
        shape=(400,)
    ),

    tf.keras.layers.Embedding(
        input_dim=10000,
        output_dim=128,
        mask_zero=True
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
```

---

## LSTM Results

```text
Test Accuracy = 89.76%
Test Loss = 0.2710
F1-score = 0.90
```

---

# Pretrained Transformer Result

Using a pretrained Hugging Face Sentiment Pipeline without task-specific Fine-Tuning:

```text
Accuracy ≈ 89%
F1-score ≈ 0.89
```

---

# Fine-Tuning RoBERTa

A pretrained RoBERTa Transformer was selected as the improved core model.

The selected preset was:

```text
roberta_base_en
```

---

## Load the Matching Tokenizer

```python
roberta_tokenizer = keras_hub.models.Tokenizer.from_preset(
    "roberta_base_en"
)
```

The tokenizer converts:

```text
Raw Text
↓
Tokens
↓
Token IDs
```

---

## Load the Pretrained Backbone

```python
roberta_backbone = keras_hub.models.Backbone.from_preset(
    "roberta_base_en"
)
```

The RoBERTa Base Backbone contained approximately:

```text
124 million parameters
```

and:

```text
12 Transformer Encoder Layers
```

---

# Backbone Inputs

RoBERTa expects:

```text
token_ids
padding_mask
```

---

# TensorFlow Dataset Pipeline

The Pandas data was converted to:

```python
tf.data.Dataset
```

Example:

```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_train.astype(str).tolist(),
        y_train.to_numpy()
    )
)
```

This keeps each Review paired with its correct Label.

---

# Batching

```text
40,000 Reviews
↓
Smaller Batches
↓
Model
```

Batching reduces GPU memory requirements.

---

# StartEndPacker

The packer performs:

```text
Token IDs
↓
Start Token
↓
End Token
↓
Padding or Truncation
↓
Padding Mask
```

Final sequence length:

```python
sequence_length = 256
```

Example:

```python
packer = keras_hub.layers.StartEndPacker(
    sequence_length=sequence_length,
    start_value=roberta_tokenizer.start_token_id,
    end_value=roberta_tokenizer.end_token_id,
    pad_value=roberta_tokenizer.pad_token_id,
    return_padding_mask=True
)
```

---

# Why 256 Instead of 512?

The initial experiment used:

```text
512 Tokens
```

but Full Fine-Tuning became extremely slow.

The final experiment reduced this to:

```text
256 Tokens
```

This made training significantly more practical.

---

# Preprocessing Function

```python
def preprocess(text, label):
    token_ids = roberta_tokenizer(text)

    token_ids, padding_mask = packer(token_ids)

    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask
    }

    return inputs, label
```

---

# Applying Preprocessing

```python
preprocessed_train_ds = train_ds.map(preprocess)

preprocessed_val_ds = val_ds.map(preprocess)

preprocessed_test_ds = test_ds.map(preprocess)
```

Remember:

```text
preprocess()
→ Defines the transformation

map(preprocess)
→ Applies it to the dataset
```

---

# RoBERTa Classification Head

The Backbone returns:

```text
(batch_size, sequence_length, 768)
```

The first token representation was selected:

```python
x = x[:, 0, :]
```

Then a task-specific classifier was added.

```python
inputs = roberta_backbone.input

x = roberta_backbone(inputs)

x = x[:, 0, :]

x = layers.Dropout(0.1)(x)

x = layers.Dense(
    768,
    activation="relu"
)(x)

x = layers.Dropout(0.1)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

classifier = keras.Model(
    inputs,
    outputs
)
```

---

# Final RoBERTa Architecture

```text
token_ids + padding_mask
          ↓
Pretrained RoBERTa
          ↓
Contextual Representations
          ↓
First Token Representation
          ↓
Dropout
          ↓
Dense(768, ReLU)
          ↓
Dropout
          ↓
Dense(1, Sigmoid)
          ↓
Positive Probability
```

---

# Compile

```python
classifier.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=5e-5
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

# Why Small Learning Rate?

```text
5e-5 = 0.00005
```

RoBERTa already contains useful pretrained knowledge.

Fine-Tuning should modify the pretrained weights slowly.

```text
Pretrained Knowledge
↓
Small Weight Updates
↓
Task Specialization
```

---

# EarlyStopping

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)
```

Purpose:

- Stop unnecessary training.
- Reduce overfitting.
- Restore best weights.

---

# ModelCheckpoint

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    "/content/drive/MyDrive/binx_training/day5_week7/best_roberta_model.keras",
    monitor="val_loss",
    save_best_only=True
)
```

Purpose:

```text
Training
↓
Validation
↓
Best Model
↓
Save to Google Drive
```

This also protects the model if Colab disconnects.

---

# Fine-Tuning

```python
history = classifier.fit(
    preprocessed_train_ds,
    validation_data=preprocessed_val_ds,
    epochs=10,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)
```

The maximum is 10 epochs.

EarlyStopping can stop training sooner.

---

# What Happens During Fine-Tuning?

```text
Batch
↓
Forward Pass
↓
Prediction
↓
Binary Crossentropy
↓
Backpropagation
↓
Adam
↓
Small Weight Updates
↓
Next Batch
```

The major difference from training from scratch is:

```text
Weights start pretrained
instead of random
```

---

# Fine-Tuned RoBERTa Results

Final test results:

```text
Test Accuracy = 92.58%
Test Loss = 0.1886
F1-score = 0.93
```

---

# Classification Report

| Class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| Negative | 0.92 | 0.93 | 0.93 |
| Positive | 0.93 | 0.92 | 0.93 |

The model showed balanced performance across both sentiment classes.

---

# Sprint 2 Model Comparison

| Model | Test Accuracy | F1-Score | Test Loss |
|---|---:|---:|---:|
| LSTM | 89.76% | 0.90 | 0.2710 |
| Pretrained Transformer | ~89% | 0.89 | N/A |
| Fine-Tuned RoBERTa | **92.58%** | **0.93** | **0.1886** |

The Fine-Tuned RoBERTa achieved the strongest overall performance.

---

# What the Comparison Demonstrated

```text
LSTM
→ Learned IMDB from training data
→ 89.76%

Pretrained Transformer
→ General knowledge
→ No project-specific Fine-Tuning
→ ~89%

Fine-Tuned RoBERTa
→ General language knowledge
+
IMDB-specific Fine-Tuning
→ 92.58%
```

This demonstrated the practical value of Fine-Tuning.

---

# Sprint 2 Review

## Sprint Goal

Develop a stronger core architecture for the project and improve model performance over the previous baseline.

---

## Work Completed

- Studied CNN architecture.
- Learned image-specific convolution concepts.
- Learned Transfer Learning.
- Studied sequential architectures.
- Built SimpleRNN and LSTM models.
- Applied LSTM to ECG classification.
- Compared LSTM against SimpleRNN.
- Studied Attention.
- Studied Transformer architecture.
- Used a pretrained Hugging Face Transformer.
- Built an IMDB LSTM sentiment classifier.
- Fine-tuned pretrained RoBERTa.
- Used EarlyStopping.
- Used ModelCheckpoint.
- Evaluated multiple models.
- Compared model metrics.
- Documented results.

---

# Sprint Retrospective

## What Went Well

- LSTM significantly outperformed SimpleRNN on ECG data.
- The IMDB LSTM provided a strong text baseline.
- Pretrained Transformer inference was successfully tested.
- Fine-Tuning significantly improved Transformer performance.
- RoBERTa achieved balanced Precision, Recall, and F1.
- ModelCheckpoint protected the best trained model.
- Multiple architectures were compared using real metrics.

---

## What Could Be Improved

The initial RoBERTa configuration used:

```text
sequence_length = 512
```

which made Full Fine-Tuning extremely slow.

The model was later changed to:

```text
sequence_length = 256
```

which provided a much better computational tradeoff.

---

## Concrete Change for Sprint 3

Before starting long model training:

```text
Small Test Run
↓
Check GPU Memory
↓
Check Seconds per Batch
↓
Estimate Epoch Time
↓
Adjust Sequence Length
↓
Adjust Batch Size
↓
Start Full Training
```

This will prevent unnecessary long training runs.

---

# Week 7 Architecture Comparison

| Architecture | Best For | Main Advantage | Main Limitation |
|---|---|---|---|
| Dense Network | Tabular Data | Simple and effective | Does not preserve spatial/sequential structure |
| CNN | Images | Learns spatial features | Mainly designed for spatial data |
| SimpleRNN | Short Sequences | Simple recurrent memory | Weak long-term memory |
| LSTM | Sequential Data | Gated long-term memory | Sequential processing is slower |
| Transformer | Text / Long Sequences | Attention + parallel processing | Computationally expensive |
| Pretrained Transformer | NLP | Powerful pretrained knowledge | May need Fine-Tuning |
| Fine-Tuned Transformer | Specific NLP Task | Strong task-specific performance | Requires GPU and careful tuning |

---

# Important Concepts Learned During the Week

## CNN

```text
Image
↓
Filters
↓
Feature Maps
↓
Pooling
↓
Visual Features
↓
Classification
```

## LSTM

```text
Sequence
↓
Step-by-step processing
↓
Gated Memory
↓
Long-term information
↓
Prediction
```

## Transformer

```text
Tokens
↓
Self-Attention
↓
Direct contextual relationships
↓
Contextual Representations
↓
Prediction
```

## Transfer Learning

```text
Pretrained Knowledge
↓
Reuse
↓
New Task
```

## Fine-Tuning

```text
Pretrained Model
+
New Dataset
↓
Small Weight Updates
↓
Specialized Model
```

---

# The Most Important Architecture Rule

```text
Do not choose a model because it is more complicated.

Choose a model because its architecture matches the structure of the data.
```

Examples:

```text
Image
→ CNN

Text
→ LSTM / Transformer

Time Series
→ LSTM / GRU

Tabular
→ Dense Network / Classical ML
```

---

# The Complete NLP Workflow Learned This Week

The easiest way to remember the final Transformer workflow is:

```text
TEXT
↓
NUMBERS
↓
STRUCTURE
↓
UNDERSTANDING
↓
DECISION
↓
LEARNING
↓
EXAM
```

More specifically:

```text
Raw Text
↓
Tokenizer
↓
Token IDs
↓
StartEndPacker
↓
Padding + Padding Mask
↓
Pretrained RoBERTa Backbone
↓
Contextual Representation
↓
Classification Head
↓
Sigmoid
↓
Fine-Tuning
↓
Validation
↓
Best Model
↓
Test Evaluation
```

---

# Evaluation Metrics Used

## Accuracy

Percentage of predictions that were correct.

```text
Correct Predictions
-------------------
Total Predictions
```

## Precision

Of all samples predicted as a class, how many were actually that class?

## Recall

Of all real samples belonging to a class, how many did the model successfully find?

## F1-Score

Balances Precision and Recall.

## Loss

Measures how wrong the model predictions are and is used during optimization.

---

# Callbacks Used

## EarlyStopping

```text
Validation stops improving
↓
Stop Training
↓
Restore Best Weights
```

## ModelCheckpoint

```text
Best Validation Result
↓
Save Model
```

---

# Tools Used During Week 7

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- KerasHub
- Hugging Face Transformers
- Kaggle
- Jupyter Notebook
- Google Colab
- GPU
- Google Drive
- Git
- GitHub

---

# Repository Workflow

```text
Write Code
↓
Run Notebook
↓
Document Results
↓
git add
↓
git commit
↓
git push
↓
Pull Request
↓
Mentor Review
↓
Merge
```

Example commit:

```bash
git add .
git commit -m "feat: add RoBERTa fine-tuning, evaluation, and model comparison"
git push
```

---

# Final Week Results

## ECG Sequence Classification

```text
SimpleRNN
Accuracy = 82.97%

LSTM
Accuracy = 97.54%
```

The LSTM was substantially better at learning ECG sequence patterns.

---

## IMDB Sentiment Classification

```text
LSTM
Accuracy = 89.76%
F1 = 0.90
Loss = 0.2710
```

```text
Pretrained Transformer
Accuracy ≈ 89%
F1 = 0.89
```

```text
Fine-Tuned RoBERTa
Accuracy = 92.58%
F1 = 0.93
Loss = 0.1886
```

The Fine-Tuned RoBERTa achieved the strongest sentiment-classification performance.

---

# Final Week Conclusion

Week 7 connected several major Deep Learning architectures to the type of data they are designed to process.

CNNs demonstrated how neural networks can preserve spatial structure and automatically learn image features.

RNNs introduced sequential memory, while LSTMs showed how gated memory improves learning across longer sequences.

Attention and Transformers replaced step-by-step recurrent memory with direct contextual relationships between sequence elements.

Finally, pretrained models and Fine-Tuning demonstrated how previously learned knowledge can be reused and adapted to a specific task.

The practical results reinforced the main lesson of the week:

```text
Data Structure
↓
Choose Suitable Architecture
↓
Build Baseline
↓
Evaluate
↓
Use Transfer Learning when appropriate
↓
Fine-Tune
↓
Compare Results
↓
Select Best Model
```

The final Fine-Tuned RoBERTa model achieved the strongest sentiment-classification performance, demonstrating the effectiveness of combining pretrained language representations with task-specific Fine-Tuning.