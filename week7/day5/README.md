# Day 5 — Advancing the Core Model & Sprint Review

## Overview

Day 5 closes Sprint 2.

The main goal is to select the architecture that best matches the project data, improve the core model, evaluate the results, compare experiments, document the work, and complete the Sprint Review and Retrospective.

For the text-classification experiment, three approaches were compared:

1. LSTM trained on IMDB.
2. Pretrained Transformer used without task-specific Fine-Tuning.
3. Pretrained RoBERTa with project-specific Fine-Tuning.

The final Fine-Tuned RoBERTa achieved the strongest performance.

---

# Learning Objectives

By the end of this day, I should be able to:

- Match model architecture to the data type.
- Select an appropriate core model.
- Use a pretrained model.
- Explain Transfer Learning and Fine-Tuning.
- Fine-tune a pretrained Transformer.
- Use a small Learning Rate during Fine-Tuning.
- Use EarlyStopping.
- Use ModelCheckpoint.
- Evaluate models consistently.
- Compare Accuracy, Loss, Precision, Recall, and F1-score.
- Log model experiments.
- Complete Sprint Review.
- Write a Retrospective.
- Define one concrete improvement for Sprint 3.

---

# Step 1 — Confirm and Justify the Core Architecture

The project uses text data for binary sentiment classification on the IMDB movie-review dataset.

Possible architectures include:

| Data Type | Suitable Architecture |
|---|---|
| Tabular | Dense Network / Gradient Boosting |
| Images | CNN + Transfer Learning |
| Text | LSTM / Pretrained Transformer |
| Time Series | LSTM / GRU |

For the improved Sprint 2 model, a Transformer-based architecture was selected.

---

# Why RoBERTa?

RoBERTa is:

- A pretrained Transformer.
- Encoder-based.
- Designed for language understanding.
- Suitable for classification.
- Able to model long-range contextual relationships.

The project task is:

```text
Movie Review
↓
Understand context
↓
Negative / Positive
```

Therefore an Encoder-based Transformer is a suitable architecture.

---

# Transfer Learning

A pretrained model already contains knowledge learned from a large dataset.

Instead of starting with random weights:

```text
Random Weights
↓
Learn everything
```

Transfer Learning starts with:

```text
Pretrained Weights
↓
Reuse existing knowledge
```

---

# Fine-Tuning

Fine-Tuning means:

```text
Pretrained Model
+
Our Dataset
↓
Small updates to pretrained weights
↓
Model becomes specialized for our task
```

For this project:

```text
Pretrained RoBERTa
+
IMDB Reviews
↓
Fine-Tuning
↓
IMDB Sentiment Classifier
```

---

# IMDB Dataset

The dataset contains:

```text
50,000 movie reviews
```

with two sentiment classes:

```text
Negative
Positive
```

The classes are balanced approximately 50/50.

---

# Encode Target

```python
df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})
```

Therefore:

```text
0 = Negative
1 = Positive
```

---

# Train / Validation / Test Split

The dataset was divided into:

```text
40,000 Training Reviews
5,000 Validation Reviews
5,000 Test Reviews
```

Using stratification preserved the class distribution.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
```

---

# Baseline LSTM

Before Fine-Tuning RoBERTa, an LSTM sentiment classifier was trained.

The text was converted into integer sequences using:

```python
tf.keras.layers.TextVectorization
```

Example:

```python
text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=400
)
```

---

# LSTM Architecture

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

# LSTM Results

```text
Test Accuracy = 89.76%
Test Loss = 0.2710
F1-score ≈ 0.90
```

Classification performance was balanced across Positive and Negative classes.

---

# Pretrained Transformer Without Fine-Tuning

A pretrained sentiment Transformer was then tested using Hugging Face:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    device=0
)
```

It achieved approximately:

```text
Accuracy ≈ 89%
F1-score ≈ 0.89
```

This model was not specifically trained on our IMDB training split.

---

# Fine-Tuned RoBERTa

To perform real Fine-Tuning, the project used:

```text
Keras
+
KerasHub
+
RoBERTa
```

The pretrained model:

```text
roberta_base_en
```

was loaded.

---

# Load RoBERTa Tokenizer

```python
import keras_hub

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

The tokenizer must match the pretrained model because RoBERTa was pretrained using its own vocabulary and token representation.

---

# Load RoBERTa Backbone

```python
roberta_backbone = keras_hub.models.Backbone.from_preset(
    "roberta_base_en"
)
```

The Backbone is the pretrained Transformer without the final project-specific classifier.

RoBERTa Base contained approximately:

```text
124 million parameters
```

and 12 Transformer Encoder layers.

---

# RoBERTa Input

The backbone expects:

```text
token_ids
padding_mask
```

`token_ids` represent the actual tokens.

`padding_mask` tells the Transformer which positions are real and which positions are padding.

---

# TensorFlow Dataset

The raw Pandas data was organized using:

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

This pairs:

```text
Review 1 ↔ Label 1
Review 2 ↔ Label 2
...
```

---

# Batching

The dataset is divided into Batches.

```text
Entire Dataset
↓
Batch 1
Batch 2
Batch 3
...
```

Batching allows the large Transformer to process a manageable number of reviews at a time.

---

# StartEndPacker

RoBERTa expects special sequence formatting.

The `StartEndPacker`:

- Adds the start token.
- Adds the end token.
- Adds padding for short sequences.
- Truncates long sequences.
- Creates the Padding Mask.

```python
sequence_length = 256

packer = keras_hub.layers.StartEndPacker(
    sequence_length=sequence_length,
    start_value=roberta_tokenizer.start_token_id,
    end_value=roberta_tokenizer.end_token_id,
    pad_value=roberta_tokenizer.pad_token_id,
    return_padding_mask=True
)
```

---

# Why Sequence Length Was Reduced

An initial experiment used:

```text
sequence_length = 512
```

Training became extremely slow.

Transformers use Self-Attention, whose computational cost increases rapidly as sequence length increases.

The final experiment used:

```text
sequence_length = 256
```

This significantly reduced training cost while still maintaining strong model performance.

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

The function performs:

```text
Raw Text
↓
Tokenizer
↓
Token IDs
↓
Packer
↓
token_ids + padding_mask
```

The label is returned unchanged.

---

# Apply Preprocessing

```python
preprocessed_train_ds = train_ds.map(preprocess)

preprocessed_val_ds = val_ds.map(preprocess)

preprocessed_test_ds = test_ds.map(preprocess)
```

Important concept:

```text
preprocess()
→ Defines what transformation should happen

map(preprocess)
→ Applies the transformation to the dataset
```

---

# Improve the Input Pipeline

The TensorFlow data pipeline can also use:

```python
num_parallel_calls=tf.data.AUTOTUNE
```

and:

```python
.prefetch(tf.data.AUTOTUNE)
```

Example:

```python
preprocessed_train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
```

This allows preprocessing and model training to overlap more efficiently.

---

# RoBERTa Classification Head

The pretrained RoBERTa Backbone returns contextual representations for every token.

Its output has approximately this structure:

```text
(batch_size, sequence_length, 768)
```

The first token representation is selected:

```python
x = x[:, 0, :]
```

This changes:

```text
(batch, sequence_length, 768)
```

into:

```text
(batch, 768)
```

providing one vector for each review.

---

# Final Architecture

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

# Architecture Flow

```text
token_ids + padding_mask
          ↓
Pretrained RoBERTa
          ↓
Contextual Token Representations
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

# Model Size

The final model contained approximately:

```text
124.6 million parameters
```

All parameters were trainable during Full Fine-Tuning.

This means both:

```text
RoBERTa pretrained weights
+
new classification head
```

could be updated during training.

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

# Why Learning Rate = 5e-5?

```text
5e-5 = 0.00005
```

Fine-Tuning uses a small Learning Rate because the pretrained weights already contain useful language knowledge.

Large updates could destroy this useful information.

Therefore:

```text
Training From Scratch
→ Can require larger updates

Fine-Tuning
→ Small controlled updates
```

---

# Binary Crossentropy

The task is:

```text
Negative = 0
Positive = 1
```

and the output is:

```python
Dense(
    1,
    activation="sigmoid"
)
```

Therefore:

```python
loss="binary_crossentropy"
```

is appropriate.

---

# EarlyStopping

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)
```

It:

- Monitors Validation Loss.
- Stops unnecessary training.
- Helps reduce overfitting.
- Restores the best model weights.

---

# ModelCheckpoint

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    "/content/drive/MyDrive/binx_training/day5_week7/best_roberta_model.keras",
    monitor="val_loss",
    save_best_only=True
)
```

The checkpoint preserves the best model even if:

- Colab disconnects.
- Later epochs perform worse.
- Training is interrupted.

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

`epochs=10` is the maximum.

EarlyStopping can stop training earlier if Validation Loss stops improving.

---

# Fine-Tuning Process

Inside every Batch:

```text
Forward Pass
↓
Prediction
↓
Binary Crossentropy Loss
↓
Backpropagation
↓
Adam
↓
Small Weight Update
↓
Next Batch
```

The important difference from normal training is:

```text
Weights start pretrained
instead of random
```

---

# Final RoBERTa Evaluation

The best saved model was evaluated using:

```python
test_loss, test_accuracy = classifier.evaluate(
    preprocessed_test_ds
)
```

Final result:

```text
Test Accuracy = 92.58%
Test Loss = 0.1886
```

---

# Predictions

```python
roberta_pred_prob = classifier.predict(
    preprocessed_test_ds
)
```

Because the final layer uses Sigmoid, predictions are probabilities.

Example:

```text
0.95
0.04
0.72
```

---

# Convert Probability to Class

```python
roberta_pred = (
    roberta_pred_prob >= 0.5
).astype(int).flatten()
```

Rule:

```text
Probability < 0.5
→ 0
→ Negative

Probability >= 0.5
→ 1
→ Positive
```

---

# RoBERTa Classification Report

Final approximate report:

| Class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| Negative | 0.92 | 0.93 | 0.93 |
| Positive | 0.93 | 0.92 | 0.93 |

Overall:

```text
Accuracy ≈ 0.93
Macro F1 ≈ 0.93
Weighted F1 ≈ 0.93
```

The model performed consistently across both classes.

---

# Step 3 — Model Comparison

| Model | Test Accuracy | F1-Score | Test Loss | Approach |
|---|---:|---:|---:|---|
| LSTM | 89.76% | 0.90 | 0.2710 | Trained on IMDB |
| Pretrained Transformer | ~89% | 0.89 | N/A | No project-specific Fine-Tuning |
| Fine-Tuned RoBERTa | **92.58%** | **0.93** | **0.1886** | Fine-Tuned on IMDB |

The Fine-Tuned RoBERTa model achieved the strongest overall performance.

---

# Why Fine-Tuning Improved the Model

The pretrained Transformer already understood general language patterns.

But before Fine-Tuning:

```text
General Language Knowledge
```

After Fine-Tuning on IMDB:

```text
General Language Knowledge
+
IMDB-specific sentiment patterns
```

Therefore the model became better adapted to the target task.

---

# Experiment Summary

| Experiment | Main Observation |
|---|---|
| LSTM | Strong sequential baseline |
| Pretrained Transformer | Good general performance but not specialized |
| Fine-Tuned RoBERTa | Best overall performance |
| Sequence Length 512 | Computationally expensive |
| Sequence Length 256 | Much more practical while maintaining strong performance |

---

# Step 4 — Documentation and Git Workflow

Sprint 2 work should be:

```text
Notebook completed
↓
Markdown documentation added
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

# Step 5 — Sprint Review

## Sprint Goal

Improve the text-classification core model using a stronger architecture and Transfer Learning.

## Work Completed

- Built an LSTM sentiment baseline.
- Evaluated a pretrained Transformer.
- Learned the difference between inference and Fine-Tuning.
- Loaded a pretrained RoBERTa model.
- Prepared RoBERTa tokenization and padding.
- Built a custom Classification Head.
- Fine-Tuned RoBERTa on IMDB.
- Used EarlyStopping.
- Saved the best model using ModelCheckpoint.
- Evaluated the final model.
- Compared all model results.

---

# Sprint Review Result

The strongest model was:

```text
Fine-Tuned RoBERTa
```

with:

```text
Accuracy = 92.58%
F1 = 0.93
Test Loss = 0.1886
```

This was an improvement over the LSTM baseline.

---

# Retrospective

## What Went Well

- The LSTM provided a strong baseline.
- RoBERTa Fine-Tuning improved final model performance.
- The classes remained balanced in Precision, Recall, and F1.
- ModelCheckpoint protected the trained model.
- EarlyStopping helped control unnecessary training.
- The final experiment clearly demonstrated the value of Transfer Learning.

---

# What Could Be Improved

The initial Transformer experiment used:

```text
sequence_length = 512
```

This made Full Fine-Tuning extremely slow.

The sequence length was later reduced to:

```text
256
```

which made training significantly more practical.

---

# Concrete Change for Sprint 3

Before starting a long training run:

1. Run a small experiment first.
2. Measure time per Batch.
3. Estimate Epoch duration.
4. Check GPU memory.
5. Select an appropriate Sequence Length.
6. Select an appropriate Batch Size.
7. Only then start Full Training.

This avoids wasting time on unnecessarily expensive configurations.

---

# The Entire Fine-Tuning Workflow

The easiest way to remember the complete process is:

```text
TEXT
↓
TOKEN IDS
↓
PACKING
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
Raw IMDB Review
      ↓
RoBERTa Tokenizer
      ↓
Token IDs
      ↓
StartEndPacker
      ↓
Start / End / Padding / Padding Mask
      ↓
RoBERTa Backbone
      ↓
Contextual Representation
      ↓
Classification Head
      ↓
Sigmoid
      ↓
Positive Probability
      ↓
Fine-Tuning
      ↓
Test Evaluation
```

---

# Architecture Selection Summary

| Data | Recommended Core Model |
|---|---|
| Tabular | Dense Network / Classical ML |
| Image | CNN / Transfer Learning |
| Text | LSTM / Transformer |
| Time Series | LSTM / GRU |

The most complex model is not always the best choice.

The architecture should match the structure of the data.

---


# Final Results

```text
LSTM
Accuracy = 89.76%
F1 = 0.90

Pretrained Transformer
Accuracy ≈ 89%
F1 = 0.89

Fine-Tuned RoBERTa
Accuracy = 92.58%
F1 = 0.93
Loss = 0.1886
```

---

# Final Conclusion

Sprint 2 demonstrated the importance of choosing an architecture that matches the data and the importance of Transfer Learning.

The LSTM successfully modeled text as a sequence, but the Fine-Tuned RoBERTa Transformer achieved stronger overall performance.

The key lesson is:

```text
Pretrained Model
→ already knows general patterns

Fine-Tuning
→ adapts that knowledge to our specific task

Evaluation
→ verifies whether the adaptation actually improved performance
```

Fine-Tuning RoBERTa on the IMDB dataset increased the final test performance and produced the strongest model of the Sprint.