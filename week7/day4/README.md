# Day 4 — Attention & Transformers

## Overview

Day 4 introduces **Attention** and **Transformer architectures**.

RNNs and LSTMs can model sequences, but they process them step-by-step.

Transformers remove recurrence and allow sequence elements to interact directly using Attention.

This makes Transformers powerful for:

- Text classification
- Sentiment analysis
- Question answering
- Translation
- Summarization
- Text generation

The practical lab used a pretrained Hugging Face Transformer for sentiment analysis and compared it with an LSTM model.

---

# Learning Objectives

By the end of this day, I should be able to:

- Explain the limitations of RNNs and LSTMs.
- Explain Attention.
- Understand Self-Attention.
- Understand Query, Key, and Value conceptually.
- Explain why Transformers can process sequences in parallel.
- Explain Positional Encoding.
- Understand Encoder and Decoder architectures.
- Understand pretrained Transformers.
- Distinguish BERT, DistilBERT, and GPT-style models.
- Use a Hugging Face Pipeline.
- Compare LSTM with a pretrained Transformer.

---

# 1. Limitation of RNNs

RNNs and LSTMs process sequences sequentially.

```text
Token 1
↓
Token 2
↓
Token 3
↓
Token 4
```

Token 4 cannot be processed until the previous steps have been processed.

This creates two main problems.

## Problem 1 — Limited Parallelism

The sequence must be processed step-by-step.

This makes training slower.

## Problem 2 — Long-Range Dependencies

Even LSTMs can struggle when important information is extremely far away in a long sequence.

---

# 2. Attention

Attention solves this problem differently.

Instead of passing information only through previous states, every token can directly examine other tokens.

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

and understand that the phrase has negative meaning.

---

# 3. Self-Attention

**Self-Attention** means each token examines all tokens in the same sequence.

Conceptually:

```text
Token 1 ─┬→ Token 1
         ├→ Token 2
         ├→ Token 3
         └→ Token 4

Token 2 ─┬→ Token 1
         ├→ Token 2
         ├→ Token 3
         └→ Token 4
```

Each connection receives an Attention Weight indicating relevance.

---

# Attention Weights

Attention does not treat every word equally.

For a particular token, the model may assign:

```text
Very relevant word → High weight
Less relevant word → Low weight
```

The resulting representation is a weighted combination of contextual information.

---

# 4. Query, Key, and Value

Self-Attention commonly uses three representations:

- Query
- Key
- Value

A simple way to remember them:

```text
Query
→ What am I looking for?

Key
→ What information do I represent?

Value
→ What actual information should I provide?
```

The Query of one token is compared with the Keys of other tokens.

This comparison determines Attention Scores.

Then the scores are used to combine the Values.

---

# Attention Formula

Conceptually:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dk)V
```

The important idea is:

```text
Compare Query and Keys
↓
Calculate relevance scores
↓
Convert to Attention weights
↓
Use weights to combine Values
```

Understanding the concept is more important here than manually implementing the mathematics.

---

# 5. Why Attention Is Powerful

## Direct Long-Range Access

A token can directly access another distant token.

It does not need to pass information through hundreds of recurrent steps.

## Parallel Processing

Transformers can process sequence positions in parallel.

RNN:

```text
1 → 2 → 3 → 4
```

Transformer:

```text
1
2
3
4
↓
Processed together
```

This allows much more efficient training on modern hardware.

---

# 6. Multi-Head Attention

Instead of using one Attention operation, Transformers use multiple Attention Heads.

Different heads may learn different relationships.

Conceptually:

```text
Head 1 → local word relationships
Head 2 → long-distance relationships
Head 3 → grammatical relationships
Head 4 → semantic relationships
```

These relationships are learned automatically.

---

# 7. Transformer Architecture

A Transformer contains repeated blocks with components such as:

```text
Input
↓
Embedding
+
Positional Information
↓
Self-Attention
↓
Feed Forward Network
↓
Normalization / Residual Connections
↓
Next Transformer Layer
```

---

# 8. Positional Encoding

Transformers process positions in parallel.

Therefore, unlike an RNN, they do not automatically know the order.

For example:

```text
dog bites man
```

and:

```text
man bites dog
```

contain the same words but have different meaning.

Position information must therefore be added explicitly.

Conceptually:

```text
Token Embedding
+
Position Information
=
Transformer Input
```

---

# 9. Encoder vs Decoder

## Encoder

The Encoder is primarily designed to understand input.

```text
Text
↓
Encoder
↓
Contextual Representations
```

Good for:

- Classification
- Sentiment Analysis
- Information Extraction
- Some Question Answering tasks

Examples:

- BERT
- DistilBERT
- RoBERTa

---

## Decoder

The Decoder is designed to generate tokens.

```text
Previous Tokens
↓
Decoder
↓
Probability of next token
↓
Generate token
↓
Repeat
```

GPT-style models are Decoder-based.

---

# Encoder vs Decoder Summary

| Encoder | Decoder |
|---|---|
| Understands input | Generates output |
| Sees contextual input | Predicts next tokens |
| Classification | Text generation |
| BERT / RoBERTa | GPT |

---

# 10. Pretrained Transformers

Training a Transformer from scratch requires enormous amounts of:

- Text
- Compute
- GPUs
- Training time

Instead, modern NLP commonly uses:

**Pretrained Transformers**

The model first learns general language representations from huge text collections.

Then it can be reused for downstream tasks.

This follows the same general idea as Transfer Learning in CNNs.

---

# BERT

BERT is an Encoder-based Transformer designed for language understanding.

Common tasks:

- Sentiment Analysis
- Classification
- Question Answering
- Named Entity Recognition

---

# DistilBERT

DistilBERT is a smaller, faster version inspired by BERT.

Advantages:

- Fewer parameters.
- Faster inference.
- Lower memory usage.

It is useful when computational resources are limited.

---

# GPT

GPT models are Decoder-based Transformers.

Their main strength is:

```text
Text Generation
```

They repeatedly predict the next token.

---

# 11. Hugging Face Pipeline

The Hugging Face `pipeline()` function provides an easy way to use pretrained models.

Example:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)
```

The pipeline automatically handles:

- Model loading.
- Tokenizer loading.
- Preprocessing.
- Model inference.
- Output formatting.

---

# Sample Sentiment Analysis

```python
texts = [
    "This course is amazing!",
    "I really hate this movie.",
    "The project was difficult but very useful."
]

results = classifier(texts)

print(results)
```

Example results from the experiment:

```text
"This course is amazing!"
→ POSITIVE
→ ~0.9999

"I really hate this movie."
→ NEGATIVE
→ ~0.9996

"The project was difficult but very useful."
→ POSITIVE
→ ~0.9919
```

---

# 12. Applying the Pretrained Transformer to IMDB

The pretrained Transformer was later evaluated on the same IMDB test data used for the LSTM comparison.

```python
transformer_results = classifier(
    X_test.tolist(),
    batch_size=32,
    truncation=True
)
```

---

# batch_size

```python
batch_size=32
```

does not mean only 32 reviews are evaluated.

All reviews are supplied.

The model internally processes them in groups of 32.

```text
5000 reviews
↓
32 at a time
↓
until all reviews are processed
```

---

# truncation=True

Transformers have a maximum supported sequence length.

If a review is too long:

```python
truncation=True
```

cuts the extra tokens rather than producing an error.

---

# Convert Labels to Numeric Values

The pipeline returned:

```text
POSITIVE
NEGATIVE
```

The project labels were:

```text
1
0
```

Therefore the labels were converted:

```python
transformer_pred = [
    1 if result["label"] == "POSITIVE" else 0
    for result in transformer_results
]
```

---

# Pretrained Transformer Result

The Transformer used without task-specific fine-tuning achieved approximately:

```text
Accuracy ≈ 89%
F1-score ≈ 0.89
```

Approximate classification results:

| Class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| Negative | 0.86 | 0.93 | 0.89 |
| Positive | 0.92 | 0.85 | 0.89 |

---

# Why Was the Pretrained Transformer Not Better Than the LSTM?

The pretrained pipeline was not specifically fine-tuned on our IMDB training split.

The LSTM, however, was trained directly using IMDB reviews.

Therefore the comparison was:

```text
LSTM
→ Specifically trained on our IMDB data

Pretrained Transformer
→ General pretrained sentiment model
→ No project-specific fine-tuning
```

A pretrained model is powerful, but project-specific Fine-Tuning can further adapt it to the target dataset.

This became an important motivation for the Day 5 experiment.

---

# RNN vs Attention

## RNN

```text
Token 1
↓
Memory
↓
Token 2
↓
Memory
↓
Token 3
```

Information moves step-by-step.

## Attention

```text
Every Token
↕
Every Other Token
```

Tokens can directly examine each other.

---

# Main Difference

```text
RNN
→ Step-by-step memory

Transformer
→ Direct weighted relationships between sequence positions
```

---

# Hands-On Lab

## Step 1

Load a pretrained Transformer using Hugging Face and test it on sample text.

## Step 2

Apply it to project data and compare its metric with an LSTM.

## Step 3

Explain how Attention differs from RNN memory.

## Step 4

Select which architecture should serve as the project's core model and explain why.

---

# Key Concepts Summary

| Concept | Meaning |
|---|---|
| Attention | Determines which information is most relevant |
| Self-Attention | Tokens attend to other tokens in the same sequence |
| Query | What information a token is looking for |
| Key | What a token represents for comparison |
| Value | Actual information contributed |
| Multi-Head Attention | Multiple learned attention relationships |
| Positional Encoding | Gives Transformer information about token order |
| Encoder | Primarily understands input |
| Decoder | Primarily generates output |
| Pretrained Transformer | Transformer already trained on large text datasets |
| Pipeline | High-level Hugging Face interface |

---



# Main Conclusion

Day 4 demonstrated the evolution:

```text
RNN
↓
Step-by-step sequence processing

LSTM
↓
Better memory

Transformer
↓
Attention
↓
Direct long-range relationships
↓
Parallel processing
```

The most important practical lesson was that using a pretrained Transformer is easy, but **Fine-Tuning is necessary when we want the model to adapt specifically to our own task and dataset**.