# Day 5 — Tuning, Evaluation & Sprint Review

## Overview

In this day, the Neural Network was systematically tuned, evaluated, and compared with the previous models.

The main goal was to improve the model by changing one hyperparameter at a time while monitoring validation performance.

The tuning process focused on:

- Learning Rate
- Network Size
- Dropout Rate
- Batch Size

EarlyStopping and ModelCheckpoint were also used to improve training efficiency and preserve the best model.

---

## Learning Objectives

By the end of this day, we were able to:

- Tune a Neural Network systematically.
- Change one hyperparameter at a time.
- Use EarlyStopping to stop training when validation performance stops improving.
- Use ModelCheckpoint to save the best model.
- Evaluate the final tuned Neural Network on the Test Set.
- Compare the tuned Neural Network with previous models.
- Prepare evidence for the Sprint Review.
- Complete the Sprint Retrospective.

---

# 1. Neural Network Tuning

The Neural Network was tuned one hyperparameter at a time.

This approach makes it easier to understand how each hyperparameter affects model performance.

The main validation metrics used during tuning were:

- Validation Loss
- Validation Recall
- Number of Epochs Trained

---

## Learning Rate Tuning

The following Learning Rates were tested:

| Learning Rate | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 0.0001 | **0.2587** | **0.7746** | 61 |
| 0.001 | 0.2596 | 0.7700 | 18 |
| 0.01 | 0.2700 | 0.7653 | 13 |

### Selected Learning Rate

The selected Learning Rate was:

`0.0001`

It achieved the lowest Validation Loss and the highest Validation Recall among the tested values.

---

## Network Size Tuning

After selecting the Learning Rate, different network sizes were tested.

| Network Size | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 16 → 8 | 0.2671 | 0.7559 | 146 |
| 32 → 16 | **0.2576** | **0.7793** | 103 |
| 64 → 32 | 0.2619 | 0.7676 | 70 |

### Selected Network Size

The selected architecture was:

`32 → 16`

It achieved the best Validation Loss and Validation Recall among the tested architectures.

---

## Dropout Rate Tuning

The selected Learning Rate and Network Size were kept fixed while different Dropout Rates were tested.

| Dropout Rate | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 0.1 | 0.2721 | 0.7582 | 72 |
| 0.3 | 0.2640 | 0.7606 | 87 |
| 0.5 | **0.2608** | **0.7770** | 111 |

### Selected Dropout Rate

The selected Dropout Rate was:

`0.5`

It achieved the lowest Validation Loss and the highest Validation Recall.

---

## Batch Size Tuning

Finally, different Batch Sizes were tested while keeping the other selected hyperparameters fixed.

| Batch Size | Best Validation Loss | Best Validation Recall | Epochs Trained |
|---|---:|---:|---:|
| 16 | 0.2650 | 0.7746 | 69 |
| 32 | **0.2617** | **0.7770** | 115 |
| 64 | 0.2635 | 0.7582 | 163 |

### Selected Batch Size

The selected Batch Size was:

`32`

It achieved the best validation performance among the tested values.

---

# 2. Final Selected Hyperparameters

The final tuned Neural Network used the following configuration:

| Hyperparameter | Selected Value |
|---|---:|
| Learning Rate | 0.0001 |
| Network Size | 32 → 16 |
| Dropout Rate | 0.5 |
| Batch Size | 32 |

The final architecture was:

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

The model used the Adam optimizer with:

```python
learning_rate = 0.0001
```

The loss function was:

```python
binary_crossentropy
```

The main evaluation metrics were:

```python
accuracy
recall
```

---

# 3. EarlyStopping and ModelCheckpoint

EarlyStopping was used to automatically stop training when the Validation Loss stopped improving.

```python
final_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

The `restore_best_weights=True` option restores the model weights from the best epoch.

ModelCheckpoint was also used to save the best model during training.

```python
final_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "final_tuned_model.keras",
    monitor="val_loss",
    save_best_only=True
)
```

This ensures that the best-performing version of the model is preserved even if later epochs perform worse.

---

# 4. Final Tuned Model Evaluation

After tuning, the final Neural Network was evaluated on the Test Set.

The results were:

| Metric | Result |
|---|---:|
| Test Loss | 0.2397 |
| Test Accuracy | 0.9122 |
| Test Recall | 0.8309 |

The final tuned Neural Network achieved strong performance on unseen data.

---

# 5. Model Performance Comparison

The final tuned Neural Network was compared with the previous Neural Network versions and the Random Forest baseline.

| Model | Test Loss | Accuracy | Recall |
|---|---:|---:|---:|
| Random Forest Baseline | — | 0.880 | **0.850** |
| Original Neural Network | 0.3300 | 0.881 | 0.790 |
| Regularized Neural Network | 0.2541 | 0.893 | 0.793 |
| Final Tuned Neural Network | **0.2397** | **0.912** | 0.831 |

The Final Tuned Neural Network achieved the highest Accuracy.

However, Random Forest still achieved the highest Recall.

Because Recall is important for identifying positive Heart Disease cases in this project, Random Forest remains slightly stronger according to Recall.

---

# 6. Training Curves

Three training curves were used to inspect model behavior:

- Loss Curve
- Accuracy Curve
- Recall Curve

---

## Loss Curve Interpretation

Both Training Loss and Validation Loss decreased steadily during training.

The Validation Loss remained stable and did not begin increasing, which indicates that the final tuned model controlled overfitting effectively.

The Validation Loss was slightly lower than the Training Loss for most epochs.

This can occur because Dropout is active during training but disabled during validation.

Overall, the loss curves show stable learning and good generalization.

---

## Accuracy Curve Interpretation

Both Training Accuracy and Validation Accuracy increased steadily during training.

Validation Accuracy reached approximately `0.89–0.90` and remained stable during the later epochs.

The gap between Training Accuracy and Validation Accuracy remained small.

This indicates that the model generalized well without severe overfitting.

Validation Accuracy was slightly higher than Training Accuracy for most epochs because Dropout is active during training but disabled during validation.

Overall, the Accuracy curves indicate stable and effective learning.

---

## Recall Curve Interpretation

Training Recall improved gradually during training and reached approximately `0.78`.

Validation Recall initially fluctuated, then became more stable and gradually improved to approximately `0.77`.

Toward the end of training, Training Recall and Validation Recall became very close.

This indicates that the model generalized well without a large gap between training and validation performance.

Overall, the Recall curve shows that the tuned Neural Network became better at identifying positive Heart Disease cases while maintaining stable validation performance.

---

# 7. Sprint Review

During this sprint, the Neural Network was:

- Built
- Regularized
- Tuned
- Evaluated
- Compared with the classical Machine Learning baseline

The tuning process improved the Neural Network compared with both the original and regularized versions.

The Final Tuned Neural Network achieved:

- Test Loss: `0.2397`
- Test Accuracy: `0.9122`
- Test Recall: `0.8309`

The tuned Neural Network achieved the highest Accuracy among the evaluated models.

However, the Random Forest baseline still achieved a slightly higher Recall of `0.85`.

---

# 8. Sprint Retrospective

## What Went Well

The Neural Network improved significantly after systematic tuning.

Using EarlyStopping and ModelCheckpoint made the training process more efficient and helped preserve the best model.

Changing one hyperparameter at a time made it easier to understand the effect of each tuning decision.

---

## What Could Be Improved

Some experiments required many epochs before EarlyStopping stopped the training.

The Neural Network still did not exceed the Random Forest baseline in Recall.

---

## Concrete Change for the Next Sprint

In the next sprint, model improvements will focus more directly on Recall and reducing False Negatives while maintaining strong overall performance.

---

# Final Conclusion

Systematic tuning improved the Neural Network considerably.

The final model achieved:

- Accuracy: **91.22%**
- Recall: **83.09%**
- Loss: **0.2397**

Compared with the previous Neural Networks, the tuned model achieved better Accuracy, Recall, and Loss.

However, Random Forest still achieved a slightly better Recall of **85%**.

Therefore, the tuned Neural Network provides the strongest overall Accuracy, while Random Forest remains the preferred model when Recall is the highest priority.