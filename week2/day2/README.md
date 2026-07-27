# Probability & Distributions with Python

This project covers the fundamentals of **Probability and Probability Distributions** using Python, NumPy, Matplotlib, and Jupyter Notebook.

The goal of this training is to understand probability concepts and apply them through practical simulations, especially in **Data Science and Machine Learning**.

---

# Learning Objectives

By the end of this lesson, I learned how to:

- Apply probability rules:
  - Complement Rule
  - Addition Rule
  - Multiplication Rule

- Understand and calculate:
  - Conditional Probability
  - Bayes' Theorem

- Recognize common probability distributions:
  - Normal Distribution
  - Binomial Distribution
  - Uniform Distribution

- Use Python libraries to simulate probability experiments and visualize results.

---

# Tools Used

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

# 1. Probability Basics

Probability measures uncertainty and describes how likely an event is to happen.

The probability value is between:

- **0** → Impossible event
- **1** → Certain event

## Probability Formula

Probability = Number of favorable outcomes / Total possible outcomes

### Example:

Rolling a fair dice:

The probability of getting number 6:

Probability(6) = 1 / 6

---

# 2. Core Probability Rules

## Complement Rule

The complement rule calculates the probability that an event does not happen.

Formula:

Probability(Not A) = 1 - Probability(A)

### Example:

If:

Probability(Pass) = 0.8

Then:

Probability(Fail) = 1 - 0.8 = 0.2

---

## Addition Rule

The addition rule is used when we want the probability of either event happening.

Formula:

Probability(A or B) = Probability(A) + Probability(B) - Probability(A and B)

The intersection is subtracted because overlapping events are counted twice.

---

## Multiplication Rule

The multiplication rule calculates the probability of two events happening together.

For independent events:

Probability(A and B) = Probability(A) × Probability(B)

### Example:

The probability of getting heads twice in two coin flips:

0.5 × 0.5 = 0.25

---

# 3. Conditional Probability

Conditional probability calculates the probability of an event happening given that another event has already happened.

Formula:

Probability(A given B) = Probability(A and B) / Probability(B)

Meaning:

The probability of event A happening when event B is already known to be true.

---

## Machine Learning Example

Predicting customer churn:

"Given customer features, what is the probability that this customer will leave?"

This is a conditional probability problem.

Machine Learning models often use conditional probability to make predictions based on available features.

---

# 4. Bayes' Theorem

Bayes' theorem updates probability when new evidence becomes available.

Formula:

Probability(A given B) = (Probability(B given A) × Probability(A)) / Probability(B)

---

## Components of Bayes' Theorem

### Prior Probability

The initial belief before seeing new evidence.

Prior = Probability(A)

---

### Likelihood

How likely the evidence is if the event is true.

Likelihood = Probability(B given A)

---

### Posterior Probability

The updated probability after considering new evidence.

Posterior = Probability(A given B)

---

## Applications in Machine Learning

Bayes' theorem is used in:

- Naive Bayes Classifier
- Spam Detection
- Medical Diagnosis
- Text Classification

---

# 5. Probability Distributions

A probability distribution describes how likely different values of a variable are.

---

# Normal Distribution (Gaussian Distribution)

A symmetric distribution with a bell-shaped curve.

## Characteristics:

- Most values are close to the mean.
- Values become less common as they move away from the mean.

## Examples:

- Human heights
- Measurement errors
- Natural data patterns

---

# Binomial Distribution

A distribution that describes the number of successes in a fixed number of yes/no trials.

## Examples:

- Number of heads in 10 coin flips.
- Number of successful predictions.

## Conditions:

- Fixed number of trials.
- Two possible outcomes.
- Same probability for each trial.

---

# Uniform Distribution

A distribution where every possible outcome has the same probability.

## Example:

Rolling a fair dice:

Probability(1) = Probability(2) = Probability(3) = ... = Probability(6)

Every result has the same chance.

---

# Hands-On Lab

## Step 1: Coin Flip Simulation

## Objective

Simulate 10,000 coin flips using NumPy and confirm that the proportion of heads approaches 0.5.

## Code

```python
import numpy as np

flips = np.random.choice(
    ["Heads", "Tails"],
    size=10000
)

heads = np.sum(flips == "Heads")

probability = heads / 10000

print(probability)
```

## Explanation

The result should be close to:

0.5

This demonstrates the **Law of Large Numbers**.

As the number of experiments increases, the experimental probability becomes closer to the theoretical probability.

---

# Step 2: Normal Distribution Simulation

## Objective

Generate random values using NumPy normal distribution and visualize the data using a histogram.

## Code

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(
    loc=0,
    scale=1,
    size=1000
)

plt.hist(data, bins=30)
plt.show()
```

## Explanation

The histogram should show a bell-shaped curve.

This confirms the main characteristics of a normal distribution, where most values are concentrated around the mean.

---

# Step 3: Conditional Probability Simulation

## Scenario

We have:

- 60 male students.
- 40 female students.
- 30 successful male students.
- 30 unsuccessful male students.

We want to calculate:

Probability(Pass given Male)

---

## Formula

Probability(Pass given Male)

= Probability(Pass and Male) / Probability(Male)

---

## Python Implementation

```python
import numpy as np

stud = np.array(
    ['male'] * 60 + ['female'] * 40
)

grades = np.array(
    ['yes'] * 30 +
    ['no'] * 30 +
    ['yes'] * 20 +
    ['no'] * 20
)

total = len(stud)

# Probability of Male

males = stud == 'male'

male_count = np.sum(males)

P_male = male_count / total


# Probability of Pass and Male

male_pass = np.sum(
    (stud == 'male') &
    (grades == 'yes')
)

P_pass_and_male = male_pass / total


# Conditional Probability

result = P_pass_and_male / P_male

print(result)
```

---

## Result

Output:

```
0.5
```

Explanation:

Probability(Pass given Male)

= (30 / 100) / (60 / 100)

= 30 / 60

= 0.5

The Python simulation confirms the manual calculation.

---

# Step 4: Documentation

Each experiment was documented using Markdown explanations.

The documentation includes:

- The objective of each experiment.
- The method used.
- The obtained results.
- The meaning of each result.

---

# Machine Learning Connection

Probability is a fundamental concept in Machine Learning.

Machine Learning models often output probabilities such as:

- Probability that a customer will leave.
- Probability that an image contains a specific object.
- Probability that an email is spam.

Probability distributions help in:

- Understanding data behavior.
- Choosing suitable statistical methods.
- Building better ML models.

---

# Summary

During this training, I learned:

✅ Probability basics  
✅ Complement Rule  
✅ Addition Rule  
✅ Multiplication Rule  
✅ Conditional Probability  
✅ Bayes' Theorem  
✅ Normal Distribution  
✅ Binomial Distribution  
✅ Uniform Distribution  
✅ Probability simulation using NumPy  
✅ Data visualization using Matplotlib  

This knowledge builds a strong foundation for Data Science and Machine Learning.