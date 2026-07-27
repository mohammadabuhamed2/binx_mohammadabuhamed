# Probability & Distributions with Python

This project covers the fundamentals of **Probability and Probability Distributions** using Python, NumPy, Matplotlib, and Jupyter Notebook.

The goal of this training is to understand probability concepts and apply them in practical simulations, especially in the context of **Data Science and Machine Learning**.

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

When all outcomes are equally likely:

\[
P(A)=\frac{\text{Number of favorable outcomes}}{\text{Total possible outcomes}}
\]

## Example

Rolling a fair dice:

Probability of getting number 6:

\[
P(6)=\frac{1}{6}
\]

---

# 2. Core Probability Rules

## Complement Rule

The complement rule calculates the probability that an event does not happen.

Formula:

\[
P(Not\ A)=1-P(A)
\]

Example:

If:

\[
P(Pass)=0.8
\]

Then:

\[
P(Fail)=1-0.8=0.2
\]

---

## Addition Rule

The addition rule is used when we want the probability of either event happening.

Formula:

\[
P(A\ or\ B)=P(A)+P(B)-P(A\ and\ B)
\]

The intersection is subtracted because it is counted twice.

---

## Multiplication Rule

The multiplication rule calculates the probability of two events happening together.

For independent events:

\[
P(A\ and\ B)=P(A)\times P(B)
\]

Example:

Probability of getting heads twice:

\[
0.5 \times 0.5 = 0.25
\]

---

# 3. Conditional Probability

Conditional probability calculates the probability of an event happening given that another event has already happened.

Formula:

\[
P(A|B)=\frac{P(A\ and\ B)}{P(B)}
\]

## Meaning

The probability of event A happening when event B is already known to be true.

## Machine Learning Example

Predicting customer churn:

> Given customer features, what is the probability that this customer will leave?

This is a conditional probability problem.

---

# 4. Bayes' Theorem

Bayes' theorem updates probability when new evidence becomes available.

Formula:

\[
P(A|B)=\frac{P(B|A)\times P(A)}{P(B)}
\]

## Components of Bayes' Theorem

### Prior Probability

The initial belief before observing new evidence.

\[
P(A)
\]

---

### Likelihood

The probability of observing evidence if the event is true.

\[
P(B|A)
\]

---

### Posterior Probability

The updated probability after considering new evidence.

\[
P(A|B)
\]

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

# Normal Distribution (Gaussian)

A symmetric distribution with a bell-shaped curve.

## Characteristics

- Most values are close to the mean.
- Values become less frequent as they move away from the mean.

## Examples

- Human heights
- Measurement errors
- Natural data patterns

---

# Binomial Distribution

A distribution that describes the number of successes in a fixed number of yes/no trials.

## Examples

- Number of heads in 10 coin flips
- Number of successful predictions

## Conditions

- Fixed number of trials
- Only two possible outcomes
- Same probability for each trial

---

# Uniform Distribution

A distribution where every possible outcome has the same probability.

## Example

Rolling a fair dice:

\[
P(1)=P(2)=P(3)=...=P(6)
\]

---

# Hands-On Lab

## Step 1: Coin Flip Simulation

## Objective

Simulate 10,000 coin flips using NumPy and check whether the proportion of heads approaches 0.5.

## Implementation

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

## Result Explanation

The obtained probability should be close to:

\[
0.5
\]

This demonstrates the **Law of Large Numbers**, where increasing the number of experiments makes the experimental probability approach the theoretical probability.

---

# Step 2: Normal Distribution Simulation

## Objective

Generate random values from a normal distribution using:

```python
np.random.normal()
```

and visualize them using a histogram.

## Implementation

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

## Result Explanation

The histogram should have a bell-shaped curve.

This demonstrates the main characteristics of a normal distribution, where most values are concentrated around the mean.

---

# Step 3: Conditional Probability Simulation

## Scenario

We have:

- 60 male students
- 40 female students
- 30 successful male students
- 30 unsuccessful male students

We calculate:

\[
P(Pass|Male)
\]

## Formula

\[
P(Pass|Male)=
\frac{P(Pass \cap Male)}
{P(Male)}
\]

## Implementation

```python
import numpy as np

stud = np.array(
    ['male']*60 + ['female']*40
)

grades = np.array(
    ['yes']*30 +
    ['no']*30 +
    ['yes']*20 +
    ['no']*20
)


total = len(stud)

# P(Male)
males = stud == 'male'

male_count = np.sum(males)

P_male = male_count / total


# P(Pass and Male)
male_pass = np.sum(
    (stud == 'male') &
    (grades == 'yes')
)

P_pass_and_male = male_pass / total


# P(Pass | Male)

result = P_pass_and_male / P_male

print(result)
```

## Result

The output is:

```
0.5
```

## Explanation

The calculation is:

\[
P(Pass|Male)
=
\frac{30/100}{60/100}
\]

\[
=
\frac{30}{60}
=
0.5
\]

The simulation confirms the manual probability calculation.

---

# Step 4: Documentation

Each experiment was documented using Markdown explanations.

The documentation includes:

- The objective of each experiment.
- The implemented method.
- The obtained result.
- The meaning of the result.

---

# Machine Learning Connection

Probability is a fundamental concept in Machine Learning.

ML models often output probabilities such as:

- Probability of customer churn.
- Probability of image classification.
- Probability of spam emails.

Probability distributions help in:

- Understanding data behavior.
- Choosing suitable statistical methods.
- Building and evaluating ML models.

---

# Summary

During this training, I learned:

✅ Probability fundamentals  
✅ Complement, Addition, and Multiplication rules  
✅ Conditional Probability  
✅ Bayes' Theorem  
✅ Normal Distribution  
✅ Binomial Distribution  
✅ Uniform Distribution  
✅ Probability simulation using NumPy  
✅ Data visualization using Matplotlib  

This knowledge builds a strong foundation for Data Science and Machine Learning.