
# Python Fundamentals Training

This repository contains practical implementations and exercises covering core Python concepts, developed during my training.

## 1. Lists and Manipulations
I practiced managing lists, including indexing, sorting, reversing, and using basic methods like `append`, `insert`, and `pop`.

```python
a = [1, 3, 2, 3, 4]
a[0] = "mohammad"
a.sort()
print(a) # Output: [1, 2, 3, 3, 'mohammad']
2. String Formatting
Learned how to efficiently inject variables into strings using .format() and f-strings.

Python
day = 2
tomorrow = "third"
print(f"It's day {day} of training and tomorrow will be the {tomorrow} day.")
3. Data Structures
Applied various data structures to manage collections of data:

Tuples: Immutable collections for fixed data.

Dictionaries: Key-value pairs for structured data (e.g., person1).

Sets: Unordered collections used for unique elements and mathematical operations (Union |, Intersection &, Difference -).

4. Control Flow & List Comprehensions
Used if/else statements for decision-making and for loops for iteration. I also implemented List Comprehensions to write cleaner, "Pythonic" code.

Python
# List Comprehension example
ll = [x for x in range(50) if x % 2 == 0]
print(ll)
5. Object-Oriented Programming (OOP)
Defined a basic class to bundle data and functionality together.

Python
class Datapoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
