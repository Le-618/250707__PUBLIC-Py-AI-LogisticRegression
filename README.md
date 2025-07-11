# Py‑AI‑LogisticRegression
A **pure-Python implementation** of Logistic Regression for binary classification, developed as an educational demo on machine learning fundamentals.
---
Participants:
- **Leonid Abdrakhmanov** - technical implementation
- **Egemen Kinay** - documentation
- **Burak Baran Zengiz** - presentation

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
    - [Preparing Your Data](#preparing-your-data)  
    - [Training the Model](#training-the-model)  
    - [Making Predictions](#making-predictions)  
5. [Constants](#constants)  
6. [Example](#example)  
7. [Testing](#testing)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## 🚀 Project Overview

This project implements Logistic Regression training model using the gradient descent algorithm. It illustrates:

- Handling input features and labels manually  
- Weight initialization and logistic/sigmoid function  
- Gradient Descent for optimization  
- Iterative loss calculation (cross-entropy)

Ideal for learners who want to understand the nuts and bolts of classification algorithms.

---

## 🔧 Features

- Binary logistic regression (0 or 1 classes)
- Manual parameter learning via gradient descent
- Configurable learning rate and iteration count
- Loss reporting for convergence analysis

---

## ⚙️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/Le-618/250707__PUBLIC-Py-AI-LogisticRegression.git
   cd 250707__PUBLIC-Py-AI-LogisticRegression
   ```

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Requirements: Python 3.12+

3. Install requirements if using the virtual environment:

   ```bash
   .venv/Scripts/pip install numpy
   .venv/Scripts/pip install pandas
   ```
   Or if your main python script doesn't 

---

## 🔧 Usage
### Preparing Your Data

Make sure you did the final

### Training the Model

```
model = LogisticRegression(learning_rate=0.1, num_iters=1000)
model.fit(X_train, y_train)
```

### Making Predictions

```python
predictions = model.predict(X_test)  # returns binary labels
probs = model.predict_proba(X_test)  # returns float probabilities (0–1)
```

---

## Constants

### `LogisticRegression(learning_rate=0.01, num_iters=1000)`

| Parameter       | Type    | Description                             |
|----------------|---------|-----------------------------------------|
| LEARNING_RATE  | float   | Gradient descent step size              |
| num_iters      | int     | Number of training iterations           |

#### `fit(X, y)`

- **X** – Feature matrix  
- **y** – Label vector

Trains weights and bias via gradient descent.

#### `predict_proba(X)`

Returns probability estimates for the positive class.

#### `predict(X)`

Thresholds probabilities at 0.5, returning binary labels.

---

## 📂 Example

```python
from logistic_regression import LogisticRegression

# Toy dataset
X = [[2.3, 4.5], [1.3, 3.2], [3.3, 0.5], [0.3, 1.5]]
y = [1, 1, 0, 0]

model = LogisticRegression(learning_rate=0.1, num_iters=2000)
model.fit(X, y)

print("Probabilities:", model.predict_proba(X))
print("Predictions:", model.predict(X))
```

---

## ✅ Testing

If there are test scripts:

```bash
python -m unittest discover tests
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a new branch: `git checkout -b feature/YourFeature`  
3. Implement and unit-test your feature  
4. Submit a pull request with clear description

---

## 📜 License

Specify your license, e.g.:

[MIT License](LICENSE)

---

### 🛠 Next Steps / Todo

- Add multi-class support via One‑vs‑Rest  
- Implement regularization (L1, L2 penalties)  
- Introduce mini‑batch or stochastic gradient descent  
- Plot loss curves using `matplotlib`  
- Integrate performance metrics like accuracy, precision, recall
