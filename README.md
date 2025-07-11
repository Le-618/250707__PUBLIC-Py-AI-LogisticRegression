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
   Requirements: Python 3.11+

3. Install requirements if using the virtual environment:

   ```bash
   .venv/Scripts/pip install numpy
   .venv/Scripts/pip install pandas
   .venv/Scripts/pip install openpyxl
   ```
   Or if you decidedyour python script to be run through the main python environment:
   ```bash
   pip install numpy
   pip install pandas
   pip install openpyxl
   ```

---

## 🔧 Usage
All you have to do is run the **main.py** script, you can configure your local environment to run it

If you don't have any IDE to help you set up the , you can write a following command in the console to run using your venv:
```bash
.venv/Scripts/python main.py
```
Or run it straight through the original python executable:

```bash
python main.py
```