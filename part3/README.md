# Applied AI & ML Essentials – Capstone Project
## Part 3: Advanced Modeling – Ensembles, Hyperparameter Tuning, and Machine Learning Pipeline

---

# Project Overview

This project extends the supervised learning models developed in Part 2 by implementing ensemble learning methods, hyperparameter tuning, cross-validation, feature importance analysis, feature ablation, learning curve analysis, and model serialization.

The objective is to compare multiple machine learning algorithms and identify the most robust model for predicting whether an individual's medical insurance charges are above the median value.

---

# Dataset

**Dataset:** Medical Insurance Cost Dataset

The dataset contains demographic and health-related information used to predict medical insurance charges.

### Features

| Feature | Description |
|---------|-------------|
| age | Age of the individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

---

# Target Variable

The binary classification target was defined as:

```python
y = (charges > charges.median()).astype(int)
```

- **0** → Charges less than or equal to the median
- **1** → Charges greater than the median

---

# Data Preprocessing

The same preprocessing workflow used in Part 2 was reproduced for this notebook:

- Loaded the dataset
- One-Hot Encoded categorical variables
- Split the data into training and testing sets
- Applied StandardScaler on the training data only to prevent data leakage

---

# Models Implemented

The following models were trained and evaluated:

- Decision Tree (Default)
- Controlled Decision Tree
- Decision Tree (Gini)
- Decision Tree (Entropy)
- Random Forest
- Gradient Boosting
- Logistic Regression (Baseline from Part 2)

---

# Decision Tree Analysis

## Default Decision Tree

A Decision Tree with default parameters was trained.

### Results

Training Accuracy: ***(Notebook Output)***

Testing Accuracy: ***(Notebook Output)***

The unconstrained decision tree achieved very high training accuracy but lower testing accuracy, indicating **overfitting**.

Decision Trees are considered **high-variance models** because they greed