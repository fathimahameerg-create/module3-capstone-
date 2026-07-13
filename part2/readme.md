# Applied AI & ML Essentials – Capstone Project
## Part 2: Supervised Machine Learning Model – Build, Train, and Evaluate

## Project Overview

This project demonstrates the development and evaluation of supervised machine learning models using the Medical Insurance Cost Dataset. Two predictive tasks were performed:

1. **Regression** – Predicting medical insurance charges.
2. **Binary Classification** – Predicting whether an individual's insurance charges are above the median value.

The project includes data preprocessing, feature encoding, scaling, regression modeling, classification modeling, model evaluation, regularization experiments, threshold sensitivity analysis, and bootstrap confidence interval estimation.

---

# Dataset

**Dataset:** Medical Insurance Cost Dataset

The dataset contains demographic and health-related information of individuals along with their medical insurance charges.

### Features

| Feature | Description |
|----------|-------------|
| age | Age of the individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

---

# Target Variables

### Regression Target (y_reg)

The regression target is:

```python
charges
```

It represents the continuous medical insurance cost.

### Classification Target (y_clf)

A binary target was created using the median of insurance charges.

```python
y_clf = (charges > charges.median()).astype(int)
```

- **0** → Charges less than or equal to the median
- **1** → Charges greater than the median

---

# Data Preprocessing

The following preprocessing steps were performed:

- Loaded the cleaned dataset from Part 1.
- Defined feature matrix and target variables.
- Applied **One-Hot Encoding** to categorical variables.
- Split the data into training (80%) and testing (20%).
- Applied StandardScaler using only the training data.

---

# Why One-Hot Encoding?

The categorical features (`sex`, `smoker`, and `region`) do not have any natural order.

Using Label Encoding would incorrectly introduce an ordinal relationship between categories (for example, assigning numbers to regions would imply that one region is greater than another).

One-Hot Encoding converts each category into separate binary columns, avoiding this false ordinal relationship.

---

# Data Leakage Prevention

Feature scaling was performed only on the training data.

```python
scaler.fit(X_train)
```

The same scaler was then applied to the test data.

```python
scaler.transform(X_test)
```

Fitting the scaler on the complete dataset would introduce **data leakage**, as the training process would gain information from the test set, leading to overly optimistic model performance.

---

# Regression Models

Two regression models were trained.

## 1. Linear Regression

Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score

The model coefficients were analysed to determine the most influential features.

### Coefficient Interpretation

A large positive coefficient means that increasing the corresponding standardized feature increases the predicted insurance charges.

A large negative coefficient means that increasing the corresponding feature decreases the predicted insurance charges.

The three features with the largest absolute coefficients were identified from the trained model.

---

## 2. Ridge Regression

A Ridge Regression model with

```python
alpha = 1.0
```

was trained using the same training and testing split.

### Ridge vs Linear Regression

| Model | MSE | R² Score |
|-------|------|-----------|
| Linear Regression | *(Notebook Output)* | *(Notebook Output)* |
| Ridge Regression | *(Notebook Output)* | *(Notebook Output)* |

### Why Ridge Regression?

Ridge Regression applies L2 regularization to reduce excessively large coefficients.

The **alpha** parameter controls the amount of regularization.

Higher alpha values shrink coefficients more strongly, helping reduce overfitting.

---

# Classification Model

A Logistic Regression model was trained to classify whether insurance charges are above the median.

The model was evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Area Under Curve (AUC)

---

# Class Imbalance

The class distribution was checked before training.

The binary classes created using the median split were approximately balanced.

Therefore, additional imbalance handling techniques such as SMOTE or class weighting were not required.

---

# Precision and Recall

### Precision

Precision measures how many predicted positive cases are actually positive.

\[
Precision = \frac{TP}{TP + FP}
\]

---

### Recall

Recall measures how many actual positive cases are correctly identified.

\[
Recall = \frac{TP}{TP + FN}
\]

---

# Important Metric

For this classification task, **Recall** is considered slightly more important.

Missing individuals with high medical costs (False Negatives) could reduce the usefulness of the predictive model.

Therefore, identifying as many high-cost individuals as possible is desirable.

---

# ROC Curve and AUC

The ROC Curve illustrates the trade-off between the True Positive Rate and False Positive Rate.

The Area Under the Curve (AUC) measures the model's ability to distinguish between the two classes.

An AUC value closer to **1.0** indicates excellent classification performance, while an AUC of **0.5** indicates random guessing.

---

# Decision Threshold Sensitivity

The logistic regression model was evaluated at five thresholds.

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

Precision, Recall, and F1 Score were calculated at each threshold.

The threshold producing the highest F1 Score was selected as the best balance between Precision and Recall.

Lowering the threshold generally increases Recall but decreases Precision.

Increasing the threshold improves Precision but may reduce Recall.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained using

```python
C = 0.01
```

This stronger regularization applies a larger L2 penalty.

### What does C control?

The parameter **C** controls the inverse strength of regularization.

- Large C → Less regularization
- Small C → Stronger regularization

Reducing C shrinks model coefficients and may improve generalization, although excessive regularization can reduce predictive performance.

---

# Bootstrap Confidence Interval

A bootstrap analysis with **500 resamples** was performed.

For each bootstrap sample:

- Test samples were drawn with replacement.
- AUC was computed for both Logistic Regression models.
- The difference in AUC values was calculated.

Finally,

- Mean AUC Difference
- 2.5th Percentile
- 97.5th Percentile

were reported as the 95% confidence interval.

If the confidence interval excludes zero, the performance difference between the two models is considered statistically reliable.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook

---

# Project Structure

```
part2/
│
├── data/
│   └── insurance.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── logistic_regression.pkl
│   ├── logistic_regression_regularized.pkl
│   ├── regression_comparison.csv
│   ├── logistic_comparison.csv
│   └── threshold_results.csv
│
├── plots/
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── part2.ipynb
├── requirements.txt
└── README.md
```

---

# How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
jupyter notebook part2.ipynb
```

or open the notebook using Visual Studio Code and execute all cells sequentially.

---

# Conclusion

This project successfully developed regression and binary classification models using the Medical Insurance Cost Dataset. The models were evaluated using multiple performance metrics, regularization techniques, threshold analysis, and bootstrap confidence intervals. The results demonstrate a complete supervised machine learning workflow, including preprocessing, model development, evaluation, and interpretation.