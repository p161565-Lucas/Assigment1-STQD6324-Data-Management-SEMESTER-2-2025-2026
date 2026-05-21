# Assignment 1 - STQD6324 Data Management (Semester 2 2025/2026)

# Iris Classification Using PySpark MLlib

## Overview

This project uses PySpark and Spark MLlib to perform multi-class classification on the Iris dataset.

The main objective is to build and compare different machine learning classification models for predicting iris flower species based on flower measurements.

Three supervised learning models are implemented:
- Logistic Regression
- Decision Tree
- Random Forest

The models are trained, tuned, and evaluated to compare their performance on the dataset.

---

## Dataset

The project uses the Iris dataset, which is a well-known dataset for classification tasks.

Dataset information:
- 150 samples
- 3 iris species
- 4 numerical features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

The dataset is balanced and does not contain missing values, so only basic preprocessing is required.

---

## Methodology

The project is implemented using PySpark.

Main workflow:
1. Load the dataset into a Spark DataFrame
2. Prepare and assemble features
3. Split the dataset into training and testing sets (70:30)
4. Train machine learning models
5. Perform hyperparameter tuning using Grid Search and 5-fold Cross Validation
6. Evaluate model performance
7. Compare the results of different models

---

## Models Used

### Logistic Regression
Used as a simple baseline classification model.

### Decision Tree
A rule-based model that is easy to understand and visualize.

### Random Forest
An ensemble learning model that combines multiple decision trees to improve performance and stability.

---

## Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

---

## Results

All three models achieved good classification performance on the Iris dataset.

Decision Tree and Random Forest achieved the highest accuracy score of 0.9821, while Logistic Regression achieved 0.9643.

Among the three models, Random Forest showed more stable and reliable performance overall.

The results indicate that tree-based models perform very well for this dataset.

---

## Tools and Technologies

- Python
- PySpark
- Spark MLlib
- Jupyter Notebook
- Pandas
- Scikit-learn

---

## How to Reproduce

1. Install required libraries:

```bash
pip install pyspark pandas scikit-learn
