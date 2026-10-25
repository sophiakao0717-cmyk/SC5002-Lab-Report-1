# SC5002 Lab Assignment 1

This project includes the Python-based Google Colab file that trains a linear regression machine learning model using the California Housing Dataset. Feature scaling is used for data standaridization, Ridge regression with an alpha of 0.1, 10, 1000 is used, and finally, five-fold cross validation is adopted. 

## Description

The workflow of the training includes:
1. Loading and random sampling from the California Housing Dataset.
2. Standardizing features using 'StandardScaler'.
3. Evaluation linear regression and Ridge regression using five-fold cross validation ($R^2$ score).
4. Visualizing model performance.

## Getting Started

### Dependencies

The project uses the following standard Python data science libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

### Pipeline Summary

1. Dataset Setup (`Task 2.1`)
- **Dataset:** California Housing (`fetch_california_housing`)
- **Sampling:** 1,000 random samples (`random_state=42`)
- **Target Variable:** `MedHouseVal` (Median House Value)
- **Feature Processing:** All features are scaled using `StandardScaler` before being split into train and test sets (80/20 train-test split).

2. Model Training & Cross-Validation (`Task 2.2`)
- **Linear Regression:** Evaluated using 5-fold cross-validation with $R^2$ scoring.
- **Ridge Regression:** Evaluated using 5-fold cross-validation with varying penalty terms:
  - $\alpha = 0.1$
  - $\alpha = 10.0$
  - $\alpha = 1000.0$

3. Visualization (`Task 2.3`)
- A bar chart generated via `matplotlib` compares the mean 5-fold cross-validation $R^2$ scores across the models.

## Results

| Model | Hyperparameter ($\alpha$) | Mean 5-Fold $R^2$ Score |
| :--- | :---: | :---: |
| **Linear Regression** | N/A | **0.6457** |
| **Ridge Regression** | $\alpha = 0.1$ | **0.6457** |
| **Ridge Regression** | $\alpha = 10.0$ | **0.6443** |
| **Ridge Regression** | $\alpha = 1000.0$ | **0.3574** |

### Key Observations
- Low regularization ($\alpha = 0.1$) yields identical performance to standard Linear Regression ($R^2 \approx 0.6457$).
- Moderate regularization ($\alpha = 10.0$) slightly penalizes model weights with minimal loss in $R^2$ ($0.6443$).
- Excessive regularization ($\alpha = 1000.0$) results in severe underfitting, dropping the $R^2$ score significantly to $0.3574$.

https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
