# Sales Prediction using Simple Linear Regression

**Enhanced, Modular Implementation with Object-Oriented Design**

## Problem Statement

Build a model that predicts **sales** based on the amount of money spent on different platforms for marketing.
This implementation focuses specifically on analyzing the relationship between **TV advertising spend** and **Sales** using a **simple linear regression model**.

The dataset used is the **Advertising dataset** from the *Introduction to Statistical Learning (ISLR)* book.

---

## Internship Context

This project is my **second task** for the **CodSoft Data Science Internship**.
I have already completed Task 1 – Titanic Survival Prediction – and will be completing two more tasks to fulfill the internship requirements.

---

## About This Implementation

The original script provided with the dataset was a basic, linear version written in notebook format. I took that as a base and developed a **modular, reusable, and production-ready version** using Python and the scikit-learn library.

The logic has been restructured using **object-oriented programming**, where all key steps—data loading, model training, performance evaluation, visualization, and prediction—are handled through a well-organized class.

---

## Key Features

### Object-Oriented Design

Encapsulates the entire analysis into a single class `LinearRegressionAnalysis`, allowing for:

* Reusability across datasets
* Cleaner separation of concerns
* Easier debugging and extension

### Flexible Data Input

The implementation supports any CSV dataset with numeric input and target columns. Users only need to specify:

```python
x_column = 'TV'
y_column = 'Sales'
```

### Error Handling

Includes try-except blocks for:

* CSV file reading
* Column validation

This ensures the program provides helpful messages and doesn't crash due to bad input.

### Performance Evaluation

Evaluates the model using:

* R² Score (training and testing)
* RMSE
* MAE
  Includes interpretation logic to explain results.

### Visualization

Generates four visualizations in one figure:

1. Actual vs Predicted values
2. Regression line on training data
3. Residuals vs Predicted values
4. Distribution of residuals

### Easy Prediction

Accepts new input values and returns predicted outputs:

```python
model.make_predictions([100, 150, 200])
```

---

## Example Usage

```python
from regression_analysis import LinearRegressionAnalysis

model = LinearRegressionAnalysis()
model.load_data('advertising.csv')
model.train_model(x_column='TV', y_column='Sales')
model.evaluate_performance()
model.plot_results()
model.make_predictions([100, 150, 200])
```

---

## Requirements

Install the required libraries with:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## Dataset

This project uses the **Advertising** dataset, typically containing:

* TV advertising spend
* Radio advertising spend
* Newspaper advertising spend
* Product sales

Only the `TV` and `Sales` columns are used for this linear regression model.

---

## Acknowledgment

To understand the core concepts of linear regression and improve this project, I referred to:

* YouTube tutorials for practical implementations
* The **Data Science Masterclass by Krish Naik**, which helped clarify the statistical and modeling aspects

These resources provided valuable guidance, especially in structuring the model evaluation and visualization parts.
