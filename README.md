# Employee Attrition Prediction Project

## Problem Overview

Employee attrition is the voluntary departure of employees from an organization. It is an important business problem because turnover creates direct and indirect costs through recruiting, onboarding, training, lost productivity, and disruption to teams.

The goal of this project is to use predictive analytics to predict whether an employee will leave the organization based on HR and demographic variables. The project is designed to help organizations move from a reactive approach to a proactive one by identifying employees who may be at higher risk of leaving.

### Research Questions

1. Can employee attrition be predicted using HR and demographic features?
2. Which variables are the strongest predictors of attrition?
3. How does class imbalance handling affect model performance?
4. Do different employee groups show different attrition risk profiles?

---

## Dataset Description

This project uses the **IBM HR Analytics Employee Attrition & Performance** dataset. It is a structured HR dataset commonly used for attrition modeling and includes employee demographics, job-related variables, compensation variables, satisfaction measures, and work-history information.

The dataset contains **1,470 employee records** and **35 original features**, with **no missing values**. The target variable is **Attrition**, which is a binary outcome indicating whether an employee left the company. In this dataset, attrition is the minority class, with roughly **16%** of employees leaving and **84%** staying.


Examples of variables used in the project include:

- Age
- Department
- JobRole
- MonthlyIncome
- OverTime
- JobSatisfaction
- TotalWorkingYears
- YearsAtCompany
- WorkLifeBalance

The target variable is **Attrition**, which indicates whether an employee left the company. In the notebook, this variable is converted into a binary flag:

- `0 = No`
- `1 = Yes`

The following columns were removed before modeling because they were either constant or not useful predictors:

- `EmployeeCount`
- `Over18`
- `StandardHours`
- `EmployeeNumber`

---

## Instructions to Run the Code

### Files in this Repository

- `Employee_Attrition_Final.ipynb` — main notebook
- `requirements.txt` — required Python packages
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — dataset file

### Option 1: Run in Jupyter Notebook

1. Clone or download this repository.
2. Make sure the notebook and dataset CSV are in the same folder.
3. Install the required packages (listed below)
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run all cells from top to bottom.

### Option 2: Run in Google Colab

1. Upload the notebook to Google Colab.
2. Upload the dataset CSV when prompted, or place it in the Colab environment.
3. Run all cells in order.

### Required Packages

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

---

## Design Decisions and Trade-Offs

### 1. Data Preprocessing

Categorical variables were encoded using one-hot encoding so they could be used in machine learning models. Numeric variables were scaled for models such as logistic regression, where differences in scale can affect model fitting. Preprocessing was handled inside a pipeline to keep the workflow organized and reduce leakage risk.

### 2. Class Imbalance Handling

Attrition is the minority class in the dataset, so the project compares standard models with class-weighted versions. Logistic Regression, Decision Tree, and Random Forest were run both with and without `class_weight="balanced"`. For XGBoost, imbalance was handled using `scale_pos_weight`.

**Trade-off:** class weighting often improves recall for the attrition class, but it can reduce precision or overall F1-score.

### 3. Model Selection

The following models were used:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

These models were chosen to balance interpretability and predictive power:

- **Logistic Regression** is easier to interpret and explain to HR stakeholders.
- **Decision Tree** provides simple rule-based structure.
- **Random Forest** captures more complex non-linear relationships and provides feature importances.
- **XGBoost** is a strong boosting method for structured tabular data.

Although the original proposal expected XGBoost to perform best on the main evaluation metrics, the final notebook results showed that Logistic Regression remained highly competitive and easier to interpret. This is an important trade-off in HR analytics, where model transparency matters because decision-makers need to understand why an employee is being flagged as high-risk.

### 4. Evaluation Strategy

The project uses:

- an 80/20 train-test split
- Stratified 5-fold cross-validation on the training data
- evaluation metrics including Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curves, and Precision-Recall curves

**Trade-off:** the models were not heavily hyperparameter tuned. This keeps the project simpler and easier to understand, but some models may improve with further tuning.

---

## Example Outputs

The notebook produces outputs that summarize both the exploratory analysis and the model results.

### Exploratory Analysis

Example exploratory findings include:

- an overall attrition rate of approximately **16%**
- higher attrition rates in certain departments and job roles
- a much higher attrition rate among employees who work overtime
- differences in attrition patterns by job satisfaction, monthly income, and years at company

One of the strongest findings in the notebook is that employees who work overtime have a substantially higher attrition rate than employees who do not, making overtime an important business risk indicator.

### Model Results

The notebook includes:

- a comparison table for all models
- weighted vs unweighted model comparison
- confusion matrix for the selected model
- ROC curves for all models
- Precision-Recall curves for all models

In the final results, **Logistic Regression (No Class Weight)** achieved the strongest overall cross-validation F1-score, while **Logistic Regression (Balanced)** achieved higher recall for identifying employees who were likely to leave. This result is important because it shows the trade-off between overall balanced performance and catching a greater share of at-risk employees.

### Model Interpretation

To support interpretation, the notebook also includes:

- Logistic Regression coefficient tables and coefficient plots
- Random Forest feature importance table and feature importance plot

These outputs help identify which variables are most associated with attrition risk, such as overtime, income, tenure, and satisfaction-related factors.

### Example Business Insight

Overall, the outputs show that employee attrition is not random and can be predicted using employee-level data. The results suggest that HR teams could use these findings to focus retention efforts on high-risk employees, especially those with heavy overtime, lower satisfaction, or shorter tenure.

---

## Key Findings

Based on the notebook results:

- employee attrition can be predicted with useful performance using HR and demographic variables
- overtime, compensation, tenure, and satisfaction-related variables appear to be important drivers of attrition
- class weighting improves the model’s ability to identify employees who may leave, although it may reduce precision and does not always improve overall F1-score
- attrition risk is not evenly distributed across the organization, suggesting that targeted retention strategies may be more effective than one broad approach
- simpler and more interpretable models, such as Logistic Regression, can perform strongly on this dataset while remaining easier to explain to HR stakeholders

---

## Limitations

This project has several limitations:

- the dataset is relatively small and may not generalize perfectly to all organizations
- the analysis is based on historical structured HR data only
- the project focuses on classification performance rather than causal inference
- further hyperparameter tuning or alternative models may improve performance

Future improvements could include hyperparameter tuning, threshold optimization, and testing the models on additional HR datasets to assess generalizability.

---

## Citations

- Gallup. *This Fixable Problem Costs U.S. Businesses $1 Trillion.*
- Kaggle. *IBM HR Analytics Employee Attrition & Performance.*
- scikit-learn documentation. *OneHotEncoder.*
- scikit-learn documentation. *StandardScaler.*
- scikit-learn documentation. *LogisticRegression.*
- scikit-learn documentation. *StratifiedKFold.*
- scikit-learn documentation. *f1_score.*
- XGBoost documentation. *Parameters (`scale_pos_weight`).*
