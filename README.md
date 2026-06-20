 Logistic-Regression-IBM-AI-Engineer

Telecom Customer Churn Prediction using Logistic Regression

[[Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[[Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange.svg)](https://scikit-learn.org/)
[[IBM AI Engineer](https://img.shields.io/badge/Course-IBM%20AI%20Engineer%20Certificate-green.svg)]
(https://www.coursera.org/professional-certificates/ibm-ai-engineer)

Project Overview
This project implements a Logistic Regression classifier to predict customer churn for a telecommunications company. The primary objective is to identify which customers are likely to leave for a cable competitor. 

Beyond simply building a baseline model, this project focuses heavily on Feature Selection and Model Evaluation. By iteratively adding and removing features and evaluating the impact using Log Loss (Binary Cross-Entropy), the project demonstrates how to optimize a model's predictive confidence and eliminate noisy variables.

Dataset
The analysis utilizes the ChurnData.csv dataset, containing customer demographics, account information, and service usage.
Target Variable: `churn` (1 = Customer left, 0 = Customer stayed)
Initial Feature Set: `tenure`, `age`, `address`, `income`, `ed` (education), `employ` (employment duration), and `equip` (has equipment).

Data Preprocessing: 
Features were standardized using `StandardScaler` to ensure that variables with larger numeric ranges did not disproportionately influence the logistic regression gradient descent.

Methodology
1. Data Preprocessing: Handled data selection, cast the target variable to integers, and applied Standard Scaling (`StandardScaler`) to normalize the feature distributions.
2. Train/Test Split: Divided the dataset into 80% training and 20% testing sets (`random_state=4`) to ensure unbiased evaluation.
3. Modeling: Implemented `scikit-learn`'s `LogisticRegression` to calculate the probability of churn.
4. Evaluation (Log Loss): Instead of standard accuracy, the model was evaluated using Log Loss. Log Loss penalizes false classifications heavily, especially those made with high confidence, providing a stricter measure of model reliability.

The Experiment: Iterative Feature Selection
To understand the impact of individual variables, six different model iterations were trained and evaluated on the test set. 

| Model Version | Feature Modification | Log Loss Score | Impact |
| v1 (Baseline)** | Original 7 features (`tenure`, `age`, `address`, `income`, `ed`, `employ`, `equip`) | 0.6258 | Baseline |
| v2 | Added `callcard` | 0.6039 | 🟢 Improved (Useful signal) |
| v3 | Added `wireless` | 0.7227 | 🔴 Worsened (Introduced noise) |
| v4 | Added `callcard` & `wireless` | 0.7761 | 🔴 Worsened (Compounded noise) |
| v5 | Removed `equip` | 0.5302 | 🟢🟢Best Model (Massive improvement) |
| v6 | Removed `income` & `employ` | 0.6529 | 🔴 Worsened (Lost predictive power) |

Key Insights & Business Value
The "Less is More" Principle: The most significant improvement occurred in v5, where the `equip` feature was removed. This indicates that `equip` was highly collinear or noisy, confusing the model's decision boundary. Removing it lowered the Log Loss from `0.6258` to `0.5302`.
Feature Quality > Feature Quantity: Adding `wireless` (v3) degraded the model's confidence. Simply throwing more data into a model does not guarantee better performance; rigorous feature selection is required to separate signal from noise.

Business Impact: By utilizing the optimized model (v5), the telecom company can more accurately identify at-risk customers, allowing marketing teams to target retention campaigns efficiently, reducing customer acquisition costs.

Technologies Used
Python 3.14.5
Pandas & NumPy: Data manipulation, scaling, and array operations.
Scikit-Learn: Model building (`LogisticRegression`), preprocessing (`StandardScaler`), train/test splitting, and evaluation (`log_loss`).
Matplotlib: Visualizing feature coefficients to interpret model weights.

Further Exolanation of this Logistic Regression & Log Loss Experiment

The Core Concepts
    The Goal: We are predicting Customer Churn (whether a telecom customer will leave for a cable competitor). This is a Classification problem (Yes/No), which is why we used Logistic Regression instead of Linear Regression.
    The Metric (Log Loss): Also known as Binary Cross-Entropy. Unlike Accuracy (which just asks "Did you get it right?"), Log Loss asks "How confident were you when you got it right or wrong?" 
        If the model predicts a 99% chance of churn, and the customer does churn, Log Loss is very low (rewarded for high confidence).
        If the model predicts a 99% chance of churn, but the customer stays, Log Loss skyrockets (heavily penalized for being confidently wrong).
        Rule of thumb: Lower Log Loss = Better, more confident model.

The Experiment: Tweaking the Features
We started with a baseline of 7 features and tested how adding or removing specific variables affected the model's confidence (Log Loss). 
Here is what the data tells us:

    Baseline (v1): 0.6258
    Adding callcard (v2): 0.6039 🟢 (Improved! This feature provides useful signal).
    Adding wireless (v3): 0.7227 🔴 (Worsened! This feature introduced noise).
    Adding both (v4): 0.7761 🔴 (Even worse. Too much noise).
    Removing equip (v5): 0.5302 🟢🟢 (HUGE Improvement! Removing equip made the model significantly more confident and accurate).
    Removing income & employ (v6): 0.6529 🔴 (Worsened. These features actually contained useful predictive power).

The Big Takeaway: More features do not equal a better model. Removing a noisy, irrelevant feature (equip) improved the model's performance drastically more than adding new features did.


