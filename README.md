# UFC Fight Outcome Prediction

## Project Overview

This project focuses on predicting the outcome of UFC fights using machine learning techniques. The goal is to estimate the probability that the fighter in the Red corner will win based on various performance metrics and historical fight data. The project includes feature engineering, model training and tuning, and development of a user-friendly web application for real-time predictions.

---

## Problem Statement

Predicting fight outcomes in UFC is a challenging task due to the complexity of the sport and variability in fighters' skills, styles, and physical conditions. Accurately forecasting fight winners can provide valuable insights for fans, analysts, and betting markets.

---

## Dataset

- The dataset contains detailed statistics for UFC fights, including fighter profiles, fight results, and performance metrics.
- Key features include fight statistics for both Red and Blue corner fighters, such as strikes, takedowns, and prior wins/losses.
- The target variable is the fight outcome (win/loss), labeled from the perspective of the Red corner fighter.

---

## Feature Engineering

- Created differential features by subtracting Blue corner statistics from Red corner statistics.
- Incorporated historical performance metrics such as win rates, recent fight results, and other fighter attributes.
- Applied label encoding to convert categorical target into binary labels (Red corner win = 1, Blue corner win = 0).

---

## Modeling

- Explored different classification algorithms, with CatBoost yielding the best results.
- Performed hyperparameter optimization using RandomizedSearchCV followed by GridSearchCV.
- Achieved an F1 score of approximately 0.89 on the validation set.

---

## Key Technologies and Tools

- Python (pandas, numpy, scikit-learn, catboost)
- Jupyter Notebook for exploratory data analysis (EDA) and modeling
- Streamlit for interactive web application development

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/sebahattindervisoglu/UFC-Fight-Prediction.git
   cd UFC-Fight-Prediction

Results and Evaluation
The CatBoost model with tuned hyperparameters demonstrates strong predictive power.

The model performance metrics include accuracy, precision, recall, and F1-score.

Feature importance analysis reveals key factors influencing fight outcomes.

Future Work
Incorporate additional fighter statistics such as reach, height, and fighting style.

Explore deep learning models or ensemble methods to improve prediction accuracy.

Integrate live UFC data streams for real-time predictions.

Enhance the web app with more interactive visualizations and user input options.
