# SalSat: Employee Salary and Satisfaction Prediction
Predict your estimated salary and job satisfaction based on developer survey insights from Stack Overflow.

🚀 [Try the Live App](https://salsat-prediction.streamlit.app/)

## Contents:
[Overview](#overview) |
[Features](#features) |
[How it Works](#how-it-works) |
[Machine Learning Models](#machine-learning-models) |
[Installation](#installation) |
[Tech Stack](#tech-stack) |
[Acknowledgements](#acknowledgements)



## Overview

SalSat is a web application that predicts the salary and job satisfaction of software developers based on factors like country, experience, education, and employment type.

It uses real data from the [Stack Overflow Annual Developer Survey 2024](https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024), making the predictions feel personal, relevant, and grounded in actual trends.

## Features

- Predict estimated salary in **USD, INR or EUR** based on profile
- Predict job satisfaction as **Satisfied, Neutral or Dissatisfied**
- **Interactive visual** showing where your salary lies on the global experience-salary curve
- Export a **text report** with your results
- Clean and modern UI built with [Streamlit](https://docs.streamlit.io/)

## How it Works

1. User fills out a short form (experience, country, education, etc.)
2. The data is passed to pre-trained ML models
3. Predictions for salary and satisfaction are made
4. Results are visualized and a report can be downloaded

## Machine Learning Models

The application includes two separately trained models, each optimized for its respective prediction task:

1. **Salary Prediction**: 

    A CatBoost Regressor is used to estimate annual salary based on a range of developer attributes. Instead of raw salary values, we train on transformed midpoint values from salary ranges to smooth out noise. The model then outputs continuous salary predictions, making it suitable for real-world estimates.


2. **Satisfaction Prediction**: 

    For predicting job satisfaction, we use Logistic Regression, a robust and interpretable classification algorithm. It categorizes developers as either satisfied or dissatisfied based on their profile and work-related factors.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/danielmoses-dev/salsat-prediction
   cd salsat-prediction
   
2. Activate the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   
3. Install dependencies:
    ```bash
   pip install -r requirements.txt
   
4. Run the app:
    ```bash
   streamlit run app.py
   
## Tech Stack

- Python
- Pandas, NumPy
- Pickle, Joblib
- Scikit-learn
- CatBoost
- Altair, Seaborn
- Jupyter Lab
- Streamlit
   
## Acknowledgements

- Stack Overflow and Kaggle for the [Annual Developer Survey 2024 dataset](https://www.kaggle.com/datasets/berkayalan/stack-overflow-annual-developer-survey-2024)
- Streamlit community for UI inspiration
