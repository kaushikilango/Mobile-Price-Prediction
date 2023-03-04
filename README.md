# Mobile Price Prediction using Machine Learning

This repository contains Python code for predicting the prices of mobile phones using machine learning. The goal of this project is to build a model that can accurately predict the prices of mobile phones based on various features such as battery power, screen size, RAM, etc.

## Dataset
The dataset used for this project is the Mobile Price Classification dataset, which is publicly available on Kaggle. It contains data for 2,000 mobile phones, along with their features and corresponding price range labels. The dataset has been preprocessed and cleaned before using it for machine learning modeling.

## Machine Learning Techniques
We have experimented with various machine learning techniques including:

- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting

We have also performed feature selection to identify the most important features for the prediction task. We have used various evaluation metrics such as accuracy, precision, recall, and F1 score to evaluate the performance of our models.

## Requirements
The following Python packages are required to run the code in this repository:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage
The main script for this project is `mobile_price_prediction.py`. This script loads the dataset, performs data preprocessing, trains the machine learning models, and evaluates their performance. We have also provided Jupyter notebooks for data exploration and visualization.

To run the main script, navigate to the project directory and run the following command:

`python mobile_price_prediction.py`


## Results
Our best performing model was the Random Forest Classifier, which achieved an accuracy of 89.5% on the test set. We have also visualized the feature importance using a bar chart to show which features are most important for the prediction task.

## Conclusion
In this project, we have demonstrated how machine learning can be used to predict the prices of mobile phones based on their features. Our results show that it is possible to build an accurate prediction model using machine learning techniques. This project can be extended to other similar prediction tasks and can be used by mobile phone manufacturers to price their products competitively.
