<img width="1476" alt="image" src="https://github.com/user-attachments/assets/d185760c-31ab-4af1-bf3a-ca7c1a030f78" />


## Project Overview

This project implements multiple machine learning models to predict whether a client will subscribe to a term deposit based on features from direct marketing campaigns.  
The analysis uses real-world banking data and compares the performance of Logistic Regression, Support Vector Machines (SVM), and Artificial Neural Networks (ANN).  
The focus is on model interpretability, performance evaluation, and practical preprocessing strategies.

## Objectives

- Load and prepare a financial marketing dataset for binary classification  
- Perform exploratory data analysis and categorical feature engineering  
- Implement and compare three classification models:
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Artificial Neural Network (ANN)
- Evaluate models based on accuracy, confusion matrix, and recall/precision  
- Interpret model behavior and make recommendations for deployment

## Tools and Libraries

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- TensorFlow / Keras (if ANN was implemented using these)

## Project Structure

term-deposit-subscription-prediction/  
├── term_deposit_prediction_logreg_svm_ann.ipynb   – Main notebook with all modeling and analysis  
├── banking.csv                                    – Original dataset with client and campaign data  
├── README.md                                      – Project documentation  

## Dataset Description

- Source: UCI Bank Marketing Dataset  
- Target variable: `y` – whether the client subscribed to a term deposit (yes/no)  
- Total records: ~41,000 rows  
- Selected features include:
  - Demographics: age, job, marital status, education  
  - Financial status: housing loan, personal loan  
  - Campaign metrics: contact type, campaign frequency, previous outcome  
  - Economic indicators: consumer price index, employment rate, euribor3m  
  - Time features: last contact date, month, day of week

Note: `duration` was excluded from the final model to prevent data leakage, as it is only known after the call takes place.

## Analysis Workflow

1. Load and inspect the dataset  
2. Clean categorical values (e.g. consolidate education levels)  
3. Visualize class imbalance and distributions  
4. Encode categorical features and scale numerics  
5. Split data into train and test sets  
6. Train and evaluate:
   - Logistic Regression  
   - SVM (with linear or RBF kernel)  
   - Neural Network with fully connected layers  
7. Compare accuracy and confusion matrices  
8. Conclude with insights for potential deployment

## How to Run

1. Clone this repository  
2. Ensure Python 3.x and pip are installed  
3. Install dependencies using pip:  
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

4. Place `banking.csv` in the project folder  
5. Run the notebook `term_deposit_prediction_logreg_svm_ann.ipynb`

## Requirements

Install dependencies using:  
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

## License

This project is licensed under the MIT License. See the LICENSE file for details.
