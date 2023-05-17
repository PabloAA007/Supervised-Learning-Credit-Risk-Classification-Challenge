# Report Credit Risk Classification

## Overview of the Analysis

This analysis's goal was to evaluate credit risk using machine learning methods. Owing to the vast majority of risk-free loans compared to healthy loans, credit risk is fundamentally an imbalanced classification problem. Developing a model that determines the creditworthiness of borrowers using past lending data from a lending services provider.

Whether a loan is a high risk or low risk is indicated by the target variable "loan_status." The imbalance in the dataset highlighting by the initial analysis of the target variable using the `value_counts` function, which showed that there were many more low-risk loans than high-risk loans.

There were several steps in the machine-learning process:

1. **Data Split**: We first divide the data into features (represented by "X") and the intended variable (represented by "y"). 
2. **Train-Test Split**: We then divided these into datasets for training and testing. 
3. **Model Training and Evaluation (Original Data)**: Using the original data, we trained a Logistic Regression model and assessed its performance using several measures, such as accuracy, precision, recall, and F1 score.
4. **Resampling**: To resolve the class imbalance in the dataset, we subsequently used the 'RandomOverSampler' module from the imbalanced-learn toolkit. 
5. **Model Training and Evaluation (Resampled Data)**: Using this oversampled data, we trained a new Logistic Regression model and assessed its performance.


## Results

# Machine Learning Model 1
- Balanced Accuracy: 0.9520479254722232
- Precision: Low-risk loans (0.99), High-risk loans (0.85)
- Recall: Low-risk loans (1.0), High-risk loans (0.91)
- Specificity: Low-risk loans (0.85), High-risk loans (0.99)
- F1 Score: Low-risk loans (0.99), High-risk loans (0.88)
- Geometric Mean Score: Low-risk loans (0.92), High-risk loans (0.94)
- Index Balanced Accuracy: Low-risk loans (0.85), High-risk loans (0.94)
- Support: Low-risk loans (18765), High-risk loans (619)

# Machine Learning Model 2
- Balanced Accuracy: 0.9936781215845847
- Precision: Low-risk loans (1.0), High-risk loans (0.84)
- Recall: Low-risk loans (0.99), High-risk loans (0.99)
- Specificity: Low-risk loans (0.84), High-risk loans (1.0)
- F1 Score: Low-risk loans (1.0), High-risk loans (0.91)
- Geometric Mean Score: Low-risk loans (0.92), High-risk loans (0.99)
- Index Balanced Accuracy: Low-risk loans (0.92), High-risk loans (0.99)
- Support: Low-risk loans (18765), High-risk loans (619)

## Summary

Overall, the second model proved superior to the first by using oversampled data. This issue was most apparent in the better recall for high-risk loans, and the higher F1 score for high-risk loans. Identifying risky loans is critical in credit risk analysis, as not recognizing a high-risk loan could lead to serious financial consequences.

For high-risk loans, the second model showed a small drop in precision, indicating that there might be an exchange between precision and recall. It may be necessary to strike a balance between minimizing false negatives and false positives based on the bank's requirements.

The better choice, if the goal is to reduce the chances of overlooking high-risk loans, would be the second model that employs oversampled data.
