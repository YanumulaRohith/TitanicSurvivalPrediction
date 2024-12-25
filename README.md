# Titanic Survival Prediction

This project aims to predict whether a passenger survived the Titanic disaster using machine learning techniques. The Titanic dataset includes various features such as passenger demographics, ticket information, and cabin details, which are used to train a classification model. This README provides an overview of the project's approach, data preprocessing, model development, and performance metrics.

## Project Overview

The Titanic dataset provides detailed information about passengers on the Titanic, including whether they survived the disaster or not. Our goal is to build a machine learning model that can predict whether a passenger survived based on the available features. 

The following features are available in the dataset:
- **PassengerId**: Unique ID for each passenger
- **Survived**: Whether the passenger survived (1) or not (0)
- **Pclass**: Passenger class (1, 2, or 3)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger (male/female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Fare paid by the passenger
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Approach

### 1. **Data Preprocessing**

- **Handling Missing Data**: Missing values in the `Age`, `Fare`, and `Cabin` columns were handled as follows:
  - `Age` and `Fare` were imputed using their mean values.
  - `Cabin` values were imputed with the mode (most frequent value) because many values were missing.

- **Removing Outliers**: 
  - Outliers in the `Age` and `Fare` columns were detected and removed using the Interquartile Range (IQR) method.
  - The DataFrame shape reduced from 418 to 340 after removing outliers.

- **Feature Engineering**:
  - The `Sex` column was mapped to binary values (`0` for male and `1` for female).
  - Columns like `Name`, `Ticket`, `Cabin`, `SibSp`, and `Parch` were dropped, as they did not contribute significantly to the model's prediction performance.

- **Scaling**: Data scaling was performed to normalize numerical features before feeding them into the model.

### 2. **Model Selection and Training**

- **Model Used**: A **Logistic Regression** model was used for classification. Logistic regression is a widely used model for binary classification tasks, such as predicting survival (1 or 0).

- **Training**: The model was trained using a training dataset, and the accuracy was evaluated on a test dataset. 

### 3. **Model Performance**

- **Accuracy**: The model achieved a perfect accuracy of `1.0`, meaning it correctly predicted all passenger survival outcomes on the test set.

- **Confusion Matrix**:
  ```
  [[43  0]
   [ 0 25]]
  ```
  - True Negatives (TN): 43 instances of non-survived passengers correctly predicted as non-survived.
  - True Positives (TP): 25 instances of survived passengers correctly predicted as survived.
  - There were no False Positives (FP) or False Negatives (FN), indicating perfect classification.

- **Classification Report**:
  ```
              precision    recall  f1-score   support
          0       1.00      1.00      1.00        43
          1       1.00      1.00      1.00        25
      accuracy                           1.00        68
     macro avg       1.00      1.00      1.00        68
  weighted avg       1.00      1.00      1.00        68
  ```

  The model achieved a **precision**, **recall**, and **f1-score** of `1.00` for both classes (`0` and `1`), indicating perfect performance across all metrics.

### 4. **Convergence Warning**

During the logistic regression training, a **convergence warning** was issued:
```
ConvergenceWarning: lbfgs failed to converge (status=1): 
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
```
This warning indicates that the model did not converge within the default number of iterations (`max_iter=100`). To resolve this, we could increase the number of iterations or scale the data for better optimization.

## Conclusion

The model is performing extremely well with perfect accuracy, precision, recall, and F1-score. However, it is important to validate the model's performance on a broader dataset to avoid overfitting. Further tuning of the model and experimenting with different algorithms could lead to even better generalization.

## Future Improvements

- **Cross-validation**: Implementing cross-validation techniques to validate the model's performance on different subsets of the data.
- **Model Tuning**: Experimenting with different machine learning models (e.g., Random Forest, XGBoost) and hyperparameter tuning for better performance.
- **Feature Selection**: Testing more advanced feature engineering and selection methods to identify the most predictive features.

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/YanumulaRohith/TitanicSurvivalPrediction.git
   cd TitanicSurvivalPrediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script:
   ```bash
   python titanic_survival_prediction.py
   ```

## Acknowledgments

This project uses the Titanic dataset from Kaggle, which is a popular dataset for practicing machine learning techniques. For more information, visit the [Kaggle Titanic dataset page](https://www.kaggle.com/c/titanic).

---
