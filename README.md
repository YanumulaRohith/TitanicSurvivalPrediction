# Titanic Survival Prediction

This project aims to predict whether a passenger survived the Titanic disaster based on various features such as age, gender, ticket class, fare, and cabin. The dataset contains real-world passenger information from the Titanic disaster, which is used to build a machine learning model to classify survival outcomes.

The solution utilizes data preprocessing, feature engineering, and machine learning techniques to predict survival with accuracy.

## Project Overview

- **Objective**: Predict whether a passenger survived the Titanic disaster using machine learning techniques.
- **Dataset**: The dataset contains features like PassengerId, Survived (target), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

## Approach

### 1. **Data Preprocessing**

- **Handling Missing Values**:

  - The missing `Age` values are filled with the median of the column.
  - The missing `Embarked` values are filled with the most frequent embarkation port.
  - The `Cabin` column is dropped due to a large number of missing values.

- **Feature Encoding**:

  - `Sex` is converted to binary values (0 for male, 1 for female).
  - `Embarked` is one-hot encoded into separate columns (S and C).

- **Feature Engineering**:

  - A new feature `FamilySize` is created by combining the `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard) columns.

- **Data Normalization**:
  - Some models may require scaling; however, this was not applied in this case as the RandomForest and Logistic Regression algorithms do not necessarily require normalization for this dataset.

### 2. **Model Selection**

We used two machine learning algorithms to predict survival:

- **Logistic Regression**: A simple linear model for binary classification.
- **Random Forest Classifier**: A robust ensemble method that uses multiple decision trees to make predictions.

### 3. **Evaluation Metrics**

- **Accuracy**: The percentage of correct predictions made by the model.
- **Confusion Matrix**: A table showing the true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

### 4. **Model Saving**

- The trained model is saved using `joblib` for future use or deployment.

## How to Use the Code

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Install Dependencies**:

   Use `pip` to install the required libraries.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook or Python Script**:

   You can run the Jupyter notebook `Titanic_Survival_Prediction.ipynb` or the Python script `titanic_model.py` to train the model and make predictions.
