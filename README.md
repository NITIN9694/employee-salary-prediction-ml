# Salary Prediction System

An end-to-end Machine Learning project that predicts employee salary based on features like age, education, job title, gender, and years of experience.

---

## Project Overview

The goal of this project is to build a machine learning model that predicts salary from employee-related information.

This project follows a real-world ML workflow:

- Data cleaning
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model comparison
- Model saving/deployment preparation

---

## Dataset Features

Input Features:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience

Target Feature:

- Salary

---

## Problem Type

This is a Regression problem because salary is a continuous numerical value.

---

## Project Workflow

### 1. Data Loading

Loaded dataset using Pandas.

```python
df = pd.read_csv("salary.csv")
```

### 2. Data Cleaning

Checked for:

- Missing values
- Null values
- Incorrect data types

Found missing values:

```text
Age                    2
Years of Experience    2
Salary                 2
```

Removed missing values:

```python
df = df.dropna()
```

Reason:

Missing values caused:

```text
loss: NaN
mae: NaN
```

because neural networks cannot process null values.

---

### 3. Feature Encoding

Machine learning models cannot understand text directly.

Converted categorical columns:

- Gender
- Education Level
- Job Title

using:

```python
pd.get_dummies()
```

Example:

Before:

| Job Title |
|------------|
| Engineer |
| Doctor |

After:

| Engineer | Doctor |
|------------|----------|
| 1 | 0 |
| 0 | 1 |

---

### 4. Train-Test Split

Split data into training and testing datasets:

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Reason:

- Training data → model learning
- Test data → performance evaluation

---

### 5. Model Training

Models tried:

1. Neural Network
2. Random Forest
3. XGBoost

---

## Neural Network Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])
```

Problems encountered:

### Overfitting

Observed:

```text
Training Loss ↓
Validation Loss ↑
```

Solutions used:

- Dropout
- EarlyStopping

---

## Model Comparison

| Model | MAE |
|---------|--------|
| Neural Network | 102466 |
| Random Forest | 9777 |
| XGBoost | 9877 |

---

## Best Model

Random Forest Regressor achieved the best result.

Final MAE:

```text
9777
```

Meaning:

Predictions are off by approximately 9.7k salary units on average.

---

## Libraries Used

- Python
- Pandas
- NumPy
- TensorFlow
- Scikit-Learn
- XGBoost
- Matplotlib
- Joblib

---

## Project Structure

salary-prediction-system/

├── data/

├── notebooks/

├── models/

│ └── salary_prediction.pkl

├── app.py

├── requirements.txt

├── README.md

└── salary.csv

---

## Future Improvements

- Deploy using FastAPI
- Build Streamlit UI
- Hyperparameter tuning
- Use larger datasets
- Add feature importance visualization

---

## Learning Outcomes

Through this project I learned:

- Handling missing values
- Encoding categorical features
- Train/test splitting
- Neural network training
- Overfitting detection
- Model evaluation
- Comparing ML algorithms
- Model saving and deployment workflow

---

## Author

Nitin Jha
