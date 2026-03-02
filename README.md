# Diabetes Prediction using Machine Learning

## Project Overview
This project predicts whether a patient has diabetes based on medical diagnostic features using Machine Learning models.

The goal is to build a classification model while prioritizing Recall, as missing a diabetic patient can be critical in medical diagnosis.

---

## Dataset
Dataset used: Diabetes Dataset

Features include:
- Pregnancies
- Glucose
- BloodPressure
- BMI
- Age
- Insulin
- DiabetesPedigreeFunction

Target:
- Outcome (0 = No Diabetes, 1 = Diabetes)

The dataset is slightly imbalanced.

---

## Models Used

- Logistic Regression
- Random Forest Classifier

---

## Model Performance

### Logistic Regression
- Accuracy: 0.75
- Precision: 0.52
- Recall: 0.83
- F1 Score: 0.64

### Random Forest
- Recall: 0.83
- F1 Score: 0.68
- Cross Validation Score: 0.84

---

## Final Model Selection

Both models achieved similar Recall (0.83), which is important for minimizing false negatives in medical diagnosis.

Random Forest achieved a higher F1-score (0.68) and slightly better cross-validation performance (0.84).

Therefore, Random Forest was selected as the final model due to its better balance between precision and recall.
