## Diabetes Prediction Model
# Project Overview
This project is a machine learning-based diabetes prediction system that predicts whether a patient is likely to have 
diabetes based on medical and lifestyle features. The model is built using Python, scikit-learn pipelines, 
and deployed with FastAPI, Streamlit, and Docker for scalable, real-time predictions.

# Key Features
 End-to-end ML pipeline: Data preprocessing, training, evaluation, and deployment.
 Logistic Regression model with hyperparameter tuning and cross-validation.
 Performance metrics: Precision, Recall, ROC-AUC, Confusion Matrix.
 Web deployment: Interactive interface with Streamlit.
 API deployment: Predict via FastAPI endpoints.
 Containerized: Docker support for easy deployment and reproducibility.
 Model persistence: Saved with joblib for fast reuse.

# Model Performance
 Metric	Value
 Model Precision	0.5417
 Model Recall	0.7091
 Cross-Validation Score	0.8314
 Best Grid Parameters	{'model__C': 0.2}
 Best Grid Score	0.8393
 Improved Precision	0.5244
 Improved Recall	0.7818
 ROC-AUC Score	0.8110
 Confusion Matrix	[[60, 39], [12, 43]]

Optimized for high recall, minimizing false negatives—a critical factor in medical diagnosis.

# Tech Stack
 Python 3.10+
 scikit-learn – Machine learning & pipelines
 pandas, numpy – Data handling
 joblib – Model saving & loading
 FastAPI – API deployment
 Streamlit – Web UI for user-friendly predictions
 Docker – Containerization for consistent deployments

# Conclusion
This project provides a robust, scalable, and user-friendly system for diabetes prediction. 
It showcases ML model development, hyperparameter optimization, pipeline automation, and deployment skills, making it 
ideal for real-world healthcare applications.
