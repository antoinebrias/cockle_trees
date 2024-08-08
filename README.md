# Cockle Density Classification

This Python script performs a comparative analysis of two popular machine learning models, **Random Forest** and **XGBoost**, using a dataset on cockle densities and green macroalgal (GMA) biomass from Yaquina Bay, Oregon. The script covers the following steps:

1. **Data Preprocessing**:
   - Loads and formats the dataset, including parsing dates and handling numeric columns.
   - Encodes categorical variables and selects relevant features for modeling.
   - Splits the data into training and testing sets.

2. **Model Training and Evaluation**:
   - Trains Random Forest and XGBoost classifiers on the training data.
   - Evaluates model performance using accuracy, classification reports, confusion matrices, and ROC AUC scores.

3. **Feature Importance Analysis**:
   - Plots feature importance for both models to highlight the most influential features in predictions.
   - Includes diagonal labeling for better readability.

This script provides insights into the performance and interpretability of ensemble learning methods on ecological data, making it a valuable tool for data scientists and researchers in environmental studies.

## Requirements
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## Usage
Ensure the dataset is available at the specified file path and run the script to analyze and visualize model performance and feature importance. The dataset is available here: https://catalog.data.gov/dataset/cockle-green-macroalgae-field-survey-data-2014
