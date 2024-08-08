import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


# Load the dataset
file_path = './cockle-fieldsurveydata-xlsx-1.xls'
df = pd.read_excel(file_path, 'Data')

# Display basic information about the dataset
print(df.info())
print(df.head())

# Remove rows with at least one missing value
df.dropna(inplace=True)

# Parse dates
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Handle commas in numeric columns by replacing commas and converting to float
for col in ['Easting', 'Shallow_cm', 'Deep_cm', 'Avg_cm']:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Extract features from the date column
#df['Year'] = df['Date'].dt.year
#df['Month'] = df['Date'].dt.month
#df['Day'] = df['Date'].dt.day

# Drop the original date column
df.drop('Date', axis=1, inplace=True)


['Uncovered', 'Semi_covered', 'Buried', 'Total','Present']

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Define the feature matrix and target vector
X = df.drop(['Uncovered', 'Semi_covered', 'Buried', 'Total','Present','Site','Station','Easting'], axis=1)  
y = df['Present']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train Boosted Trees (XGBoost)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Boosted Trees (XGBoost) Accuracy:", xgb_accuracy)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("\nBoosted Trees (XGBoost) Classification Report:\n", classification_report(y_test, xgb_predictions))



# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
xgb_cm = confusion_matrix(y_test, xgb_predictions)

# ROC Curve and AUC Score
rf_proba = rf_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

rf_auc = roc_auc_score(y_test, rf_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)

# Plot Feature Importances
def plot_feature_importances(model, features, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

# Print Confusion Matrices
print("\nRandom Forest Confusion Matrix:\n", rf_cm)
print("\nBoosted Trees (XGBoost) Confusion Matrix:\n", xgb_cm)

# Print ROC AUC Scores
print("\nRandom Forest ROC AUC Score:", rf_auc)
print("Boosted Trees (XGBoost) ROC AUC Score:", xgb_auc)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, color='blue', label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(xgb_fpr, xgb_tpr, color='red', label=f'XGBoost (AUC = {xgb_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importances for both models
plot_feature_importances(rf_model, X.columns, 'Random Forest Feature Importances')
plot_feature_importances(xgb_model, X.columns, 'XGBoost Feature Importances')
