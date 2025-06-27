import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and explore data
df = pd.read_csv('heart_full.csv')
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
df = pd.get_dummies(df, drop_first=True)

# Feature-target separation
X = df.drop("target", axis=1)
y = df["target"]

# Print column names used for training (sanity check)
print("\nFinal training features:")
print(X.columns.tolist())

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model training with better parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

# Comprehensive evaluation
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, predictions))
print("ROC AUC:", roc_auc_score(y_test, probabilities))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Save model with full path
model_path = "C:/Users/Prudvi raju/Downloads/heart_attack_predictor/heart_disease_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved successfully to: {model_path}")
