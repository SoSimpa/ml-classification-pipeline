"""End-to-end ML Classification Pipeline
Demonstrates preprocessing, model comparison, and evaluation for computational data science.
Uses a synthetic binary classification dataset inspired by student success or customer churn.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# Synthetic data: predicting "success" (e.g., course completion or positive outcome)
data = {
    'study_hours': np.random.normal(15, 5, n).clip(0, 40),
    'prior_gpa': np.random.normal(3.2, 0.6, n).clip(1.0, 4.0),
    'attendance_rate': np.random.beta(8, 2, n),
    'extracurriculars': np.random.randint(0, 5, n),
    'difficulty_level': np.random.choice([1, 2, 3, 4], n),
}
df = pd.DataFrame(data)

# Target: success probability based on features
logit = (-2.0 + 0.08*df['study_hours'] + 0.6*df['prior_gpa'] 
         + 2.5*df['attendance_rate'] + 0.15*df['extracurriculars'] 
         - 0.3*df['difficulty_level'] + np.random.normal(0, 0.5, n))
df['success'] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

print('Class balance:', df['success'].value_counts(normalize=True))

features = ['study_hours', 'prior_gpa', 'attendance_rate', 'extracurriculars', 'difficulty_level']
X = df[features]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'accuracy': acc, 'roc_auc': auc}
    print(f'\n=== {name} ===')
    print(f'Accuracy: {acc:.3f}')
    print(f'ROC-AUC: {auc:.3f}')
    print(classification_report(y_test, y_pred))

# Cross-validation on best model
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f'\nRandom Forest 5-fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})')

# Confusion matrix for RF
best_model.fit(X_train, y_train)
y_pred_rf = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=['Fail', 'Success'])
disp.plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print('\nPipeline complete. Model comparison and evaluation finished.')
