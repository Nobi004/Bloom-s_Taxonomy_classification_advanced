import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
import joblib
import numpy as np

# Load data
train = pd.read_csv('artifacts/data/train.csv')
val = pd.read_csv('artifacts/data/val.csv')
test = pd.read_csv('artifacts/data/test.csv')

# TFIDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['text'])
X_val = vectorizer.transform(val['text'])
X_test = vectorizer.transform(test['text'])

y_train = train['label']
y_val = val['label']
y_test = test['label']

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')

# SVM
svm = LinearSVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')

# Save models
os.makedirs('artifacts/models', exist_ok=True)
joblib.dump(lr, 'artifacts/models/lr.pkl')
joblib.dump(svm, 'artifacts/models/svm.pkl')
joblib.dump(xgb, 'artifacts/models/xgb.pkl')
joblib.dump(vectorizer, 'artifacts/models/tfidf.pkl')

# Save results
results = {
    'lr': {'acc': lr_acc, 'f1': lr_f1, 'pred': lr_pred.tolist()},
    'svm': {'acc': svm_acc, 'f1': svm_f1, 'pred': svm_pred.tolist()},
    'xgb': {'acc': xgb_acc, 'f1': xgb_f1, 'pred': xgb_pred.tolist()}
}
np.save('artifacts/results/classical_results.npy', results)

print(f"Classical models trained. LR Acc: {lr_acc:.3f}, SVM Acc: {svm_acc:.3f}, XGB Acc: {xgb_acc:.3f}")