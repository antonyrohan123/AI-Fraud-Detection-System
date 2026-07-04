import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

df = pd.read_csv("dataset/creditcard.csv")

scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(
    df[['Amount']]
)

X = df.drop('Class', axis=1)

y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = IsolationForest(
    contamination=0.001,
    random_state=42
)

model.fit(X_train)

pred = model.predict(X_test)

pred = np.where(pred == -1, 1, 0)

print(classification_report(y_test, pred))
import joblib

joblib.dump(model, "models/fraud_model.pkl")

print("Model Saved Successfully!")