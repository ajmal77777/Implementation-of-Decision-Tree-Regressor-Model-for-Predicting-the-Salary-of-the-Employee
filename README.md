# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Salary.csv dataset and inspect shape, structure, and missing values
2. Encode categorical 'Position' column into numeric labels using LabelEncoder
3. Split data into features (Position, Level) and target (Salary)
4. Divide dataset into training (80%) and testing (20%) sets
5. Train a Decision Tree Regressor on the training data
6. Predict salaries on test data and evaluate model using R² score
7. Visualize the trained decision tree and display feature importances

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOHAMED AJMAL H
RegisterNumber:  212225230173
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

csv_path = r"E:\Salary.csv"    
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

print("Shape:", data.shape)
print(data.head())

print("\nInfo:")
print(data.info())

print("\nMissing Values:\n", data.isnull().sum())

if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
print(X.head())

print("\nTarget Sample:")
print(y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR2 Score: {r2:.4f}")

plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.tight_layout()
plt.savefig("decision_tree_plot.png", dpi=150)
print("\nDecision tree plot saved as decision_tree_plot.png")

importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
print(importances)
```

## Output:

<img width="1373" height="689" alt="image" src="https://github.com/user-attachments/assets/0dfd3978-4249-4c0d-9f11-062d969112c8" />

<img width="730" height="644" alt="image" src="https://github.com/user-attachments/assets/32888399-ffc8-4717-98f0-339db845712a" />

<img width="1277" height="785" alt="image" src="https://github.com/user-attachments/assets/68acee3f-ef09-4289-96c2-c68bca17e7d6" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
