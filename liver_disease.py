import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file ="/storage/emulated/0/Download/Liver Patient Dataset (LPD)_train.csv/Liver Patient Dataset more corrected_train.csv"
df = pd.read_csv(file)
print(df)



x = df.iloc[:, 2:11]
y = df[["Result"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("training data")
print(x_train)
print("test data")
print(x_test)

print("training data")
print(y_train)
print("test data")
print(y_test)

model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(x_train, y_train)

# Test model
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))


model = DecisionTreeClassifier()

model.fit(x, y)

age = float(input("enter your age :"))
TB = float(input("enter your total bilirubin :"))
DB = float(input("enter your direct bilirubin :"))
AAP = float(input("enter your alkphos alkaline phosphotes :"))
sgpt = float(input("enter your sgpt :"))
sgot = float(input("enter your sgot :"))
TP = float(input("enter your total protein :"))
ALB = float(input("enter your albumin :"))
AG_ratio = float(input("enter your ag_ratio :"))

result = model.predict([[age, TB, DB, AAP, sgpt, sgot, TP, ALB, AG_ratio]])[0]

if result == 1:
	print("liver diseas patient")
elif result == 2:
	print("non liver disease")
else:
	print("other reason")





