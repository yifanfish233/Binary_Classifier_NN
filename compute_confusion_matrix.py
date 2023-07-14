from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

with open("classifier_feature_record.txt", "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
y_label = [int(line.split("  ")[1]) for line in lines]
y_pred = [int(float(line.split("  ")[2]) > 0) for line in lines]

print("Accuracy: \n", accuracy_score(y_label, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_label, y_pred))
print("Classification Report: \n", classification_report(y_label, y_pred))
