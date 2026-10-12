# Breast Cancer Prediction — Traditional Machine Learning

# ============================================================

# 1. IMPORT LIBRARIES

# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning models

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Data splitting and cross-validation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Preprocessing and pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Evaluation metrics

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
ConfusionMatrixDisplay,
classification_report,
roc_curve
)

# ============================================================

# 2. LOAD DATASET

# ============================================================

dataset = pd.read_csv("breast_cancer.csv")

# ============================================================

# 3. UNDERSTAND THE DATASET

# ============================================================

print("Dataset Shape:")
print(dataset.shape)

print("\nFirst Five Records:")
print(dataset.head())

print("\nDataset Information:")
dataset.info()

print("\nStatistical Summary:")
print(dataset.describe())

print("\nMissing Values:")
print(dataset.isnull().sum())

# ============================================================

# 4. ENCODE THE TARGET VARIABLE

# ============================================================

# Diagnosis:

# B = Benign

# M = Malignant

#

# Machine learning models require numerical target values.

#

# B -> 0

# M -> 1

dataset["diagnosis"] = dataset["diagnosis"].map({
"B": 0,
"M": 1
}).astype(int)

print("\nEncoded Dataset:")
print(dataset.head())

# ============================================================

# 5. SEPARATE FEATURES AND TARGET

# ============================================================

# X contains the 30 diagnostic features.

# The first column is ID and the second column is diagnosis.

# Therefore, columns 2 to 31 are selected.

X = dataset.iloc[:, 2:32].values

# y contains the diagnosis.

# 0 = Benign

# 1 = Malignant

y = dataset.iloc[:, 1].values

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# ============================================================

# 6. SPLIT DATA INTO TRAINING AND TESTING SETS

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.25,
random_state=0,
stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ============================================================

# 7. DEFINE MACHINE LEARNING MODELS

# ============================================================

models = [
(
    "Logistic Regression",
    LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=0
    )
),

(
    "LDA",
    LinearDiscriminantAnalysis()
),

(
    "KNN",
    KNeighborsClassifier()
),

(
    "Decision Tree",
    DecisionTreeClassifier(
        random_state=0
    )
),

(
    "Naive Bayes",
    GaussianNB()
),

(
    "SVM",
    SVC(
        gamma="auto",
        probability=True,
        random_state=0
    )
)

]

# ============================================================

# 8. STRATIFIED 10-FOLD CROSS-VALIDATION

# ============================================================

results = []
names = []

print("\n================ MODEL COMPARISON ================\n")

for name, model in models:


# Pipeline:
# 1. Standardize the features
# 2. Train the machine learning model
#
# The scaler is fitted separately inside each CV fold.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# Stratified K-Fold maintains approximately the same
# proportion of benign and malignant cases in each fold.

    kfold = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=0
    )

    cv_results = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="accuracy"
    )

    results.append(cv_results)
    names.append(name)

    print(
        f"{name}: "
        f"{cv_results.mean():.4f} "
        f"(± {cv_results.std():.4f})"
    )


# ============================================================

# 9. MODEL COMPARISON GRAPH

# ============================================================

mean_scores = [
result.mean()
for result in results
]

plt.figure(figsize=(10, 6))

plt.bar(
names,
mean_scores,
width=0.6
)

plt.ylim(0.90, 1.00)

plt.ylabel("Mean Cross-Validation Accuracy")
plt.xlabel("Machine Learning Algorithm")

plt.title(
"Comparison of Machine Learning Algorithms"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
"algorithm_comparison.png",
dpi=300,
bbox_inches="tight"
)

plt.savefig(
    "algorithm_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================

# 10. SELECT THE BEST MODEL

# ============================================================

best_index = np.argmax(mean_scores)

best_model_name = names[best_index]
best_model = models[best_index][1]

print("\nBest Model:")
print(best_model_name)

print(
f"Mean CV Accuracy: "
f"{mean_scores[best_index] * 100:.2f}%"
)

# ============================================================

# 11. TRAIN THE BEST MODEL

# ============================================================

# Create a complete pipeline for the selected model.

final_pipeline = Pipeline([
("scaler", StandardScaler()),
("model", best_model)
])

# Train only on the training dataset.

final_pipeline.fit(
X_train,
y_train
)

# ============================================================

# 12. PREDICT ON THE UNSEEN TEST DATA

# ============================================================

y_pred = final_pipeline.predict(X_test)

# Probability of malignant class

y_probability = final_pipeline.predict_proba(X_test)[:, 1]

# ============================================================

# 13. EVALUATION METRICS

# ============================================================

accuracy = accuracy_score(
y_test,
y_pred
)

precision = precision_score(
y_test,
y_pred
)

recall = recall_score(
y_test,
y_pred
)

f1 = f1_score(
y_test,
y_pred
)

roc_auc = roc_auc_score(
y_test,
y_probability
)

print("\n================ TEST SET EVALUATION ================\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ============================================================

# 14. CLASSIFICATION REPORT

# ============================================================

print("\nClassification Report:")
print(
classification_report(
y_test,
y_pred,
target_names=[
"Benign",
"Malignant"
]
)
)

# ============================================================

# 15. CONFUSION MATRIX

# ============================================================

cm = confusion_matrix(
y_test,
y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
confusion_matrix=cm,
display_labels=[
"Benign",
"Malignant"
]
)

disp.plot()

plt.title(
f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)



plt.close()

# ============================================================

# 16. ROC CURVE

# ============================================================

fpr, tpr, thresholds = roc_curve(
y_test,
y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
fpr,
tpr,
label=f"{best_model_name} (AUC = {roc_auc:.3f})"
)

plt.plot(
[0, 1],
[0, 1],
linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
"ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)



plt.close()

# ============================================================

# 17. PREDICTION FOR A NEW PATIENT

# ============================================================

# 30 diagnostic measurements for a new patient.

value = [[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,
8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
]]

# The pipeline automatically applies the SAME scaler

# that was fitted during model training.

new_prediction = final_pipeline.predict(value)

malignant_probability = (
final_pipeline.predict_proba(value)[0][1]
)

print("\n================ NEW PATIENT PREDICTION ================\n")

if new_prediction[0] == 1:


    print("Prediction: Malignant")


else:

    print("Prediction: Benign")

print(
f"Estimated probability of malignant class: "
f"{malignant_probability * 100:.2f}%"
)
