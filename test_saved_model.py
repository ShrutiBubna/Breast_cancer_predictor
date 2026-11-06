
import joblib
import pandas as pd

# Load the saved trained pipeline
model = joblib.load("best_tabular_model.joblib")

print("Saved model loaded successfully!")

# Load dataset to obtain one test example
dataset = pd.read_csv("breast_cancer.csv")

# Select 30 diagnostic features
X = dataset.iloc[:, 2:32]

# Select a sample for demonstration
sample = X.iloc[[100]].to_numpy()

# Predict using the saved model
prediction = model.predict(sample)

probability = model.predict_proba(sample)[0][1]

print("\nPrediction Result:")

print(
    "Predicted Diagnosis:",
    "Malignant" if prediction[0] == 1 else "Benign"
)

print(
    f"Malignant-Class Probability: {probability * 100:.2f}%"
)
