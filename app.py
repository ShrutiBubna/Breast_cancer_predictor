
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import cv2
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TumorTrace",
    page_icon="🎗️",
    layout="wide"
)

MODEL_PATH = "saved_models/best_mobilenetv2.keras"
THRESHOLD = 0.30

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_prediction_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_prediction_model()
TABULAR_MODEL_PATH = "best_tabular_model.joblib"

@st.cache_resource
def load_tabular_model():
    return joblib.load(TABULAR_MODEL_PATH)

tabular_model = load_tabular_model()


# ==========================================
# TRADITIONAL ML FEATURE NAMES
# ==========================================

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

def predict_tabular(features):

    input_data = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    prediction = int(
        tabular_model.predict(input_data)[0]
    )

    probabilities = tabular_model.predict_proba(
        input_data
    )[0]

    malignant_probability = float(probabilities[1])

    return prediction, malignant_probability


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_image(img_array):
    probability = float(
        model.predict(img_array, verbose=0)[0][0]
    )
    return probability

# ==========================================
# 5. GRAD-CAM FUNCTION
# ==========================================

def generate_gradcam(img_array, original_image):

    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = base_model.get_layer("out_relu")

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_outputs = grad_model(
            img_array, training=False
        )

        x = model.get_layer(
            "global_average_pooling2d"
        )(base_outputs)

        x = model.get_layer("dropout")(
            x, training=False
        )

        x = model.get_layer("dense")(x)

        x = model.get_layer("dropout_1")(
            x, training=False
        )

        predictions = model.get_layer("dense_1")(x)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color, cv2.COLOR_BGR2RGB
    )

    original = np.array(
        original_image.resize((224, 224))
    )

    overlay = cv2.addWeighted(
        original, 0.6,
        heatmap_color, 0.4, 0
    )

    return heatmap, overlay

# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎗️ TumorTrace")

    st.caption("AI-Based Breast Cancer Classification")

    st.divider()

    st.subheader("Project Information")

    st.write("**Model:** MobileNetV2")
    st.write("**Learning:** Transfer Learning")
    st.write("**Input:** Histopathology Image")
    st.write("**Image Size:** 224 × 224")
    st.write("**Explainability:** Grad-CAM")

    st.divider()

    st.subheader("Classification Threshold")

    st.metric(
        "Selected Threshold",
        f"{THRESHOLD * 100:.0f}%"
    )

    st.caption(
        "Predictions at or above this threshold "
        "are classified as Malignant."
    )

    st.divider()

    st.caption("Academic Research Project")

# ==========================================
# 7. MAIN INTERFACE
# ==========================================

st.title("🎗️ TumorTrace")

st.write(
    "An AI-powered research prototype for classifying "
    "breast histopathology images as Benign or Malignant."
)

st.divider()

# Project overview

st.subheader("About the System")

st.write(
    "This system uses a MobileNetV2-based deep learning model "
    "to analyze breast histopathology images. It also integrates "
    "Grad-CAM to visualize image regions that influence the "
    "model's prediction."
)


# ==========================================
# TRADITIONAL MACHINE LEARNING MODULE
# ==========================================

st.divider()

st.header("🧪 Traditional Machine Learning Prediction")

st.write(
    "Enter the 30 numerical diagnostic features to predict "
    "whether the case is Benign or Malignant using Logistic Regression."
)

# Load feature medians as sensible initial input values
@st.cache_data
def load_feature_medians():
    data = pd.read_csv("breast_cancer.csv")
    data = data.drop(columns=["id", "diagnosis"], errors="ignore")
    return data.median()

feature_medians = load_feature_medians()

st.info(
    "The fields are initially filled with dataset median values. "
    "Replace them with the measurements for the case you want to analyze."
)

with st.form("traditional_ml_form"):

    st.subheader("Enter Diagnostic Measurements")

    features = {}

    columns = st.columns(3)

    for i, feature in enumerate(FEATURE_NAMES):

        with columns[i % 3]:

            features[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(feature_medians[feature]),
                step=0.01,
                format="%.4f",
                key=f"traditional_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict Using Traditional ML",
        type="primary",
        use_container_width=True
    )

if submitted:

    prediction, malignant_probability = predict_tabular(features)

    st.subheader("Traditional ML Prediction Result")

    if prediction == 1:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")

    col1, col2 = st.columns(2)

    col1.metric(
        "Malignant Probability",
        f"{malignant_probability * 100:.2f}%"
    )

    col2.metric(
        "Model",
        "Logistic Regression"
    )

    st.progress(malignant_probability)

    st.caption(
        "Prediction generated using the saved Logistic Regression "
        "pipeline with StandardScaler preprocessing."
    )


# ==========================================
# 8. IMAGE UPLOAD
# ==========================================

st.divider()

st.header("Upload Histopathology Image")

st.write(
    "Upload an image in PNG, JPG, or JPEG format "
    "to generate a prediction."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            original_image,
            caption="Input Histopathology Image",
            width=400
        )

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                img_array = preprocess_image(original_image)

                probability = predict_image(img_array)

                prediction = (
                    "Malignant"
                    if probability >= THRESHOLD
                    else "Benign"
                )

                heatmap, overlay = generate_gradcam(
                    img_array, original_image
                )

            # ==================================
            # 9. PREDICTION RESULTS
            # ==================================

            st.divider()

            st.header("Prediction Results")

            if prediction == "Malignant":
                st.error("Prediction: Malignant")
            else:
                st.success("Prediction: Benign")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Malignant Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Classification Threshold",
                    f"{THRESHOLD * 100:.0f}%"
                )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                "The probability represents the model's estimated "
                "malignant score, not a confirmed medical diagnosis."
            )

            # ==================================
            # 10. EXPLAINABLE AI
            # ==================================

            st.divider()

            st.header("Explainable AI – Grad-CAM")

            st.write(
                "Grad-CAM (Gradient-weighted Class Activation Mapping) "
                "helps visualize image regions that contribute "
                "to the model's prediction."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    heatmap,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                    clamp=True
                )

            with col2:
                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            with st.expander("How to understand Grad-CAM"):

                st.write(
                    "- Brighter regions indicate stronger "
                    "model activation for the selected output.\n"
                    "- The overlay combines the heatmap "
                    "with the original histopathology image.\n"
                    "- These highlighted regions show model "
                    "attention, not medically confirmed cancer areas."
                )

    except Exception as e:
        st.error(f"Unable to process the image: {e}")

# ==========================================
# 11. PROJECT DETAILS
# ==========================================

st.divider()

with st.expander("Technical Details"):

    st.write("**Deep Learning Architecture:** MobileNetV2")
    st.write("**Approach:** Transfer Learning")
    st.write("**Classification:** Binary Classification")
    st.write("**Output Classes:** Benign and Malignant")
    st.write("**Explainable AI Technique:** Grad-CAM")

with st.expander("Limitations"):

    st.write(
        "This research prototype may produce incorrect predictions. "
        "Its performance on unseen images may differ from validation "
        "results. Grad-CAM visualizations are not medical evidence."
    )

# ==========================================
# 12. MODEL PERFORMANCE DASHBOARD
# ==========================================

st.divider()

st.header("📊 Model Performance Dashboard")

st.write(
    "Comparison of traditional machine learning "
    "models evaluated on the Wisconsin Breast Cancer dataset."
)

import pandas as pd

performance_data = {
    "Model": [
        "Logistic Regression",
        "LDA",
        "KNN",
        "Decision Tree",
        "Naive Bayes",
        "SVM"
    ],
    "Accuracy": [0.9860, 0.9510, 0.9371, 0.9441, 0.9021, 0.9510],
    "Precision": [1.0000, 0.9792, 0.8929, 0.8814, 0.8421, 0.9259],
    "Recall": [0.9623, 0.8868, 0.9434, 0.9811, 0.9057, 0.9434],
    "F1-Score": [0.9808, 0.9307, 0.9174, 0.9286, 0.8727, 0.9346],
    "ROC-AUC": [0.9958, 0.9948, 0.9828, 0.9517, 0.9805, 0.9933]
}

df = pd.DataFrame(performance_data)

# Best model metrics
best_model = df.loc[df["Accuracy"].idxmax()]

st.subheader("Best Performing Model")

col1, col2, col3 = st.columns(3)

col1.metric("Best Model", best_model["Model"])
col2.metric("Accuracy", f'{best_model["Accuracy"] * 100:.2f}%')
col3.metric("ROC-AUC", f'{best_model["ROC-AUC"] * 100:.2f}%')

st.divider()

# Performance comparison table
st.subheader("Model Comparison")

display_df = df.copy()

metric_columns = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"
]

for column in metric_columns:
    display_df[column] = display_df[column].map(
        lambda x: f"{x * 100:.2f}%"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# Accuracy chart
st.subheader("Accuracy Comparison")

accuracy_chart = df.set_index("Model")[["Accuracy"]]

st.bar_chart(accuracy_chart)

# Other performance metrics
st.subheader("Detailed Performance Metrics")

selected_metric = st.selectbox(
    "Select a metric to compare",
    ["Precision", "Recall", "F1-Score", "ROC-AUC"]
)

metric_chart = df.set_index("Model")[[selected_metric]]

st.bar_chart(metric_chart)

st.caption(
    "Note: These metrics belong to the traditional ML models. "
    "They should not be interpreted as MobileNetV2 performance."
)

# ==========================================
# 12. MEDICAL DISCLAIMER
# ==========================================

st.divider()

st.warning(
    "Medical Disclaimer: This application is an educational "
    "research prototype. It is not intended for clinical diagnosis "
    "or medical decision-making. Always consult qualified "
    "healthcare professionals for medical advice."
)

st.caption(
    "TumorTrace | Academic Research Project"
)
