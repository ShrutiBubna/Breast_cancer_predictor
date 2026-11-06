# TumorTace

### Breast Cancer Prediction and Explainable AI

TumorTace is a machine-learning-based breast cancer prediction project.
It brings together tabular-data classification and medical-image
classification in a Streamlit application, with explainability support
to help users understand image-model predictions.

> **Important:** TumorTace is an academic project and research
> prototype. It is not a medical device and must not be used to diagnose
> breast cancer or replace advice from a qualified healthcare
> professional.

## Project Overview

The project explores two approaches to breast cancer prediction:

-   **Tabular machine learning:** Uses clinical/tabular features and a
    trained classification model.
-   **Image-based deep learning:** Includes CNN and MobileNetV2 model
    implementations for classifying breast-image data.
-   **Explainable AI:** Uses Grad-CAM visualizations to highlight image
    regions that influence a CNN-based prediction.
-   **Model evaluation:** Includes scripts and saved evaluation outputs
    for comparing model performance.

## Main Features

-   Streamlit-based user interface.
-   Tabular prediction using a saved machine-learning model.
-   Image-classification workflows using trained deep-learning models.
-   Grad-CAM visualizations for model interpretability.
-   Separate training, evaluation, preprocessing, and analysis scripts.
-   Saved model files for running prediction workflows without
    retraining.

## Technology Stack

-   Python
-   Streamlit
-   TensorFlow / Keras
-   Scikit-learn
-   NumPy and Pandas
-   Matplotlib
-   Pillow
-   Joblib

## Repository Structure

``` text
Breast_cancer_predictor/
│
├── app.py                         # Streamlit application
├── Values_ml.py                   # Tabular ML prediction logic
├── cnn_model.py                   # CNN model implementation
├── cnn_model_v2.py                # CNN model variant
├── cnn_model_v3.py                # CNN model variant
├── mobilenetv2_model.py            # MobileNetV2 model workflow
├── gradcam.py                      # Grad-CAM explainability
├── gradcam_mobilenetv2.py          # Grad-CAM for MobileNetV2
├── saved_models/                   # Trained model files
├── *.csv                           # Training/evaluation outputs
├── *.py                            # Training, preprocessing and evaluation scripts
├── .gitignore
└── README.md
```

The exact files in the repository may change as the project develops.

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/ShrutiBubna/Breast_cancer_predictor.git
cd Breast_cancer_predictor
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the application and the model
workflows. For example:

``` bash
pip install streamlit tensorflow scikit-learn numpy pandas matplotlib pillow joblib
```

If a `requirements.txt` file is added to the repository, install from it
instead:

``` bash
pip install -r requirements.txt
```

### 4. Run the application

From the project directory, run:

``` bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in
your browser.

## Trained Models and Datasets

The repository contains saved model files under `saved_models/` and
selected training/evaluation outputs.

The original image dataset directories and archive files are excluded
from GitHub to keep the repository manageable. To retrain the image
models, obtain the dataset separately and place it in the directory
structure expected by the relevant preprocessing and training scripts.

Model files and dataset paths must match the paths configured in the
application and scripts.

## Explainable AI with Grad-CAM

TumorTace includes Grad-CAM scripts that generate heatmaps for
image-model predictions. These visualizations can help illustrate which
regions of an input image contributed to a model's output.

Grad-CAM is an interpretation aid, not proof that a prediction is
correct or that a highlighted region corresponds to a clinically
meaningful finding.

## Project Workflow

1.  Provide tabular data or an image through the application workflow.
2.  The relevant trained model processes the input.
3.  The application presents the model output.
4.  For supported image workflows, Grad-CAM can provide a visual
    explanation.
5.  Training and evaluation scripts support model development and
    comparison.

## Limitations

-   Predictions depend on the quality, representativeness, and
    preprocessing of the training data.
-   Performance on one dataset does not guarantee performance on other
    populations, scanners, or clinical settings.
-   Grad-CAM highlights model-sensitive image regions; it does not
    establish causation or clinical validity.
-   The project has not been presented here as clinically validated or
    approved for medical use.

## Contributors

-   **Shruti Bubna**
-   **Roshni Gagrai**

## Academic Context

Developed as an academic project in the Department of Master of Computer
Applications (MCA), National Institute of Technology Warangal.

**Academic year:** 2025--2028

## License

No license has been specified yet. Unless a license is added to this
repository, the code should be treated as having all rights reserved by
its authors.
