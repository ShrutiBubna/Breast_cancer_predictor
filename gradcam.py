
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "saved_models/best_cnn_model.keras"

IMAGE_PATH = (
    r"dataset_cancer_v1\classificacao_binaria"
    r"\40X\malignant\SOB_M_DC-14-2523-40-010.png"
)

IMG_SIZE = (224, 224)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")

# ==========================================
# LOAD AND PREPROCESS IMAGE
# ==========================================

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize(IMG_SIZE)

image_array = np.array(image, dtype=np.float32) / 255.0

input_image = np.expand_dims(image_array, axis=0)


# ==========================================
# CREATE GRAD-CAM MODEL
# ==========================================

last_conv_layer = model.get_layer("conv2d_2")

# Create a symbolic input
grad_input = tf.keras.Input(shape=(224, 224, 3))

x = grad_input
conv_outputs = None

# Rebuild the forward graph using existing trained layers
for layer in model.layers:

    x = layer(x)

    if layer.name == "conv2d_2":
        conv_outputs = x

# Create model with two outputs:
# 1. Final convolutional feature maps
# 2. Final prediction

grad_model = tf.keras.Model(
    inputs=grad_input,
    outputs=[conv_outputs, x]
)


# ==========================================
# CALCULATE GRADIENTS
# ==========================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_image,
        training=False
    )

    malignant_score = predictions[:, 0]

gradients = tape.gradient(
    malignant_score,
    conv_outputs
)

# ==========================================
# GENERATE GRAD-CAM HEATMAP
# ==========================================

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    conv_outputs * pooled_gradients,
    axis=-1
)

heatmap = tf.maximum(heatmap, 0)

heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)

heatmap = heatmap.numpy()

# Resize heatmap to image dimensions
heatmap = tf.image.resize(
    heatmap[..., np.newaxis],
    IMG_SIZE
).numpy().squeeze()

# ==========================================
# OVERLAY HEATMAP ON IMAGE
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title("Original Histopathology Image")
plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(image_array)
plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.4
)

probability = float(predictions[0][0])

predicted_class = (
    "Malignant" if probability >= 0.5 else "Benign"
)

plt.title(
    f"Grad-CAM: {predicted_class}\n"
    f"Malignant probability: {probability:.2%}"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "gradcam_malignant.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGrad-CAM completed!")
print("Malignant probability:", round(probability, 4))
print("Predicted class:", predicted_class)
print("Saved as gradcam_malignant.png")
