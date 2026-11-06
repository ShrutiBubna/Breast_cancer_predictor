import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained MobileNetV2 model
MODEL_PATH = "saved_models/best_mobilenetv2.keras"

model = load_model(MODEL_PATH)

# Select a test image
IMAGE_PATH = r"dataset_cancer_v1\classificacao_binaria\40X\malignant\SOB_M_DC-14-12312-40-001.png"# Load and preprocess image
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)

input_array = np.expand_dims(img_array.copy(), axis=0)
input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    input_array
)

# Find the last convolutional layer in MobileNetV2
base_model = model.get_layer("mobilenetv2_1.00_224")
print("\nMODEL LAYERS:")
for layer in model.layers:
    print(layer.name, layer.output.shape)

print("\nMOBILENETV2 LAST LAYERS:")
for layer in base_model.layers[-5:]:
    print(layer.name, layer.output.shape)
last_conv_layer = base_model.get_layer("out_relu")

# Create a model to extract the last convolutional features
grad_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, base_model.output]
)

# Calculate Grad-CAM
with tf.GradientTape() as tape:

    conv_outputs, base_outputs = grad_model(
        input_array, training=False
    )

    # Pass extracted features through the existing classifier
    x = model.get_layer("global_average_pooling2d")(base_outputs)
    x = model.get_layer("dropout")(x, training=False)
    x = model.get_layer("dense")(x)
    x = model.get_layer("dropout_1")(x, training=False)
    predictions = model.get_layer("dense_1")(x)

    loss = predictions[:, 0]

# Calculate gradients
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

# Overlay heatmap on original image
original = cv2.imread(IMAGE_PATH)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original, (224, 224))

heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(
    heatmap, cv2.COLORMAP_JET
)
heatmap_color = cv2.cvtColor(
    heatmap_color, cv2.COLOR_BGR2RGB
)

overlay = cv2.addWeighted(
    original, 0.6, heatmap_color, 0.4, 0
)

# Display results
probability = float(predictions[0][0])
prediction = "Malignant" if probability >= 0.5 else "Benign"

print("Predicted class:", prediction)
print("Malignant probability:", round(probability, 4))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_malignant.png", dpi=300)
plt.show()