import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "PlantDiseaseModel.keras")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Verify model and class indices paths
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

if not os.path.exists(class_indices_path):
    st.error(f"Class indices file not found at {class_indices_path}")
    st.stop()

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names and convert keys to integers
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}  # Ensure integer keys

# Function to load and preprocess the image
def load_and_preprocess_image(uploaded_img, target_size=(224, 224)):
    img = Image.open(uploaded_img)
    img = img.resize(target_size)
    img_array = np.array(img)
    if len(img_array.shape) == 2:  # Handle grayscale images
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0  # Normalize
    return img_array

# Prediction function
def predict_image_class(model, image, class_indices):
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    st.write(f"Predicted class index: {predicted_class_idx}")  # Debugging
    if predicted_class_idx not in class_indices:
        raise ValueError(f"Predicted index {predicted_class_idx} not found in class indices.")
    return class_indices[predicted_class_idx]

# Streamlit App
st.title("Plant Disease Classifier / Prediction")

uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = img.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Classify"):
            try:
                # Preprocess the image
                preprocessed_image = load_and_preprocess_image(uploaded_img)
                st.write(f"Preprocessed image shape: {preprocessed_image.shape}")

                # Debug: Display class indices
                # st.write(f"Class indices: {class_indices}")

                # Make prediction
                prediction = predict_image_class(model, preprocessed_image, class_indices)
                st.success(f"Prediction: {str(prediction)}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
