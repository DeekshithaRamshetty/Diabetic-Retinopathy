import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout

# Define the custom layer
class FixedDropout(Dropout):
    def _init_(self, rate, **kwargs):
        super(FixedDropout, self)._init_(rate, **kwargs)

# Set page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="👁",
    layout="centered"
)

# Load your saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r'C:\Users\ravit\Desktop\Bootcamp\retinopathy_model.h5',
        custom_objects={'FixedDropout': FixedDropout}
    )

model = load_model()

# Class labels (modify according to your model)
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Preprocess image function
def preprocess_image(image):
    img = image.resize((224, 224))  # Match your model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("👁 Diabetic Retinopathy Detection")
st.write("Upload an eye fundus image to check for diabetic retinopathy")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button('Detect'):
        with st.spinner('Analyzing...'):  
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            st.subheader("Results")
            st.write(f"Prediction: {CLASS_NAMES[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}%")

            if predicted_class == 0:
                st.success("No signs of diabetic retinopathy detected")
            elif predicted_class in [1, 2]:
                st.warning("Early signs of diabetic retinopathy detected")
            else:
                st.error("Advanced diabetic retinopathy detected - Please consult a doctor")
