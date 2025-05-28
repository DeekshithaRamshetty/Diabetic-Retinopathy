import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns

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
    try:
        return tf.keras.models.load_model(
            r'C:\Users\saiki\OneDrive\Desktop\React-tutorial\my-app\src\major\retinopathy_model(1).h5',
            custom_objects={'FixedDropout': FixedDropout}
        )
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

model = load_model()

# Class labels
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Preprocess image function
def preprocess_image(image):
    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize((224, 224))  # Match your model's expected sizing
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Sidebar Navigation
st.sidebar.title("Navigation")
module = st.sidebar.radio("Go to", ("Prediction", "Education", "Visualization"))

# Custom CSS
st.markdown("""
    <style>
    .header {
        font-size: 30px;
        color: #025e73;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 22px;
        color: #014f86;
        font-weight: 600;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Module: Prediction (Updated to address "No DR" issue)
if module == "Prediction":
    st.title("👁 Diabetic Retinopathy Detection")
    st.write("Upload an eye fundus image to check for diabetic retinopathy")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if st.button('Detect'):
            with st.spinner('Analyzing...'):
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)

                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    if processed_image is None:
                        st.error("Failed to preprocess the image. Please try a different image.")
                        st.stop()

                    # Make prediction
                    predictions = model.predict(processed_image)
                    if predictions.shape[1] != len(CLASS_NAMES):
                        st.error("Model output does not match expected class labels.")
                        st.stop()

                    # Debug: Show raw prediction probabilities
                    st.write("Raw prediction probabilities (for debugging):")
                    for i, prob in enumerate(predictions[0]):
                        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100

                    # Display results
                    st.subheader("Results")
                    st.write(f"Prediction: {CLASS_NAMES[predicted_class]}")
                    st.write(f"Confidence: {confidence:.2f}%")

                    # Add confidence threshold
                    if confidence < 50:
                        st.warning("⚠ Low confidence in prediction. The result may not be reliable. Please consult a doctor for a professional diagnosis.")

                    # Warn if prediction is consistently "No DR"
                    if predicted_class == 0:
                        st.success("No signs of diabetic retinopathy detected")
                        st.info("Note: If all images are predicted as 'No DR,' the model might be biased or the images may not be suitable for DR detection. Try using a known DR-positive image to test.")
                    elif predicted_class in [1, 2]:
                        st.warning("Early signs of diabetic retinopathy detected")
                    else:
                        st.error("Advanced diabetic retinopathy detected - Please consult a doctor")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Module: Education (Unchanged, using Streamlit components)
elif module == "Education":
    st.markdown('<div class="header">📘 Learn About Diabetic Retinopathy</div>', unsafe_allow_html=True)

    st.markdown("### 📈 Stages of Diabetic Retinopathy (DR)")
    st.write("- *No DR:* No signs of retinopathy.")
    st.write("- *Mild DR:* Small balloon-like swelling in retina blood vessels.")
    st.write("- *Moderate DR:* Some blood vessel blockages.")
    st.write("- *Severe DR:* Extensive blockages reduce retinal oxygen supply.")
    st.write("- *Proliferative DR:* Advanced stage with abnormal blood vessel growth and possible bleeding.")

    st.markdown("### ⚠ Risk Factors")
    st.write("- Long-term diabetes (Type 1 or Type 2)")
    st.write("- High blood sugar levels (poor glucose control)")
    st.write("- High blood pressure and cholesterol")
    st.write("- Smoking")
    st.write("- Pregnancy in diabetic individuals")

    st.markdown("### 👁 Symptoms to Watch")
    st.write("- Blurred or fluctuating vision")
    st.write("- Dark spots or floaters in vision")
    st.write("- Difficulty seeing at night")
    st.write("- Sudden vision loss (in severe cases)")

    st.markdown("### 💊 Treatment Options")
    st.write("- Blood sugar, blood pressure, and cholesterol control")
    st.write("- Laser therapy to stop or slow leakage")
    st.write("- Anti-VEGF injections to reduce swelling")
    st.write("- Vitrectomy surgery for severe bleeding")
    st.write("- Routine eye exams for early detection")

    st.markdown("### ✅ Prevention Tips")
    st.write("- Maintain a healthy lifestyle and balanced diet")
    st.write("- Exercise regularly")
    st.write("- Quit smoking")
    st.write("- Schedule annual dilated eye exams")

# Module: Visualization (Unchanged)
elif module == "Visualization":
    st.markdown('<div class="header">📊 Visualization</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Confusion Matrix")
        cm = np.array([
            [30, 2, 0, 0, 0],
            [1, 25, 3, 0, 0],
            [0, 2, 20, 2, 0],
            [0, 0, 1, 15, 1],
            [0, 0, 0, 2, 10]
        ])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt.gcf())
        plt.close()

    with col2:
        st.subheader("Training Accuracy & Loss")
        epochs = list(range(1, 11))
        accuracy = [0.60, 0.65, 0.70, 0.74, 0.78, 0.80, 0.83, 0.86, 0.88, 0.90]
        loss = [1.2, 1.0, 0.85, 0.75, 0.65, 0.55, 0.45, 0.40, 0.35, 0.30]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs, accuracy, marker='o', color='green')
        ax[0].set_title("Training Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(epochs, loss, marker='o', color='red')
        ax[1].set_title("Training Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")

        st.pyplot(fig)
