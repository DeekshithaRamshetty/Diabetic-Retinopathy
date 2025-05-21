# Diabetic-Retinopathy
🧠 Diabetic Retinopathy Classification using Deep Learning
📌 Project Overview
This project is a deep learning-based web application to detect and classify Diabetic Retinopathy (DR) from retinal fundus images. The model is trained using a preprocessed dataset of retinal images and then deployed as a web app using Streamlit.

We used:

EfficientNetB0 with imagenet weights for transfer learning

ImageDataGenerator for data augmentation and preprocessing

Keras/TensorFlow for building and training the model

Streamlit for creating a user-friendly interface

H5 model saving to reuse the model without retraining

👁️ What is Diabetic Retinopathy?
Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). Early detection is crucial to prevent vision loss or blindness.

🔬 Stages of Diabetic Retinopathy
Your model predicts one of the following five stages of DR:

Stage	Description
No_DR	No signs of diabetic retinopathy. Retina appears healthy.
Mild	Presence of a few microaneurysms (tiny bulges in blood vessels).
Moderate	Blockage in some blood vessels that nourish the retina.
Severe	A larger number of blocked blood vessels and more bleeding.
Proliferative_DR	Most advanced stage with growth of abnormal blood vessels, which can lead to retinal detachment or blindness.
