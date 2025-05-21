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
 Model predicts one of the following five stages of Diabetic Retinopathy:

No_DR: This stage indicates that there are no signs of diabetic retinopathy. The retina appears healthy and unaffected.

Mild: At this stage, a few microaneurysms (tiny bulges in the blood vessels of the retina) are present, which are the earliest signs of damage.

Moderate: There is some blockage in the blood vessels that nourish the retina. This can lead to changes in the retina and possible vision problems.

Severe: A larger number of blood vessels are blocked, leading to significant bleeding and damage within the retina. This stage requires prompt medical attention.

Proliferative_DR: This is the most advanced stage of diabetic retinopathy. Abnormal blood vessels begin to grow on the surface of the retina, which can leak, bleed, or cause the retina to detach, leading to blindness if untreated.







