import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from model import load_custom_plant_model, preprocess_image, CLASS_NAMES

# Page Configuration
st.set_page_config(
    page_title="Custom Plant Species Identification",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Custom Plant & Tree Species Identification")
st.write("Upload an image of a local tree or plant (e.g., Sandalwood, Neem, Teak) to identify it using your trained AI model.")

@st.cache_resource
def get_model():
    return load_custom_plant_model()

model = get_model()

if model is None:
    st.error("⚠️ Trained model file (`plant_model.h5`) not found! Please run `python train.py` after adding your dataset folder.")
else:
    uploaded_file = st.file_uploader("Choose a plant/tree image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.info("Analyzing plant features...")
            processed_img = preprocess_image(image)
            
            # Make Prediction
            predictions = model.predict(processed_img)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)
            
            predicted_species = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
            
            st.success("Analysis Complete!")
            st.metric(label="Predicted Species", value=predicted_species)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")