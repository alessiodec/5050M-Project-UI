import streamlit as st
import tensorflow as tf
import urllib.request
import os

st.title("Model Loader")

# Define GitHub raw file URL
model_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/alessio_test_1.h5"
model_path = "alessio_test_1.h5"

# Function to download and load the model
def load_model():
    if not os.path.exists(model_path):
        try:
            urllib.request.urlretrieve(model_url, model_path)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return
    
    try:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading trained model: {e}")

# Button to trigger model loading
if st.button("Load Model"):
    load_model()
