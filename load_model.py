import tensorflow as tf
import urllib.request
import os

MODEL_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/alessio_test_1.h5"
MODEL_PATH = "alessio_test_1.h5"

def load_trained_model():
    """Downloads and loads the trained model from GitHub."""
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            return None, f"Error downloading model: {e}"

    try:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        return model, None  # No errors
    except Exception as e:
        return None, f"Error loading trained model: {e}"
