from tensorflow.keras.models import load_model

def load_models():
    # Load the models
    cr_model = load_model("models/CorrosionRateModel.keras")
    sr_model = load_model("models/SaturationRateModel.keras")
    
    return cr_model, sr_model
