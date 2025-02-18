import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor
import pickle
import os
import gdown  # Used to download files from Google Drive

def download_if_needed(url, output_path):
    """
    Download the file from the given URL if it doesn't exist locally.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} ...")
        gdown.download(url, output_path, quiet=False)
    return output_path

def load_cr_model():
    """Load the corrosion rate prediction model from Google Drive links."""
    class CRModel:
        def __init__(self):
            # Google Drive URLs (converted to direct download links using the 'uc?id=' format)
            dnn_model_url = "https://drive.google.com/uc?id=17ti0wPhRKGUsybP5Ev_-cIdTIfIJxT9x"
            catboost_model_url = "https://drive.google.com/uc?id=1Iw4083efHgsnJZiPrHWCSyKBExIqmVW8"
            catboost_model_5_url = "https://drive.google.com/uc?id=1nzUa6Wu7cML6_dip26MvSs40JydJ33Go"
            preprocessors_url = "https://drive.google.com/uc?id=1PwC1v562wBZYgJR_shK1gnQsb20Oymaz"
            
            # Define local file names to store the downloaded models and preprocessors
            dnn_model_path = "dnn_model.keras"
            catboost_model_path = "catboost_model.cbm"
            catboost_model_5_path = "catboost_model_5.cbm"
            preprocessors_path = "preprocessors.pkl"
            
            # Download files if they do not exist locally
            download_if_needed(dnn_model_url, dnn_model_path)
            download_if_needed(catboost_model_url, catboost_model_path)
            download_if_needed(catboost_model_5_url, catboost_model_5_path)
            download_if_needed(preprocessors_url, preprocessors_path)
            
            # Load the deep neural network model
            self.dnn_model = load_model(dnn_model_path)
            
            # Load the first CatBoost model
            self.catboost_model = CatBoostRegressor()
            self.catboost_model.load_model(catboost_model_path)
            
            # Load the second CatBoost model
            self.catboost_model_5 = CatBoostRegressor()
            self.catboost_model_5.load_model(catboost_model_5_path)
            
            # Load preprocessing objects (scalers, polynomial feature generator, weights, etc.)
            with open(preprocessors_path, "rb") as f:
                preprocessors = pickle.load(f)
                self.scaler_dnn = preprocessors["scaler_dnn"]
                self.scaler_tree = preprocessors["scaler_tree"]
                self.poly = preprocessors["poly"]
                self.weights = preprocessors["weights"]
        
        def predict(self, X):
            """
            Predict corrosion rate based on the input DataFrame `X` using a blended model.
            """
            # Prepare copies for the DNN and tree-based models
            X_dnn = X[['pH', 'T', 'PCO2', 'v', 'd']].copy()
            X_tree = X[['pH', 'T', 'PCO2', 'v', 'd']].copy()
            
            # Apply logarithmic transformation to select features
            for col in ['PCO2', 'v', 'd']:
                X_dnn[col] = np.log10(X_dnn[col])
                X_tree[col] = np.log10(X_tree[col])
            
            # Scale data for the DNN model
            X_dnn_scaled = self.scaler_dnn.transform(X_dnn).astype('float32')
            
            # Generate additional features for the tree-based models
            X_tree['pH_T_interaction'] = X_tree['pH'] * X_tree['T']
            X_tree['pH_squared'] = X_tree['pH'] ** 2
            X_tree['Re_approx'] = X_tree['v'] * X_tree['d']
            
            # Create polynomial features for the 'pH' and 'T' columns
            poly_features = self.poly.transform(X_tree[['pH', 'T']])
            feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            for i, name in enumerate(feature_names):
                X_tree[name] = poly_features[:, i]
                
            # Scale data for the tree-based models
            X_tree_scaled = self.scaler_tree.transform(X_tree)
            
            # Get predictions from each model
            dnn_pred = self.dnn_model.predict(X_dnn_scaled, verbose=0).ravel()
            catboost_pred = self.catboost_model.predict(X_tree_scaled)
            catboost_5_pred = self.catboost_model_5.predict(X_tree_scaled)
            
            # Return the weighted ensemble prediction
            return (self.weights['dnn'] * dnn_pred +
                    self.weights['catboost'] * catboost_pred +
                    self.weights['catboost_5'] * catboost_5_pred)
    
    return CRModel()

# For example usage:
if __name__ == "__main__":
    model = load_cr_model()
    # Example input DataFrame with appropriate columns
    df = pd.DataFrame({
        "pH": [7.0],
        "T": [25.0],
        "PCO2": [0.03],
        "v": [1.2],
        "d": [0.5]
    })
    prediction = model.predict(df)
    print("Prediction:", prediction)
