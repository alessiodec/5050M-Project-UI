import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor
import pickle
import os

def load_cr_model(model_dir):
    """Load the corrosion rate prediction model."""
    class CRModel:
        def __init__(self, model_dir):
            self.dnn_model = load_model(os.path.join(model_dir, "dnn_model.keras"))
            self.catboost_model = CatBoostRegressor()
            self.catboost_model.load_model(os.path.join(model_dir, "catboost_model.cbm"))
            self.catboost_model_5 = CatBoostRegressor()
            self.catboost_model_5.load_model(os.path.join(model_dir, "catboost_model_5.cbm"))
            
            with open(os.path.join(model_dir, "preprocessors.pkl"), "rb") as f:
                preprocessors = pickle.load(f)
                self.scaler_dnn = preprocessors["scaler_dnn"]
                self.scaler_tree = preprocessors["scaler_tree"]
                self.poly = preprocessors["poly"]
                self.weights = preprocessors["weights"]
        
        def predict(self, X):
            X_dnn = X[['pH', 'T', 'PCO2', 'v', 'd']].copy()
            X_tree = X[['pH', 'T', 'PCO2', 'v', 'd']].copy()
            
            for col in ['PCO2', 'v', 'd']:
                X_dnn[col] = np.log10(X_dnn[col])
                X_tree[col] = np.log10(X_tree[col])
            
            X_dnn_scaled = self.scaler_dnn.transform(X_dnn).astype('float32')
            
            X_tree['pH_T_interaction'] = X_tree['pH'] * X_tree['T']
            X_tree['pH_squared'] = X_tree['pH'] ** 2
            X_tree['Re_approx'] = X_tree['v'] * X_tree['d']
            
            poly_features = self.poly.transform(X_tree[['pH', 'T']])
            feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            for i, name in enumerate(feature_names):
                X_tree[name] = poly_features[:, i]
                
            X_tree_scaled = self.scaler_tree.transform(X_tree)
            
            dnn_pred = self.dnn_model.predict(X_dnn_scaled, verbose=0).ravel()
            catboost_pred = self.catboost_model.predict(X_tree_scaled)
            catboost_5_pred = self.catboost_model_5.predict(X_tree_scaled)
            
            return (self.weights['dnn'] * dnn_pred +
                    self.weights['catboost'] * catboost_pred +
                    self.weights['catboost_5'] * catboost_5_pred)
    
    return CRModel(model_dir)