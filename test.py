import streamlit as st
import tensorflow as tf
import urllib.request
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Corrosion Contour Plots")

# Define GitHub raw file URL
model_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/alessio_test_1.h5"
model_path = "alessio_test_1.h5"

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

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
        st.session_state.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading trained model: {e}")

# Function to generate contour plots
def plot_contours():
    if st.session_state.model is None:
        st.error("Please load the model first!")
        return
    
    # Load dataset from Google Drive
    file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return
    
    # Select relevant columns
    cols_to_keep = list(range(0, 5)) + [7, 17]
    df_subset = df.iloc[:, cols_to_keep].copy()
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])
    
    # Prepare Data for Model
    X = df_subset.iloc[:, :5].values  # Inputs
    y = df_subset.iloc[:, 5:7].values  # Outputs (CR, SR)
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Generate Contour Plots
    var_names = ['pH', 'T', 'PCO2', 'v', 'd']
    mid_points = np.median(X, axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
    
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            x_vals = np.linspace(mins[j], maxs[j], 25)
            y_vals = np.linspace(mins[i], maxs[i], 25)
            grid_x, grid_y = np.meshgrid(x_vals, y_vals)
            
            grid_points = np.tile(mid_points, (grid_x.size, 1))
            grid_points[:, j] = grid_x.ravel()
            grid_points[:, i] = grid_y.ravel()
            
            grid_points_scaled = scaler_X.transform(grid_points)
            predictions_scaled = st.session_state.model.predict(grid_points_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled)
            corrosion_rate = predictions[:, 0].reshape(grid_x.shape)
            
            cont_fill = ax.contourf(grid_x, grid_y, corrosion_rate, levels=10, cmap='viridis')
            cont_line = ax.contour(grid_x, grid_y, corrosion_rate, levels=10, colors='black', linewidths=0.5)
            ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
            ax.set_xlabel(var_names[j])
            ax.set_ylabel(var_names[i])
    
    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont_fill, cax=cbar_ax, label='Corrosion Rate')
    plt.suptitle('CR For Different Input Combinations', fontsize=18)
    
    st.pyplot(fig)

# Buttons to trigger model loading and plotting
if st.button("Load Model"):
    load_model()

if st.button("Generate Contour Plots"):
    plot_contours()
