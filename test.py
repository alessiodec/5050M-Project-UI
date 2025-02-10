import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Initialize the page state if it doesn't exist.
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def show_home():
    """Display the main menu with buttons for each page."""
    st.title("Main Menu")
    st.write("Choose a page:")
    if st.button("Input / Output Relationship Analysis"):
        st.session_state.page = 'page1'
    if st.button("Page 2 - Contour Plots"):
        st.session_state.page = 'page2'

def show_page1():
    """Display content for Page 1."""
    st.title("Page 1")
    st.write("Welcome to Page 1!")
    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_page2():
    """Display content for Page 2: Load data, preprocess, and generate contour plots."""
    st.title("Contour Plots of Corrosion Rate (CR)")
    
    # Load CSV from Google Drive
    file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return
    
    # Select relevant columns
    cols_to_keep = list(range(0, 5)) + [7, 17]  # pH, T, PCO2, v, d, CR SR
    df_subset = df.iloc[:, cols_to_keep].copy()
    
    # Log transform certain columns
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])
    
    # Display DataFrame Head
    st.write("First five rows of the dataset:")
    st.write(df_subset.head())
    
    # Prepare Data for Model
    X = df_subset.iloc[:, :5].values  # Inputs
    y = df_subset.iloc[:, 5:7].values  # Outputs (CR, SR)
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Load trained model from GitHub repo
    try:
        model = tf.keras.models.load_model("alessio_test_1.h5")  # Ensure you have alessio_test_1.h5 in the repo
    except Exception as e:
        st.error("Error loading trained model.")
        return
    
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
            predictions_scaled = model.predict(grid_points_scaled)
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
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'

# Render the appropriate page
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'page1':
    show_page1()
elif st.session_state.page == 'page2':
    show_page2()
