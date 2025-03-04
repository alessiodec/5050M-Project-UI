import os
import streamlit as st  # Only needed if you want to display using st.write
import numpy as np
import pandas as pd
import warnings

from . import Engine
from . import config

# Get absolute path to the data file
def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def load_heatsink_data(file_path=None, display_output=False):
    """
    Loads and processes the heatsink dataset.
    
    Parameters:
        file_path (str): Path to the dataset. If None, defaults to file in script directory.
        display_output (bool): If True, display mean, std, and DataFrame via st.write.
        
    Returns:
        df (DataFrame): The processed DataFrame.
        X (ndarray): Feature array from columns 'Geometric1' and 'Geometric2'.
        y (ndarray): Target variable array from column 'Pressure_Drop'.
        standardised_y (ndarray): The standardized target variable.
        mean_y (float): Mean of y.
        std_y (float): Standard deviation of y.
    """
    # Default file path (inside the same folder as this script)
    if file_path is None:
        file_path = get_data_path("Latin_Hypercube_Heatsink_1000_samples.txt")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the file is correctly placed.")

    # Read the file using a context manager
    with open(file_path, "r") as f:
        text = f.read()
    
    # Split the text into rows and then each row into columns
    data = [x.split(' ') for x in text.split('\n') if x.strip() != '']
    
    # Create DataFrame with proper column names and convert to numeric types
    df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
    df = df.apply(pd.to_numeric)
    
    # Extract features and target
    X = df[['Geometric1', 'Geometric2']].values
    y = df['Pressure_Drop'].values.reshape(-1,)  # Using Pressure_Drop as target
    
    # Compute mean and standard deviation of y
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    # Update config values
    config.mean_y = mean_y
    config.std_y = std_y
    
    # Optionally display the computed values and DataFrame in the Streamlit app
    if display_output:
        st.write("Mean of y:", mean_y)
        st.write("Standard deviation of y:", std_y)
        st.write("DataFrame:", df)
    
    # Standardize y
    standardised_y = (y - mean_y) / std_y
    
    return df, X, y, standardised_y, mean_y, std_y
