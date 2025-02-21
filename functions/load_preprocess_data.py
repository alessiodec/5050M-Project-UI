import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_preprocess_data():
    # Download data from the provided URL
    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    df = pd.read_csv(csv_url)

    # Select columns to keep; here: first 5 columns and then columns 7 and 17
    cols_to_keep = list(range(0, 5)) + [7, 17]
    df_subset = df.iloc[:, cols_to_keep].copy()

    # Apply log10 transformation to columns 2, 3, and 4 (PCO2, v, d)
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])
    
    print(df_subset.head())  # For debugging

    # Split data into inputs (first 5 columns) and outputs (last 2 columns)
    X = df_subset.iloc[:, :5].values
    y = df_subset.iloc[:, 5:7].values

    # Create and fit the scaler for inputs
    scaler_X = StandardScaler()
    scaler_X.fit(X)

    # Return the full processed dataframe, X, and the scaler
    return df_subset, X, scaler_X
