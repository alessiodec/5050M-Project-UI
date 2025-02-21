import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st 

def load_preprocess_data():
    # Import data
    csv_url = f"https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    df = pd.read_csv(csv_url)

    # Select columns to keep
    cols_to_keep = list(range(0, 5)) + [7, 17]  # pH, T, PCO2, v, d, CR SR (original order)
    df_subset = df.iloc[:, cols_to_keep].copy()  # Make new dataframe

    # Apply log10 transformation to selected columns
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])  # log10 PCO2, v, d

    print(df_subset.head())

    # Split and scale data
    X = df_subset.iloc[:, :5].values  # 5 inputs
    y = df_subset.iloc[:, 5:7].values  # 2 outputs

    # Create standard scaler instances
    scaler_X = StandardScaler()
    scaler_X.fit(X)  # Fit the scaler on the dataset

    return df_subset, X, scaler_X
