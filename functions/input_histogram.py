import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

def input_histogram():
    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    df = pd.read_csv(csv_url)
    
    cols_to_keep = list(range(0, 5))  # pH, T, PCO2, v, d
    df_subset = df.iloc[:, cols_to_keep].copy()  # Make a new DataFrame with the selected columns
    
    # Replace 0s with NaN and apply log10 transformation on columns 2, 3, and 4
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]].replace(0, np.nan))
    
    # Create a new figure for the histograms
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each column in the DataFrame
    df_subset.hist(bins=30)
    plt.suptitle("Histograms of Inputs", y=0.95)
    plt.tight_layout()
    
    # Display the current figure in Streamlit
    st.pyplot(plt.gcf())
    plt.close()  # Close the figure to prevent overlap with future plots
