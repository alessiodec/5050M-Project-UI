import streamlit as st
import pandas as pd

def descriptive_analysis(X):
    st.write("Descriptive Statistics:")
    st.write(X)
    
    # Ensure X is a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Select the first 5 columns (if available) for display
    X_subset = X.iloc[:, :5]
    
    st.write(X_subset.describe())
