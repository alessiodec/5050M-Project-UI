import streamlit as st
import pandas as pd

def descriptive_analysis(X):
    st.write("Descriptive Statistics:")
    
    # Ensure X is a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # If the DataFrame's columns are numeric (e.g., 0, 1, 2, ...),
    # assign the desired column names.
    if all(isinstance(col, int) for col in X.columns):
        # Define your desired column names
        col_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
        # If X has at least 5 columns, assign these names to the first 5
        if X.shape[1] >= 5:
            X.columns = col_names + list(X.columns[5:])
        else:
            X.columns = col_names[:X.shape[1]]
    
    # Select the first 5 columns for display
    X_subset = X.iloc[:, :5]
    st.write(X_subset.describe())
