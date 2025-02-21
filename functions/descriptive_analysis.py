import streamlit as st
import pandas as pd  # Ensure pandas is imported

def descriptive_analysis(X):
    st.write("Descriptive Statistics:")
    st.write("X:")
    
    # Ensure X is a pandas DataFrame, if it's not, convert it
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Now, you can display the first 5 columns (if available)
    # You can use `.iloc` to get the first 5 columns or `columns[:5]` to get the first 5 column names
    X_subset = X.iloc[:, :5]  # Select the first 5 columns for display
    
    # Display descriptive statistics for the first 5 columns
    st.write(X_subset.describe())
