import streamlit as slt

def descriptive_analysis(X):
    st.write("Descriptive Statistics:")
    st.write(X.describe())  # Display the descriptive statistics in the Streamlit app
