# data_analysis.py
import streamlit as st
from functions.pca_plot import pca_plot
from functions.descriptive_analysis import descriptive_analysis
from functions.input_histogram import input_histogram
from functions.plot_5x5_cr import plot_5x5_cr
from functions.plot_5x5_sr import plot_5x5_sr

def main():
    st.title('Data Analysis')
    st.write("Perform various analyses on the dataset.")

    # For example, offer two buttons for sub‑sections.
    if st.button('Statistical Analysis'):
        st.write("Performing Statistical Analysis...")
        # Check if data exists in session state
        if 'data' not in st.session_state:
            st.write("Data not found. Please load the data first.")
        else:
            _, X, _ = st.session_state.data
            pca_plot()
            descriptive_analysis(X)
            input_histogram()
    if st.button('Contour Plots'):
        st.write("Generating Contour Plots...")
        if 'models' not in st.session_state or 'data' not in st.session_state:
            st.write("Models or data not loaded.")
        else:
            cr_model, sr_model = st.session_state.models
            _, X, scaler_X = st.session_state.data
            plot_5x5_cr(X, scaler_X, cr_model)
            plot_5x5_sr(X, scaler_X, sr_model)

if __name__ == "__main__":
    main()
