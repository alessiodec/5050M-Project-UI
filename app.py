################################### IMPORT LIBRARIES & FUNCTIONS ###################################

import streamlit as st
import pandas as pd
import numpy as np

from functions.load_models import load_models  # Load the models
from functions.load_preprocess_data import load_preprocess_data  # Load & preprocess data
from functions.plot_5x5_cr import plot_5x5_cr  # Plot corrosion rate contours
from functions.plot_5x5_sr import plot_5x5_sr  # Plot saturation ratio contours
from functions.pca_plot import pca_plot  # Plot PCA results
from functions.descriptive_analysis import descriptive_analysis  # Show descriptive stats
from functions.input_histogram import input_histogram  # Display input histograms

################################### DEFINE APP SECTIONS ###################################

def data_analysis():
    st.title('Data Analysis')
    st.write("This section will contain your data analysis logic.")

    # Create buttons for statistical analysis and contour plots
    statistical_analysis_button = st.button('Statistical Analysis')
    contour_button = st.button('Contour Plots')

    if statistical_analysis_button:
        st.session_state.page = 'statistical_analysis'
    if contour_button:
        st.session_state.page = 'contour_plots'

    # Home button
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def optimisation():
    st.title('Optimisation')
    st.write("This section will contain your optimisation logic.")
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')
    st.write("This section will contain your physical relationship analysis logic.")
    
    # New button for Heatsink Analysis
    if st.button("Heatsink Analysis"):
        # Import and call the load_heatsink_data function from functions/ethan/load_hs_data.py
        from functions.ethan.load_hs_data import load_heatsink_data
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        st.write("Heatsink data loaded successfully!")
        st.write(df)

    if st.button("Go to Home"):
        st.session_state.page = 'main'

################################### DATA ANALYSIS PAGE ###################################

def contour_plots():
    st.title('Contour Plots')
    st.write("Choose the plot to display:")

    # Retrieve preloaded models and data (now three values: df_subset, X, scaler_X)
    cr_model, sr_model = st.session_state.models
    df_subset, X, scaler_X = st.session_state.data

    # Button for Corrosion Rate contour plot
    if st.button('Corrosion Rate'):
        st.write("Generating Corrosion Rate Contour Plot...")
        plot_5x5_cr(X, scaler_X, cr_model)

    # Button for Saturation Ratio contour plot
    if st.button('Saturation Ratio'):
        st.write("Generating Saturation Ratio Contour Plot...")
        plot_5x5_sr(X, scaler_X, sr_model)

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def statistical_analysis():
    st.title('Statistical Analysis')
    st.write("This section will contain the statistical analysis logic.")

    # Retrieve data; ensure it's available
    if 'data' not in st.session_state:
        st.write("Data not found. Please load the data first.")
        return

    df_subset, X, scaler_X = st.session_state.data

    if st.button('PCA Analysis'):
        st.write("Performing PCA Analysis...")
        pca_plot()

    if st.button('Descriptive Statistics'):
        st.write("Generating Statistical Description...")
        descriptive_analysis(X)

    if st.button('Input Histograms'):
        st.write("Generating Input Histograms...")
        input_histogram()

    if st.button("Go to Home"):
        st.session_state.page = 'main'

################################### MAIN APP ###################################

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Default page

    # Load models and data if not already loaded
    if 'models' not in st.session_state or 'data' not in st.session_state:
        cr_model, sr_model = load_models()
        # load_preprocess_data now returns three values
        df_subset, X, scaler_X = load_preprocess_data()
        st.session_state.models = (cr_model, sr_model)
        st.session_state.data = (df_subset, X, scaler_X)

    # Navigation based on st.session_state.page
    if st.session_state.page == 'main':
        st.title('Main Menu')
        st.write("Select an option to proceed:")

        if st.button('Data Analysis'):
            st.session_state.page = 'data_analysis'
        elif st.button('Optimisation'):
            st.session_state.page = 'optimisation'
        elif st.button('Physical Relationship Analysis'):
            st.session_state.page = 'physical_relationship_analysis'

    elif st.session_state.page == 'data_analysis':
        data_analysis()

    elif st.session_state.page == 'statistical_analysis':
        statistical_analysis()

    elif st.session_state.page == 'contour_plots':
        contour_plots()

    elif st.session_state.page == 'optimisation':
        optimisation()

    elif st.session_state.page == 'physical_relationship_analysis':
        physical_relationship_analysis()

if __name__ == "__main__":
    main()
