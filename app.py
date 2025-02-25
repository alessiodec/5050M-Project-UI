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

from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis

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
    
    # Ensure session state exists for heatsink data
    if "heatsink_data" not in st.session_state:
        st.session_state["heatsink_loaded"] = False
        st.session_state["heatsink_data"] = None

    # Button to load heatsink data
    if st.button("Load Heatsink Data"):
        from functions.ethan.load_hs_data import load_heatsink_data
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        
        st.session_state["heatsink_data"] = (df, X, y, standardised_y, mean_y, std_y)
        st.session_state["heatsink_loaded"] = True
        
        st.write("✅ Heatsink data loaded successfully!")
        st.write(df)

    # Show input fields only if the data is loaded
    if st.session_state["heatsink_loaded"]:
        st.write("✅ Heatsink Data is Loaded.")

        # Ensure session state variables exist
        if "pop_size" not in st.session_state:
            st.session_state.pop_size = 1000  # Default
        if "pop_retention" not in st.session_state:
            st.session_state.pop_retention = 20  # Default

        # Get user input and ensure values are **integers**
        pop_size = st.number_input("Enter Population Size:", min_value=100, max_value=10000, value=st.session_state.pop_size, step=100)
        pop_retention = st.number_input("Enter Population Retention Size:", min_value=10, max_value=1000, value=st.session_state.pop_retention, step=10)

        # Update session state **only if values change**
        if pop_size != st.session_state.pop_size:
            st.session_state.pop_size = int(pop_size)
        if pop_retention != st.session_state.pop_retention:
            st.session_state.pop_retention = int(pop_retention)

        # Button to run analysis
        if st.button("Run Heatsink Analysis"):
            from functions.ethan.heatsink_analysis import run_heatsink_analysis
            
            # Explicitly convert to **integer** before calling the function
            run_heatsink_analysis(int(st.session_state.pop_size), int(st.session_state.pop_retention))

    # Go back to home button
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
