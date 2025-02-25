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
from functions.alfie.optimisation_cr import minimise_cr

################################### DEFINE APP SECTIONS ###################################

def data_analysis():
    st.title('Data Analysis')
    st.write("This section will contain your data analysis logic.")

    if st.button('Statistical Analysis'):
        st.session_state.page = 'statistical_analysis'
    if st.button('Contour Plots'):
        st.session_state.page = 'contour_plots'

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def optimisation():
    st.title('Optimisation')
    st.write("This section will contain your optimisation logic.")

    # Button to minimise Corrosion Rate for given d and PCO₂
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state.page = 'minimise_cr'

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def minimise_cr_page():
    """Minimise Corrosion Rate based on user input for d and PCO₂."""
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter values for pipe diameter (`d`) and CO₂ partial pressure (`PCO₂`) to find the minimum CR.")

    # User inputs
    d = st.number_input("Enter Pipe Diameter (d):", min_value=0.01, max_value=10.0, step=0.01, value=0.5)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=0.001, max_value=10.0, step=0.001, value=20000.0)

    # Convert PCO₂ to log10 scale before optimisation
    pco2_log = np.log10(pco2)

    # Run optimisation when button is clicked
    if st.button("Run Optimisation"):
        try:
            best_params, min_cr = minimise_cr(d, pco2_log)
            
            st.write("✅ **Optimisation Completed!**")
            st.write(f"**Optimal Pipe Diameter (d):** {best_params[0][4]:.3f}")
            st.write(f"**Optimal CO₂ Partial Pressure (PCO₂):** {best_params[0][2]:.3f}")
            st.write(f"**Minimised Corrosion Rate (CR):** {min_cr:.5f}")

        except Exception as e:
            st.error(f"Error running optimisation: {e}")

    if st.button("Go to Optimisation Menu"):
        st.session_state.page = 'optimisation'


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

    elif st.session_state.page == 'minimise_cr':
        minimise_cr_page()

    elif st.session_state.page == 'physical_relationship_analysis':
        physical_relationship_analysis()

if __name__ == "__main__":
    main()
