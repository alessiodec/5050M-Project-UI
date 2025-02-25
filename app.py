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
    st.write("This section contains optimisation features.")

    # Button to minimise Corrosion Rate for given d and PCO₂
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state.page = 'minimise_cr'

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def minimise_cr_page():
    """Minimise Corrosion Rate based on user input for d and PCO₂."""
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter values for pipe diameter (`d`) and CO₂ partial pressure (`PCO₂`) to find the minimum CR.")

    # Load the dataset to get min/max ranges
    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    data = pd.read_csv(csv_url)

    d_min, d_max = data["d"].min(), data["d"].max()
    pco2_min, pco2_max = data["PCO2"].min(), data["PCO2"].max()

    # User inputs
    d = st.number_input("Enter Pipe Diameter (d):", min_value=d_min, max_value=d_max, step=0.01, value=0.5)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=pco2_min, max_value=pco2_max, step=0.001, value=1000.0)

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

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')
    st.write("This section contains physical relationship analysis.")

    # Ensure session state exists for heatsink data
    if "heatsink_data" not in st.session_state:
        st.session_state.heatsink_loaded = False
        st.session_state.heatsink_data = None

    # Button to load heatsink data
    if st.button("Load Heatsink Data"):
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)

        # Store in session state
        st.session_state.heatsink_data = (df, X, y, standardised_y, mean_y, std_y)
        st.session_state.heatsink_loaded = True

        st.write("✅ Heatsink data loaded successfully!")
        st.write(df)

    # Show input fields only if the data is loaded
    if st.session_state.heatsink_loaded:
        st.write("✅ Heatsink Data is Loaded.")

        # Default session state variables
        if "pop_size" not in st.session_state:
            st.session_state.pop_size = 1000  # Default
        if "pop_retention" not in st.session_state:
            st.session_state.pop_retention = 20  # Default
        if "iterations" not in st.session_state:
            st.session_state.iterations = 10  # Default

        # User inputs
        pop_size = st.number_input("Enter Population Size:", min_value=100, max_value=10000, value=st.session_state.pop_size, step=100)
        pop_retention = st.number_input("Enter Population Retention Size:", min_value=10, max_value=1000, value=st.session_state.pop_retention, step=10)
        iterations = st.number_input("Enter Number of Iterations:", min_value=1, max_value=100, value=st.session_state.iterations, step=1)

        # Store user inputs in session state
        st.session_state.pop_size = int(pop_size)
        st.session_state.pop_retention = int(pop_retention)
        st.session_state.iterations = int(iterations)

        if st.button("Run Heatsink Analysis"):
            try:
                run_heatsink_analysis(st.session_state.pop_size, st.session_state.pop_retention, st.session_state.iterations)
                st.write(f"Running analysis with Population Size = {st.session_state.pop_size}, Retention = {st.session_state.pop_retention}, Iterations = {st.session_state.iterations}")

            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")

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

    elif st.session_state.page == 'minimise_cr':
        minimise_cr_page()

    elif st.session_state.page == 'physical_relationship_analysis':
        physical_relationship_analysis()

if __name__ == "__main__":
    main()
