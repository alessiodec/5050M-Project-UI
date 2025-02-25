################################### IMPORT LIBRARIES & FUNCTIONS ###################################

import streamlit as st
import pandas as pd
import numpy as np

from functions.load_models import load_models            # Load the models
from functions.load_preprocess_data import load_preprocess_data  # Load & preprocess data
from functions.plot_5x5_cr import plot_5x5_cr              # Plot corrosion rate contours
from functions.plot_5x5_sr import plot_5x5_sr              # Plot saturation ratio contours
from functions.pca_plot import pca_plot                    # Plot PCA results
from functions.descriptive_analysis import descriptive_analysis  # Show descriptive stats
from functions.input_histogram import input_histogram      # Display input histograms

from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis
from functions.alfie.optimisation_cr import minimise_cr   # CR Minimisation Function

################################### DEFINE APP SECTIONS ###################################

def data_analysis():
    st.title('Data Analysis')
    st.write("Perform various analyses on the dataset.")

    if st.button('Statistical Analysis'):
        st.session_state.page = 'statistical_analysis'
    if st.button('Contour Plots'):
        st.session_state.page = 'contour_plots'
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def contour_plots():
    st.title('Contour Plots')
    st.write("Choose the plot to display:")

    cr_model, sr_model = st.session_state.models
    df_subset, X, scaler_X = st.session_state.data

    if st.button('Corrosion Rate'):
        st.write("Generating Corrosion Rate Contour Plot...")
        plot_5x5_cr(X, scaler_X, cr_model)
    if st.button('Saturation Ratio'):
        st.write("Generating Saturation Ratio Contour Plot...")
        plot_5x5_sr(X, scaler_X, sr_model)
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def statistical_analysis():
    st.title('Statistical Analysis')
    st.write("Analyze the dataset using PCA, descriptive statistics, and histograms.")

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

def optimisation():
    st.title('Optimisation')
    st.write("This section contains optimisation features.")

    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state.page = 'minimise_cr'
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter values for pipe diameter (d) and CO₂ partial pressure (PCO₂) to minimize CR.")

    # Load dataset to extract min and max values for d and PCO₂
    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    data = pd.read_csv(csv_url)

    d_min, d_max = data["d"].min(), data["d"].max()
    pco2_min, pco2_max = data["PCO2"].min(), data["PCO2"].max()

    d = st.number_input("Enter Pipe Diameter (d):", min_value=d_min, max_value=d_max, step=0.01, value=d_min)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=pco2_min, max_value=pco2_max, step=0.001, value=pco2_min)

    # Convert PCO₂ to log10 scale as required by the optimisation function
    pco2_log = np.log10(pco2)

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
    st.write("This section includes heatsink analysis and related evaluations.")

    if "heatsink_data" not in st.session_state:
        st.session_state.heatsink_loaded = False
        st.session_state.heatsink_data = None

    if st.button("Load Heatsink Data"):
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        st.session_state.heatsink_data = (df, X, y, standardised_y, mean_y, std_y)
        st.session_state.heatsink_loaded = True
        st.write("✅ Heatsink data loaded successfully!")
        st.write(df)

    if st.session_state.heatsink_loaded:
        st.write("✅ Heatsink Data is Loaded.")

        pop_size = st.number_input("Enter Population Size:", min_value=100, max_value=10000, value=1000, step=100)
        pop_retention = st.number_input("Enter Population Retention Size:", min_value=10, max_value=1000, value=20, step=10)
        iterations = st.number_input("Enter Number of Iterations:", min_value=1, max_value=100, value=10, step=1)

        if st.button("Run Heatsink Analysis"):
            try:
                run_heatsink_analysis(int(pop_size), int(pop_retention), int(iterations))
                st.write(f"Running analysis with Population Size = {pop_size}, Retention = {pop_retention}, Iterations = {iterations}")
            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")

    if st.button("Go to Home"):
        st.session_state.page = 'main'

################################### MAIN APP ###################################

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    # Load models and preprocessed data if not already loaded
    if 'models' not in st.session_state or 'data' not in st.session_state:
        cr_model, sr_model = load_models()
        df_subset, X, scaler_X = load_preprocess_data()
        st.session_state.models = (cr_model, sr_model)
        st.session_state.data = (df_subset, X, scaler_X)

    # Navigation based on session state
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
