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
from functions.alfie.optimisation_cr import minimise_cr  # CR Minimisation Function

################################### DEFINE APP SECTIONS ###################################

def data_analysis():
    st.title('Data Analysis')

    if st.button('Statistical Analysis'):
        st.session_state.page = 'statistical_analysis'
    if st.button('Contour Plots'):
        st.session_state.page = 'contour_plots'
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def optimisation():
    st.title('Optimisation')

    # Button to minimise Corrosion Rate for given d and PCO₂
    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state.page = 'minimise_cr'
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')

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

        if "pop_size" not in st.session_state:
            st.session_state.pop_size = 1000
        if "pop_retention" not in st.session_state:
            st.session_state.pop_retention = 20

        pop_size = st.number_input(
            "Enter Population Size:", min_value=100, max_value=10000, 
            value=st.session_state.pop_size, step=100
        )
        pop_retention = st.number_input(
            "Enter Population Retention Size:", min_value=10, max_value=1000, 
            value=st.session_state.pop_retention, step=10
        )

        st.session_state.pop_size = int(pop_size)
        st.session_state.pop_retention = int(pop_retention)

        if st.button("Run Heatsink Analysis"):
            try:
                run_heatsink_analysis(st.session_state.pop_size, st.session_state.pop_retention)
                st.write(f"Running analysis with Population Size = {st.session_state.pop_size}, Population Retention = {st.session_state.pop_retention}")
            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def minimise_cr_page():
    """Minimise Corrosion Rate based on user input for d and PCO₂."""
    st.title("Minimise Corrosion Rate (CR)")

    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    data = pd.read_csv(csv_url)

    min_pco2, max_pco2 = data["PCO2"].min(), data["PCO2"].max()
    min_d, max_d = data["d"].min(), data["d"].max()

    d = st.number_input("Enter Pipe Diameter (d):", min_value=min_d, max_value=max_d, step=0.01, value=min_d)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=min_pco2, max_value=max_pco2, step=0.001, value=min_pco2)

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

################################### DATA ANALYSIS PAGE ###################################

def contour_plots():
    st.title('Contour Plots')

    cr_model, sr_model = st.session_state.models
    df_subset, X, scaler_X = st.session_state.data

    if st.button('Corrosion Rate'):
        plot_5x5_cr(X, scaler_X, cr_model)

    if st.button('Saturation Ratio'):
        plot_5x5_sr(X, scaler_X, sr_model)

    if st.button("Go to Home"):
        st.session_state.page = 'main'

def statistical_analysis():
    st.title('Statistical Analysis')

    if 'data' not in st.session_state:
        st.write("Data not found. Please load the data first.")
        return

    df_subset, X, scaler_X = st.session_state.data

    if st.button('PCA Analysis'):
        pca_plot()
    if st.button('Descriptive Statistics'):
        descriptive_analysis(X)
    if st.button('Input Histograms'):
        input_histogram()
    if st.button("Go to Home"):
        st.session_state.page = 'main'

################################### MAIN APP ###################################

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if 'models' not in st.session_state or 'data' not in st.session_state:
        cr_model, sr_model = load_models()
        df_subset, X, scaler_X = load_preprocess_data()
        st.session_state.models = (cr_model, sr_model)
        st.session_state.data = (df_subset, X, scaler_X)

    if st.session_state.page == 'main':
        st.title('Main Menu')

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
