################################### IMPORT LIBRARIES & FUNCTIONS ###################################

import streamlit as st
import pandas as pd
import numpy as np

from functions.load_models import load_models             # Load models
from functions.load_preprocess_data import load_preprocess_data   # Preprocess data
from functions.plot_5x5_cr import plot_5x5_cr               # CR contour plots
from functions.plot_5x5_sr import plot_5x5_sr               # SR contour plots
from functions.pca_plot import pca_plot                     # PCA plots
from functions.descriptive_analysis import descriptive_analysis   # Descriptive stats
from functions.input_histogram import input_histogram       # Histograms

from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis
from functions.alfie.optimisation_cr import minimise_cr      # CR minimisation function

################################### DEFINE APP SECTIONS ###################################

# --- Data Analysis Section ---

def data_analysis_home():
    st.title('Data Analysis')
    st.write("Perform various analyses on the dataset.")
    
    if st.button('Statistical Analysis'):
        st.session_state.sub_page = 'statistical_analysis'
        return
    if st.button('Contour Plots'):
        st.session_state.sub_page = 'contour_plots'
        return

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
        st.session_state.sub_page = None
        return

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
        st.session_state.sub_page = None
        return

# --- Optimisation Section ---

def optimisation_home():
    st.title('Optimisation')
    st.write("This section contains optimisation features.")

    if st.button("Minimise CR for Given d and PCO₂"):
        st.session_state.sub_page = 'minimise_cr'
        return
    if st.button("Go to Home"):
        st.session_state.sub_page = None
        return

def minimise_cr_page():
    st.title("Minimise Corrosion Rate (CR)")
    st.write("Enter values for pipe diameter (d) and CO₂ partial pressure (PCO₂) to find the minimum CR.")

    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    data = pd.read_csv(csv_url)

    d_min, d_max = data["d"].min(), data["d"].max()
    pco2_min, pco2_max = data["PCO2"].min(), data["PCO2"].max()

    d = st.number_input("Enter Pipe Diameter (d):", min_value=d_min, max_value=d_max, step=0.01, value=d_min)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=pco2_min, max_value=pco2_max, step=0.001, value=pco2_min)

    if st.button("Run Optimisation"):
        try:
            best_params, min_cr = minimise_cr(d, pco2)
            st.write("✅ **Optimisation Completed!**")
            st.write(f"**Optimal Pipe Diameter (d):** {best_params[0][4]:.3f}")
            st.write(f"**Optimal CO₂ Partial Pressure (PCO₂):** {best_params[0][2]:.3f}")
            st.write(f"**Minimised Corrosion Rate (CR):** {min_cr:.5f}")
        except Exception as e:
            st.error(f"Error running optimisation: {e}")
    if st.button("Go to Optimisation Menu"):
        st.session_state.sub_page = None
        return

# --- Physical Relationship Analysis Section ---

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
        # For this section, a "Go to Home" simply keeps you on the same page.
        return

################################### MAIN APP ###################################

def main():
    # Initialize main navigation state if not set.
    if 'main_tab' not in st.session_state:
        st.session_state.main_tab = "Data Analysis"
    if 'sub_page' not in st.session_state:
        st.session_state.sub_page = None

    # Load models and data if not already present.
    if 'models' not in st.session_state or 'data' not in st.session_state:
        cr_model, sr_model = load_models()
        df_subset, X, scaler_X = load_preprocess_data()
        st.session_state.models = (cr_model, sr_model)
        st.session_state.data = (df_subset, X, scaler_X)

    # Sidebar navigation: the radio button is the single source of truth.
    options = ["Data Analysis", "Optimisation", "Physical Relationship Analysis"]
    selected_tab = st.sidebar.radio("Navigation", options, index=options.index(st.session_state.main_tab))
    st.session_state.main_tab = selected_tab

    # Render page content based on main_tab and sub_page.
    if st.session_state.main_tab == "Data Analysis":
        if st.session_state.sub_page is None:
            data_analysis_home()
        elif st.session_state.sub_page == 'statistical_analysis':
            statistical_analysis()
        elif st.session_state.sub_page == 'contour_plots':
            contour_plots()
    elif st.session_state.main_tab == "Optimisation":
        if st.session_state.sub_page is None:
            optimisation_home()
        elif st.session_state.sub_page == 'minimise_cr':
            minimise_cr_page()
    elif st.session_state.main_tab == "Physical Relationship Analysis":
        physical_relationship_analysis()

if __name__ == "__main__":
    main()
