################################### IMPORT LIBRARIES & FUNCTIONS ###################################

import streamlit as st
import pandas as pd
import numpy as np

from functions.load_models import load_models
from functions.load_preprocess_data import load_preprocess_data
from functions.plot_5x5_cr import plot_5x5_cr
from functions.plot_5x5_sr import plot_5x5_sr
from functions.pca_plot import pca_plot
from functions.descriptive_analysis import descriptive_analysis
from functions.input_histogram import input_histogram

from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis
from functions.ethan.heatsink_evolution import run_heatsink_evolution  # NEW FUNCTION

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
    if st.button("Go to Home"):
        st.session_state.page = 'main'

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')
    st.write("This section will contain your physical relationship analysis logic.")

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
        if "num_iterations" not in st.session_state:
            st.session_state.num_iterations = 10  # Default iterations

        pop_size = st.number_input("Enter Population Size:", min_value=100, max_value=10000, value=st.session_state.pop_size, step=100)
        pop_retention = st.number_input("Enter Population Retention Size:", min_value=10, max_value=1000, value=st.session_state.pop_retention, step=10)
        num_iterations = st.number_input("Enter Number of Iterations:", min_value=1, max_value=100, value=st.session_state.num_iterations, step=1)

        st.session_state.pop_size = int(pop_size)
        st.session_state.pop_retention = int(pop_retention)
        st.session_state.num_iterations = int(num_iterations)

        if st.button("Run Heatsink Analysis"):
            try:
                run_heatsink_analysis(st.session_state.pop_size, st.session_state.pop_retention)
                st.write(f"✅ Initial population created with Population Size = {st.session_state.pop_size}, Retention Size = {st.session_state.pop_retention}")
                
                # Run evolution process
                run_heatsink_evolution(st.session_state.num_iterations)
                st.write(f"✅ Evolution process completed for {st.session_state.num_iterations} iterations")

            except Exception as e:
                st.error(f"Error running heatsink analysis: {e}")

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
