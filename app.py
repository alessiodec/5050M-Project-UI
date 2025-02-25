import pandas as pd
import numpy as np

from functions.load_models import load_models
from functions.load_preprocess_data import load_preprocess_data
from functions.plot_5x5_cr import plot_5x5_cr
from functions.plot_5x5_sr import plot_5x5_sr
from functions.pca_plot import pca_plot
from functions.descriptive_analysis import descriptive_analysis
from functions.input_histogram import input_histogram
from functions.load_models import load_models  # Load the models
from functions.load_preprocess_data import load_preprocess_data  # Load & preprocess data
from functions.plot_5x5_cr import plot_5x5_cr  # Plot corrosion rate contours
from functions.plot_5x5_sr import plot_5x5_sr  # Plot saturation ratio contours
from functions.pca_plot import pca_plot  # Plot PCA results
from functions.descriptive_analysis import descriptive_analysis  # Show descriptive stats
from functions.input_histogram import input_histogram  # Display input histograms

from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis
from functions.ethan.heatsink_evolution import run_heatsink_evolution  # NEW FUNCTION
from functions.ethan.optimisation_cr import minimise_cr  # New function for CR minimisation

################################### DEFINE APP SECTIONS ###################################

@@ -33,71 +33,48 @@ def data_analysis():
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
    d = st.number_input("Enter Pipe Diameter (d):", min_value=0.01, max_value=10.0, step=0.01)
    pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=0.001, max_value=10.0, step=0.001)

    # Run optimisation when button is clicked
    if st.button("Run Optimisation"):
        try:
            best_d, best_pco2, min_cr = minimise_cr(d, pco2)
            st.write(f"✅ **Optimal Parameters Found:** d = {best_d}, PCO₂ = {best_pco2}, Min CR = {min_cr:.5f}")
        except Exception as e:
            st.error(f"Error running optimisation: {e}")

    if st.button("Go to Optimisation Menu"):
        st.session_state.page = 'optimisation'

################################### MAIN APP ###################################

def main():
if 'page' not in st.session_state:
        st.session_state.page = 'main'
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
@@ -121,6 +98,9 @@ def main():
elif st.session_state.page == 'optimisation':
optimisation()

    elif st.session_state.page == 'minimise_cr':
        minimise_cr_page()

elif st.session_state.page == 'physical_relationship_analysis':
physical_relationship_analysis()
