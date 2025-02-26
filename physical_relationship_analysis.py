# physical_relationship_analysis.py
import streamlit as st
from functions.ethan.load_hs_data import load_heatsink_data
from functions.ethan.heatsink_analysis import run_heatsink_analysis

def main():
    st.title('Physical Relationship Analysis')
    st.write("This section includes heatsink analysis and related evaluations.")

    # Initialize heatsink data if not already present.
    if "heatsink_data" not in st.session_state:
        st.session_state.heatsink_loaded = False
        st.session_state.heatsink_data = None

    if st.button("Load Heatsink Data"):
        df, X, y, standardised_y, mean_y, std_y = load_heatsink_data(display_output=True)
        st.session_state.heatsink_data = (df, X, y, standardised_y, mean_y, std_y)
        st.session_state.heatsink_loaded = True
        st.write("✅ Heatsink data loaded successfully!")
        st.write(df)

    if st.session_state.get("heatsink_loaded", False):
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

if __name__ == "__main__":
    main()
