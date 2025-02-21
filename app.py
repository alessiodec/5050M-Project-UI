import streamlit as st
from functions.load_models import load_models  # Import the load_models function
from functions.load_preprocess_data import load_preprocess_data  # Import the load_preprocess_data function
from functions.plot_5x5_cr import plot_5x5_cr  # Import the correct function for corrosion rate
from functions.plot_5x5_sr import plot_5x5_sr  # Import the correct function for saturation ratio

# Main app content
st.title("Main Dashboard")

# Define functions for each section
def data_analysis():
    st.title('Data Analysis')
    st.write("This section will contain your data analysis logic.")

    # Create a button for statistical analysis
    contour_button = st.button('Statistical Analysis')
    
    # Create a button for contour plots
    contour_button = st.button('Contour Plots')
    
    if contour_button:
        # Navigate to the contour plots page
        st.session_state.page = 'contour_plots'

def contour_plots():
    st.title('Contour Plots')
    st.write("Choose the plot to display:")

    cr_model, sr_model = load_models()
    X, scaler_X = load_preprocess_data()

    # Button for Corrosion Rate contour plot
    cr_button = st.button('Corrosion Rate')
    if cr_button:
        st.write("Generating Corrosion Rate Contour Plot...")
        plot_5x5_cr(X, scaler_X, cr_model)  # Call the plot_5x5_cr function to display the plot

    # Button for Saturation Ratio contour plot
    sr_button = st.button('Saturation Ratio')
    if sr_button:
        st.write("Generating Saturation Ratio Contour Plot...")

        plot_5x5_sr(X, scaler_X, sr_model)  # Call the plot_5x5_sr function to display the plot

def optimisation():
    st.title('Optimisation')
    st.write("This section will contain your optimisation logic.")
    # Add more content related to optimisation here

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')
    st.write("This section will contain your physical relationship analysis logic.")
    # Add more content related to physical relationship analysis here

# Main page content
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page to 'main'
    
    if st.session_state.page == 'main':
        st.title('Main Menu')
        st.write("Select an option to proceed:")

        # Create buttons for each section
        button_data_analysis = st.button('Data Analysis')
        button_optimisation = st.button('Optimisation')
        button_physical_relationship_analysis = st.button('Physical Relationship Analysis')

        # Handle button clicks
        if button_data_analysis:
            st.session_state.page = 'data_analysis'
        elif button_optimisation:
            st.session_state.page = 'optimisation'
        elif button_physical_relationship_analysis:
            st.session_state.page = 'physical_relationship_analysis'
    
    elif st.session_state.page == 'data_analysis':
        data_analysis()

    elif st.session_state.page == 'contour_plots':
        contour_plots()

    elif st.session_state.page == 'optimisation':
        optimisation()

    elif st.session_state.page == 'physical_relationship_analysis':
        physical_relationship_analysis()

# Run the app
if __name__ == "__main__":
    main()
