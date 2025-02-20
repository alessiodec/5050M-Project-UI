import streamlit as st
from functions.load_models import load_models  # Import the load_models function
from functions.load_preprocess_data import load_preprocess_data  # Import the load_preprocess_data function
from functions import plot_cr_5x5, plot_sr_5x5  # Import the renamed contour plot functions

# Call the functions to load models and preprocess data
cr_model, sr_model = load_models()
X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_preprocess_data()

# Main app content
st.title("Main Dashboard")

# Display the loaded models and data summary
st.write("Corrosion Rate Model and Saturation Rate Model are loaded successfully.")
st.write(f"Training data shape: {X_train.shape}, {y_train.shape}")
st.write(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Define functions for each section
def data_analysis():
    st.title('Data Analysis')
    st.write("This section will contain your data analysis logic.")
    
    # Create a button for contour plots
    contour_button = st.button('Contour Plots')
    
    if contour_button:
        # Navigate to the contour plots page
        st.session_state.page = 'contour_plots'

def contour_plots():
    st.title('Contour Plots')
    st.write("Choose the plot to display:")

    # Button for Corrosion Rate contour plot
    cr_button = st.button('Corrosion Rate')
    if cr_button:
        st.write("Generating Corrosion Rate Contour Plot...")
        plot_cr_5x5(cr_model, X_train, scaler_X)  # Call the plot_cr_5x5 function to display the plot

    # Button for Saturation Ratio contour plot
    sr_button = st.button('Saturation Ratio')
    if sr_button:
        st.write("Generating Saturation Ratio Contour Plot...")
        plot_sr_5x5(sr_model, X_train, scaler_X)  # Call the plot_sr_5x5 function to display the plot

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
