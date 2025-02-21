import streamlit as st
from functions.load_models import load_models  # Import the load_models function
from functions.load_preprocess_data import load_preprocess_data  # Import the load_preprocess_data function
from functions.plot_5x5_cr import plot_5x5_cr  # Import the correct function for corrosion rate
from functions.plot_5x5_sr import plot_5x5_sr  # Import the correct function for saturation ratio
from functions.pca_plot import pca_plot  # Import the PCA plot function (you need to create this function)

# Function for Descriptive Statistics
def descriptive_analysis(X):
    st.write("Descriptive Statistics:")
    st.write(X.describe())  # Display the descriptive statistics in the Streamlit app


# Main app content
st.title("Main Dashboard")

# Define functions for each section
def data_analysis():
    st.title('Data Analysis')
    st.write("This section will contain your data analysis logic.")

    # Create buttons for both statistical analysis and contour plots
    statistical_analysis_button = st.button('Statistical Analysis')
    contour_button = st.button('Contour Plots')

    if statistical_analysis_button:
        st.session_state.page = 'statistical_analysis'  # Navigate to the statistical analysis page
    
    if contour_button:
        st.session_state.page = 'contour_plots'  # Navigate to the contour plots page
    
    # Home button
    home_button = st.button("Go to Home")
    if home_button:
        st.session_state.page = 'main'  # Navigate back to the main page

def contour_plots():
    st.title('Contour Plots')
    st.write("Choose the plot to display:")

    # Access preloaded models and data
    cr_model, sr_model = st.session_state.models
    X, scaler_X = st.session_state.data

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

    # Home button
    home_button = st.button("Go to Home")
    if home_button:
        st.session_state.page = 'main'  # Navigate back to the main page

def statistical_analysis():
    st.title('Statistical Analysis')
    st.write("This section will contain the statistical analysis logic.")

    # Access the data (X and scaler_X are stored in session state)
    if 'data' not in st.session_state:
        st.write("Data not found. Please load the data first.")
        return

    X, scaler_X = st.session_state.data

    # Button for PCA analysis
    pca_analysis_button = st.button('PCA Analysis')
    if pca_analysis_button:
        st.write("Performing PCA Analysis...")
        pca_plot()  # Call the PCA plot function

    # Button for Descriptive Statistics
    descriptive_stats_button = st.button('Descriptive Statistics')
    if descriptive_stats_button:
        st.write("Descriptive Statistics for the dataset:")
        st.write(X.describe())  # Display the descriptive statistics directly

    input_histograms_button = st.button('Input Histograms')
    if input_histograms_button:
        st.write("Input Histograms will be displayed here.")
        # You can implement the logic for input histograms here later

    # Home button
    home_button = st.button("Go to Home")
    if home_button:
        st.session_state.page = 'main'  # Navigate back to the main page

def optimisation():
    st.title('Optimisation')
    st.write("This section will contain your optimisation logic.")
    # Add more content related to optimisation here

    # Home button
    home_button = st.button("Go to Home")
    if home_button:
        st.session_state.page = 'main'  # Navigate back to the main page

def physical_relationship_analysis():
    st.title('Physical Relationship Analysis')
    st.write("This section will contain your physical relationship analysis logic.")
    # Add more content related to physical relationship analysis here

    # Home button
    home_button = st.button("Go to Home")
    if home_button:
        st.session_state.page = 'main'  # Navigate back to the main page

# Main page content
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page to 'main'

    if 'models' not in st.session_state or 'data' not in st.session_state:
        # Automatically load the models and data when the app is opened
        cr_model, sr_model = load_models()
        X, scaler_X = load_preprocess_data()  # load_preprocess_data should return X and scaler_X
        st.session_state.models = (cr_model, sr_model)
        st.session_state.data = (X, scaler_X)

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

    elif st.session_state.page == 'statistical_analysis':
        statistical_analysis()  # Display the statistical analysis page

    elif st.session_state.page == 'contour_plots':
        contour_plots()

    elif st.session_state.page == 'optimisation':
        optimisation()

    elif st.session_state.page == 'physical_relationship_analysis':
        physical_relationship_analysis()

# Run the app
if __name__ == "__main__":
    main()
