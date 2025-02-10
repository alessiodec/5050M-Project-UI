import streamlit as st
from load_data import load_and_preprocess_data
from load_model import load_trained_model
from plot_contours import plot_corrosion_contours
from plot_saturation_ratio import plot_saturation_ratio_contours

st.title("Corrosion and Saturation Ratio Analysis")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Automatically load data and model when the app starts
if st.session_state.data is None:
    X, X_scaled, y, y_scaled, scaler_X, scaler_y, error = load_and_preprocess_data()
    if error:
        st.error(error)
    else:
        st.session_state.data = (X, X_scaled, y, y_scaled, scaler_X, scaler_y)
        st.success("Dataset loaded successfully!")

if st.session_state.model is None:
    model, error = load_trained_model()
    if error:
        st.error(error)
    else:
        st.session_state.model = model
        st.success("Model loaded successfully!")

# Function to render the home page
def home():
    st.write("Click below to go to the Contour Plots.")
    if st.button("Go to Contour Plots"):
        st.session_state.page = 'contour_plots'

# Function to render the contour plots page
def contour_plots():
    st.title("Contour Plots")

    # Button to generate corrosion rate contour plots
    if st.button("Generate Corrosion Rate Contour Plots"):
        if st.session_state.data and st.session_state.model:
            X, X_scaled, y, y_scaled, scaler_X, scaler_y = st.session_state.data
            fig, error = plot_corrosion_contours(st.session_state.model, X, scaler_X, scaler_y)
            if error:
                st.error(error)
            else:
                st.pyplot(fig)
        else:
            st.error("Data or model not loaded properly.")

    # Button to generate saturation ratio contour plots
    if st.button("Generate Saturation Ratio Contour Plots"):
        if st.session_state.data and st.session_state.model:
            X, X_scaled, y, y_scaled, scaler_X, scaler_y = st.session_state.data
            fig, error = plot_saturation_ratio_contours(st.session_state.model, X, scaler_X, scaler_y)
            if error:
                st.error(error)
            else:
                st.pyplot(fig)
        else:
            st.error("Data or model not loaded properly.")

    if st.button("Back to Home"):
        st.session_state.page = 'home'

# Page navigation
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'contour_plots':
    contour_plots()
