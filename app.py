import streamlit as st
from load_data_model import load_and_preprocess_data, load_trained_model
from plot_contours import plot_corrosion_contours
from plot_saturation_ratio import plot_saturation_ratio_contours

st.title("Corrosion and Saturation Ratio Contour Plots")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Button to load and preprocess data
if st.button("Load Dataset"):
    X, X_scaled, y, y_scaled, scaler_X, scaler_y, error = load_and_preprocess_data()
    if error:
        st.error(error)
    else:
        st.session_state.data = (X, X_scaled, y, y_scaled, scaler_X, scaler_y)
        st.success("Dataset loaded and preprocessed successfully!")

# Button to load model
if st.button("Load Model"):
    model, error = load_trained_model()
    if error:
        st.error(error)
    else:
        st.session_state.model = model
        st.success("Model loaded successfully!")

# Button to generate corrosion rate contour plots
if st.button("Generate Corrosion Rate Contour Plots"):
    if st.session_state.model is None:
        st.error("Please load the model first!")
    elif st.session_state.data is None:
        st.error("Please load the dataset first!")
    else:
        X, X_scaled, y, y_scaled, scaler_X, scaler_y = st.session_state.data
        fig, error = plot_corrosion_contours(st.session_state.model, X, scaler_X, scaler_y)
        if error:
            st.error(error)
        else:
            st.pyplot(fig)

# Button to generate saturation ratio contour plots
if st.button("Generate Saturation Ratio Contour Plots"):
    if st.session_state.model is None:
        st.error("Please load the model first!")
    elif st.session_state.data is None:
        st.error("Please load the dataset first!")
    else:
        X, X_scaled, y, y_scaled, scaler_X, scaler_y = st.session_state.data
        fig, error = plot_saturation_ratio_contours(st.session_state.model, X, scaler_X, scaler_y)
        if error:
            st.error(error)
        else:
            st.pyplot(fig)
