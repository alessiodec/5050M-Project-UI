import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from james_ensemble import load_cr_model  

# Streamlit UI
st.title("Corrosion Rate Prediction Evaluation")

# User inputs for dataset and model paths
csv_url = st.text_input("Enter Google Drive CSV File Link:", 
                        "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o")

model_url = st.text_input("Enter Google Drive Model File Link:", 
                          "https://drive.google.com/uc?export=download&id=1pE7hLzn-Sdtqt1qi9A1IVaWoS4WMu3NN")

if st.button("Run Evaluation"):
    st.write("Downloading dataset and model...")

    try:
        # Download dataset
        csv_path = "corrosion_data.csv"
        gdown.download(csv_url, csv_path, quiet=False)
        df = pd.read_csv(csv_path)

        # Prepare data
        X = df[['pH', 'T', 'PCO2', 'v', 'd']]
        y = df['CR']
        mask = X.notna().all(axis=1)
        X, y = X[mask], y[mask].values

        # Download model
        model_path = "saved_ensemble_model"
        gdown.download(model_url, model_path, quiet=False)

        # Load model
        ensemble_model = load_cr_model(model_path)
        ensemble_pred = ensemble_model.predict(X)

        # Compute metrics
        r2 = r2_score(y, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        mape = mean_absolute_percentage_error(y, ensemble_pred) * 100

        # Display Metrics
        st.subheader("Model Evaluation Metrics")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**MAPE:** {mape:.4f}%")

        # Error Distribution Plot
        st.subheader("Error Distribution")
        plt.figure(figsize=(8, 4))
        sns.kdeplot(np.abs(y - ensemble_pred), label="Ensemble Model", fill=True)
        plt.xlabel("Absolute Error")
        plt.ylabel("Density")
        plt.legend()
        st.pyplot(plt)

        # Residual Analysis Plot
        st.subheader("Residual Analysis")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        residuals = y - ensemble_pred

        # Residuals vs Predictions
        sns.scatterplot(x=ensemble_pred, y=residuals, ax=ax[0], alpha=0.5)
        ax[0].axhline(0, color='r', linestyle='--')
        ax[0].set_xlabel("Predicted Values")
        ax[0].set_ylabel("Residuals")
        ax[0].set_title("Residuals vs Predicted")

        # Residual Histogram
        sns.histplot(residuals, kde=True, ax=ax[1])
        ax[1].set_xlabel("Residuals")
        ax[1].set_title("Residual Distribution")

        st.pyplot(fig)

        st.success("Evaluation Completed!")

    except Exception as e:
        st.error(f"Error: {e}")
