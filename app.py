import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from james_ensemble import load_cr_model
import os

def analyse_boundary_performance(y_true, predictions_dict, feature_data, percentile=10):
    results = {}
    y_true = y_true.ravel()
    lower_mask = y_true <= np.percentile(y_true, percentile)
    upper_mask = y_true >= np.percentile(y_true, 100-percentile)
    
    st.subheader(f"Performance at {percentile}th percentile boundaries of target values:")
    
    for region, mask in [("Lower", lower_mask), ("Upper", upper_mask)]:
        st.write(f"{region} {percentile}% boundary:")
        for model_name, preds in predictions_dict.items():
            preds = preds.ravel()
            rmse = np.sqrt(mean_squared_error(y_true[mask], preds[mask]))
            mape = mean_absolute_percentage_error(y_true[mask], preds[mask]) * 100
            r2 = r2_score(y_true[mask], preds[mask])
            
            st.write(f"{model_name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%")
            results[f"{region}_{model_name}"] = {"rmse": rmse, "mape": mape, "r2": r2}
    
    return results

def plot_error_distributions(y_true, predictions_dict):
    plt.figure(figsize=(12, 6))
    y_true = y_true.ravel()
    
    for model_name, preds in predictions_dict.items():
        errors = np.abs(y_true - preds.ravel())
        sns.kdeplot(errors, label=model_name)
    
    plt.title("Error Distribution Comparison")
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)

def evaluate_ensemble(csv_path, ensemble_path):
    df = pd.read_csv(csv_path)
    X = df[['pH', 'T', 'PCO2', 'v', 'd']]
    y = df['CR']
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    y_true = y.values
    
    ensemble_model = load_cr_model(ensemble_path)
    ensemble_pred = ensemble_model.predict(X)
    
    r2 = r2_score(y_true, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
    mape = mean_absolute_percentage_error(y_true, ensemble_pred) * 100
    
    st.subheader("Ensemble Model Metrics")
    st.write(f"R2: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
    
    predictions_dict = {'Ensemble': ensemble_pred.reshape(-1, 1)}
    analyse_boundary_performance(y_true.reshape(-1, 1), predictions_dict, X)
    plot_error_distributions(y_true, predictions_dict)
    
    return predictions_dict

def main():
    st.title("Corrosion Rate Prediction Analysis")
    
    csv_path = st.text_input("Enter CSV Path:", "/path/to/dataset.csv")
    ensemble_path = st.text_input("Enter Model Path:", "/path/to/ensemble_model")
    
    if st.button("Run Evaluation"):
        if os.path.exists(csv_path) and os.path.exists(ensemble_path):
            evaluate_ensemble(csv_path, ensemble_path)
        else:
            st.error("Invalid file path. Check dataset and model paths.")

if __name__ == "__main__":
    main()

