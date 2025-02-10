import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """Loads and preprocesses the dataset from Google Drive."""
    file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        return None, f"Error loading CSV file: {e}"

    # Select relevant columns
    cols_to_keep = list(range(0, 5)) + [7, 17]
    df_subset = df.iloc[:, cols_to_keep].copy()
    
    # Apply log transformation to PCO2, v, d
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])

    # Extract inputs and outputs
    X = df_subset.iloc[:, :5].values
    y = df_subset.iloc[:, 5:7].values

    # Scale inputs and outputs
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X, X_scaled, y, y_scaled, scaler_X, scaler_y, None  # No errors
