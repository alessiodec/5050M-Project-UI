import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_preprocess_data():
    # Import data
    csv_url = f"https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    df = pd.read_csv(csv_url)

    # Select columns to keep
    cols_to_keep = list(range(0, 5)) + [7, 17]  # pH, T, PCO2, v, d, CR SR (original order)
    df_subset = df.iloc[:, cols_to_keep].copy()  # Make new dataframe

    # Apply log10 transformation to selected columns
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])  # log10 PCO2, v, d

    print(df_subset.head())

    # Split and scale data
    X = df_subset.iloc[:, :5].values  # 5 inputs
    y = df_subset.iloc[:, 5:7].values  # 2 outputs

    # Create standard scaler instances
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale the data
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split scaled features and targets into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    return X, X_train, X_test, y_train, y_test, scaler_X, scaler_y
