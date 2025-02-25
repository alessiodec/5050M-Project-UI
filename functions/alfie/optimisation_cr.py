import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE  # Differential Evolution
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.operators.sampling.lhs import LHS
from sklearn import preprocessing
import streamlit as st

# Load dataset from Google Drive
CSV_URL = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
Data_ph5 = pd.read_csv(CSV_URL)

# Keep only necessary columns
XData = Data_ph5[["pH", "T", "PCO2", "v", "d"]]
YData = Data_ph5[["CR", "SR"]]

# Apply log transformations
XData["PCO2"] = np.log10(XData["PCO2"])
XData["v"] = np.log10(XData["v"])
XData["d"] = np.log10(XData["d"])
XData = XData.dropna()
YData = YData.dropna()

# Load pre-trained models
CorrosionModel = load_model("functions/alfie/CorrosionRateModel.keras")
SaturationModel = load_model("functions/alfie/SaturationRateModel.keras")

# Scale input data
scaler = preprocessing.StandardScaler()
XDataScaled = scaler.fit_transform(XData).astype('float32')

# Function to reverse scaling and log10 transformation
def ReverseScalingandLog10(optimisationResult):
    """
    Converts pymoo-optimized results back to real-world values.

    Args:
        optimisationResult (np.array): Optimized scaled values.

    Returns:
        np.array: Real-world scaled and unlogged values.
    """
    corrosionResultX = scaler.inverse_transform(optimisationResult.copy().reshape(-1, 5))
    corrosionResultX[:, 2:] = 10**corrosionResultX[:, 2:]
    return corrosionResultX

# Define pymoo optimization problem
class MinCR(ElementwiseProblem):
    def __init__(self, d_scaled, PCO2_scaled):
        self.d = d_scaled
        self.PCO2 = PCO2_scaled
        super().__init__(
            n_var=3, 
            n_obj=1,
            n_ieq_constr=1,
            xl=np.array([XDataScaled[:, 0].min(), XDataScaled[:, 1].min(), XDataScaled[:, 3].min()]),
            xu=np.array([XDataScaled[:, 0].max(), XDataScaled[:, 1].max(), XDataScaled[:, 3].max()])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.array(X.reshape(1, -1))
        X = np.append(X, self.d)  # Append user-defined d
        X = np.insert(X, 2, self.PCO2)  # Insert user-defined PCO2
        X = X.reshape(1, -1)

        corrosionResult = CorrosionModel.predict(X, verbose=False).flatten()
        saturationResult = SaturationModel.predict(X, verbose=False).flatten()

        out["F"] = corrosionResult
        out["G"] = -10**saturationResult + 1  # Constraint: -SR + 1 <= 0

def minimise_cr(d_input, PCO2_input):
    """
    Finds optimal conditions to minimize CR given d and PCO₂.

    Args:
        d_input (float): User-defined pipe diameter.
        PCO2_input (float): User-defined CO₂ partial pressure.

    Returns:
        best_params (np.array): Optimized real-world values.
        min_CR (float): Minimum corrosion rate found.
    """
    # Scale user inputs
    d_scaled = scaler.transform(np.array([0, 0, 0, 0, d_input]).reshape(1, -1))[0][4]
    PCO2_scaled = scaler.transform(np.array([0, 0, PCO2_input, 0, 0]).reshape(1, -1))[0][2]

    # Run Differential Evolution Algorithm
    algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
    result = minimizepymoo(MinCR(d_scaled, PCO2_scaled), algorithmDE, verbose=False, termination=('n_gen', 10))

    # Convert back to real-world values
    optimised_X = np.array(result.X)
    optimised_X = np.insert(optimised_X, 2, PCO2_scaled)
    optimised_X = np.insert(optimised_X, 4, d_scaled)

    best_params = ReverseScalingandLog10(optimised_X)
    min_CR = result.F[0]

    return best_params, min_CR
