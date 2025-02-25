import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
import streamlit as st

# Load Dataset
csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
Data_ph5 = pd.read_csv(csv_url)

# Keep only the data of interest
XData = Data_ph5[["pH", "T", "PCO2", "v", "d"]]
YData = Data_ph5[["CR", "SR"]]

# Apply log transformation to PCO2, v, and d
XData["PCO2"] = np.log10(XData["PCO2"])
XData["v"] = np.log10(XData["v"])
XData["d"] = np.log10(XData["d"])
XData = XData.dropna()
YData = YData.dropna()

# Load pre-trained models
CorrosionModel = load_model("functions/alfie/CorrosionRateModel.keras")
SaturationModel = load_model("functions/alfie/SaturationRateModel.keras")

# Create and fit scaler on the input data
scaler = preprocessing.StandardScaler()
XDataScaled = scaler.fit_transform(XData).astype("float32")

def ReverseScalingandLog10(optimisationResult):
    """
    Reverses the scaling and log10 transformation for the input vector.
    Expects a 1D array with 5 elements.
    """
    result_reshaped = optimisationResult.reshape(1, -1)
    corrosionResultX = scaler.inverse_transform(result_reshaped)
    corrosionResultX[:, 2:] = 10 ** corrosionResultX[:, 2:]
    return corrosionResultX

class MinimizeCR(ElementwiseProblem):
    def __init__(self, d, PCO2):
        """
        d: user-defined pipe diameter (real-world value)
        PCO2: user-defined CO₂ partial pressure (real-world value)
              (The function will log-transform PCO2 inside.)
        """
        # Scale the fixed values for d and PCO2:
        # For d: expected to be at index 4; for PCO2: at index 2.
        d_scaled = scaler.transform(np.array([0, 0, 0, 0, d]).reshape(1, -1))[0][4]
        # We log-transform PCO2 first, then scale.
        PCO2_log = np.log10(PCO2)
        PCO2_scaled = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))[0][2]
        self.fixed_d = d_scaled
        self.fixed_PCO2 = PCO2_scaled

        # Our design variables: pH (index 0), T (index 1), and v (index 3)
        xl = np.array([XDataScaled[:, 0].min(), XDataScaled[:, 1].min(), XDataScaled[:, 3].min()])
        xu = np.array([XDataScaled[:, 0].max(), XDataScaled[:, 1].max(), XDataScaled[:, 3].max()])
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        # Reconstruct full 5-element design vector: [pH, T, PCO2, v, d]
        full_design = np.zeros(5)
        full_design[0] = X[0]      # pH
        full_design[1] = X[1]      # T
        full_design[2] = self.fixed_PCO2  # PCO2 (scaled)
        full_design[3] = X[2]      # v
        full_design[4] = self.fixed_d     # d (scaled)
        full_design = full_design.reshape(1, -1)

        corrosionResult = CorrosionModel.predict(full_design, verbose=False).flatten()
        saturationResult = SaturationModel.predict(full_design, verbose=False).flatten()

        out["F"] = corrosionResult
        out["G"] = -10**saturationResult + 1

def minimise_cr(d, PCO2):
    """
    Minimises the corrosion rate (CR) for given pipe diameter (d) and CO₂ partial pressure (PCO2).

    Args:
        d (float): Pipe diameter (real-world value).
        PCO2 (float): CO₂ partial pressure (real-world value).

    Returns:
        best_params (np.array): The full design vector (unscaled, real-world values).
        min_cr (float): The minimum corrosion rate.
    """
    problem = MinimizeCR(d, PCO2)
    algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
    result = minimizepymoo(problem, algorithmDE, verbose=True, termination=("n_gen", 10))

    # Ensure result.X is at least 1D.
    optimized_vars = np.atleast_1d(result.X).flatten()
    if optimized_vars.size != 3:
        raise ValueError(f"Expected optimized_vars to have 3 elements, got {optimized_vars.size}")

    full_design_scaled = np.zeros(5)
    full_design_scaled[0] = optimized_vars[0]  # pH
    full_design_scaled[1] = optimized_vars[1]  # T
    full_design_scaled[2] = problem.fixed_PCO2  # PCO2 (scaled)
    full_design_scaled[3] = optimized_vars[2]  # v
    full_design_scaled[4] = problem.fixed_d     # d (scaled)

    best_params = ReverseScalingandLog10(full_design_scaled)
    min_cr = result.F[0]

    return best_params, min_cr
