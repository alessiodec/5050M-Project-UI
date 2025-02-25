import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
import streamlit as st

# -------------------------------------------------------------------
# Load dataset and prepare data
# -------------------------------------------------------------------
csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
Data_ph5 = pd.read_csv(csv_url)

# Keep only the data of interest
XData = Data_ph5[["pH", "T", "PCO2", "v", "d"]]
YData = Data_ph5[["CR", "SR"]]

# Apply log transformation to PCO2, v, and d (as in the original notebook)
XData["PCO2"] = np.log10(XData["PCO2"])
XData["v"] = np.log10(XData["v"])
XData["d"] = np.log10(XData["d"])
XData = XData.dropna()
YData = YData.dropna()

# -------------------------------------------------------------------
# Load pre-trained models
# -------------------------------------------------------------------
CorrosionModel = load_model("functions/alfie/CorrosionRateModel.keras")
SaturationModel = load_model("functions/alfie/SaturationRateModel.keras")

# -------------------------------------------------------------------
# Create and fit a scaler on the input data (all 5 features)
# -------------------------------------------------------------------
scaler = preprocessing.StandardScaler()
XDataScaled = scaler.fit_transform(XData).astype("float32")

# -------------------------------------------------------------------
# Function to reverse scaling and log10 transformation.
# Expects a 1D array of length 5.
# -------------------------------------------------------------------
def ReverseScalingandLog10(optimisationResult):
    result_reshaped = optimisationResult.reshape(1, -1)
    real_values = scaler.inverse_transform(result_reshaped)
    # Reverse the log transformation for columns 2,3,4 (PCO2, v, d)
    real_values[:, 2:] = 10 ** real_values[:, 2:]
    return real_values

# -------------------------------------------------------------------
# Define the optimization problem using pymoo.
# We fix PCO2 and d (after log-transforming and scaling the user inputs)
# and optimize over pH, T, and v.
# -------------------------------------------------------------------
class minCRstSTconstantdPCO2(ElementwiseProblem):
    def __init__(self, d, PCO2):
        """
        d: fixed pipe diameter (scaled value)
        PCO2: fixed CO₂ partial pressure (scaled value)
        """
        self.fixed_d = d
        self.fixed_PCO2 = PCO2
        xl = np.array([XDataScaled[:, 0].min(), XDataScaled[:, 1].min(), XDataScaled[:, 3].min()])
        xu = np.array([XDataScaled[:, 0].max(), XDataScaled[:, 1].max(), XDataScaled[:, 3].max()])
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)
        
    def _evaluate(self, X, out, *args, **kwargs):
        # Reconstruct full design vector: [pH, T, fixed_PCO2, v, fixed_d]
        X = np.array(X.reshape(1, -1))
        X = np.append(X, self.fixed_d)
        X = np.insert(X, 2, self.fixed_PCO2)
        X = X.reshape(1, -1)
        corrosionResult = CorrosionModel.predict(X, verbose=False).flatten()
        saturationResult = SaturationModel.predict(X, verbose=False).flatten()
        out["F"] = corrosionResult
        out["G"] = -10 ** saturationResult + 1

# -------------------------------------------------------------------
# Define the minimise_cr function.
# User inputs: d and PCO2 in original units.
# -------------------------------------------------------------------
def minimise_cr(d, PCO2):
    """
    Minimises the corrosion rate (CR) for given pipe diameter (d) and CO₂ partial pressure (PCO2).
    Args:
        d (float): Pipe diameter (original, real-world value).
        PCO2 (float): CO₂ partial pressure (original, real-world value).
    Returns:
        best_params (np.array): Full design vector (unscaled, real-world values).
        min_cr (float): Minimum corrosion rate.
    """
    st.write("DEBUG: User input d =", d, "PCO2 =", PCO2)
    # Apply log10 transformation to user inputs so they match dataset transformations.
    d_log = np.log10(d)
    PCO2_log = np.log10(PCO2)
    st.write("DEBUG: d_log =", d_log, "PCO2_log =", PCO2_log)
    
    # Scale the fixed values:
    darray = scaler.transform(np.array([0, 0, 0, 0, d_log]).reshape(1, -1))
    d_scaled = darray[0][4]
    PCO2array = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))
    PCO2_scaled = PCO2array[0][2]
    st.write("DEBUG: d_scaled =", d_scaled, "PCO2_scaled =", PCO2_scaled)
    
    # Create the optimization problem with fixed scaled d and PCO2.
    problem = minCRstSTconstantdPCO2(d=d_scaled, PCO2=PCO2_scaled)
    algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
    result = minimizepymoo(problem, algorithmDE, verbose=True, termination=("n_gen", 10))
    
    st.write("DEBUG: Optimization result:", result)
    optimized_vars = np.atleast_1d(result.X).flatten()
    st.write("DEBUG: optimized_vars =", optimized_vars, "Shape:", optimized_vars.shape)
    if optimized_vars.size == 1:
        try:
            optimized_vars = np.array(result.X[0]).flatten()
            st.write("DEBUG: Rewrapped optimized_vars =", optimized_vars, "Shape:", optimized_vars.shape)
        except Exception as e:
            raise ValueError("Optimization result structure is not as expected: " + str(result.X))
    if optimized_vars.size != 3:
        raise ValueError(f"Expected optimized_vars to have 3 elements, got {optimized_vars.size}")
    
    full_design_scaled = np.zeros(5)
    full_design_scaled[0] = optimized_vars[0]   # pH
    full_design_scaled[1] = optimized_vars[1]   # T
    full_design_scaled[2] = PCO2_scaled         # fixed, scaled PCO2
    full_design_scaled[3] = optimized_vars[2]   # v
    full_design_scaled[4] = d_scaled            # fixed, scaled d
    
    best_params = ReverseScalingandLog10(full_design_scaled)
    min_cr = result.F[0]
    
    st.write("DEBUG: best_params =", best_params)
    st.write("DEBUG: min_cr =", min_cr)
    return best_params, min_cr
