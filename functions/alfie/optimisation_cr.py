import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

# Load Dataset
csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
data = pd.read_csv(csv_url)

# Extract Min and Max Values for Inputs
min_pco2, max_pco2 = data["PCO2"].min(), data["PCO2"].max()
min_d, max_d = data["d"].min(), data["d"].max()

# Preprocessing: Standardization
XData = data[["pH", "T", "PCO2", "v", "d"]]
YData = data[["CR", "SR"]]

# Apply log transformation to selected columns
XData["PCO2"] = np.log10(XData["PCO2"])
XData["v"] = np.log10(XData["v"])
XData["d"] = np.log10(XData["d"])

# Initialize Scaler and Standardize Inputs
scaler = preprocessing.StandardScaler()
XDataScaled = scaler.fit_transform(XData).astype("float32")

# Load Pre-trained Models
CorrosionModel = load_model("functions/alfie/CorrosionRateModel.keras")
SaturationModel = load_model("functions/alfie/SaturationRateModel.keras")

# Reverse scaling function for the final results
def ReverseScalingandLog10(optimisationResult):
    corrosionResultX = scaler.inverse_transform(optimisationResult.copy().reshape(-1, 5))
    corrosionResultX[:, 2:] = 10**corrosionResultX[:, 2:]  # Reverse log transformation
    return corrosionResultX

# Custom Optimization Problem Class
class MinimizeCR(ElementwiseProblem):
    def __init__(self, d, PCO2):
        # Scale d and PCO2 using the dataset's scaler
        d_scaled = scaler.transform(np.array([0, 0, 0, 0, d]).reshape(1, -1))[0][4]
        PCO2_scaled = scaler.transform(np.array([0, 0, PCO2, 0, 0]).reshape(1, -1))[0][2]

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
        X = np.append(X, self.d)
        X = np.insert(X, 2, self.PCO2)
        X = X.reshape(1, -1)

        corrosionResult = CorrosionModel.predict(X, verbose=False).flatten()
        saturationResult = SaturationModel.predict(X, verbose=False).flatten()

        out["F"] = corrosionResult
        out["G"] = -10**saturationResult + 1

# Function to Minimize Corrosion Rate
def minimise_cr(d, PCO2):
    algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
    result = minimizepymoo(MinimizeCR(d, np.log10(PCO2)), algorithmDE, verbose=True, termination=("n_gen", 10))

    # Convert back to real-world values
    resultX = np.array(result.X)
    resultX = np.insert(resultX, 2, scaler.transform(np.array([0, 0, np.log10(PCO2), 0, 0]).reshape(1, -1))[0][2])
    resultX = np.insert(resultX, 4, scaler.transform(np.array([0, 0, 0, 0, d]).reshape(1, -1))[0][4])

    # Return Optimization Results
    return f"CR = {result.F} at {ReverseScalingandLog10(resultX)}"
