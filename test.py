import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.set_option('deprecation.showPyplotGlobalUse', False)

# App Title
st.title("Interactive ML Diagrams with Streamlit")

# Sidebar controls
st.sidebar.header("User Inputs")

# Example control for the Pareto front plot: choose number of random points.
num_points = st.sidebar.slider("Number of Points (for Pareto Front)", min_value=50, max_value=500, value=100, step=50)

st.header("Pareto Front Diagram")
# Generate random points representing two objectives
points = np.random.rand(num_points, 2)

def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points
    Input:
        costs: An (n_points, n_costs) array
    Returns:
        A boolean array of indices that are Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Remove dominated points: if any cost in remaining points is lower than c,
            # then that point cannot be dominated by c.
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient

# Compute Pareto efficient (non-dominated) points
mask = is_pareto_efficient(points)
pareto_points = points[mask]

# Plot all points and highlight the Pareto front
fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], label='All Points', alpha=0.5)
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], color='red', label='Pareto Front', s=80)
ax.set_title("Pareto Front Diagram")
ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.legend()
st.pyplot(fig)

st.header("ML Diagram: SVM Decision Boundary")
# Generate a simple classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
# Train an SVM classifier with RBF kernel
clf = SVC(kernel='rbf', gamma=0.7, probability=True)
clf.fit(X, y)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
fig2, ax2 = plt.subplots()
contour = ax2.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
scatter = ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
ax2.set_title("SVM Decision Boundary")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
st.pyplot(fig2)
