import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App Title
st.title("Pareto Front Selection via Weighted Sum")

# Sidebar: Preference slider
# When the slider is at 0, full weight is on Objective 2; at 1, full weight is on Objective 1.
weight = st.sidebar.slider("Preference Weight for Objective 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
w1 = weight
w2 = 1 - weight
st.sidebar.write(f"Objective 1 Weight: {w1:.2f} | Objective 2 Weight: {w2:.2f}")

# Generate random points representing two objectives (assumed to be minimized)
num_points = 200
points = np.random.rand(num_points, 2)

# Function to compute Pareto-efficient (non-dominated) points.
def is_pareto_efficient(costs):
    """
    Returns a boolean array indicating whether each point in `costs` is Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep self as efficient.
    return is_efficient

# Compute the Pareto front (non-dominated set) among all points.
pareto_mask = is_pareto_efficient(points)
pareto_points = points[pareto_mask]

# For the Pareto front points, compute the weighted sum.
# A lower weighted sum is considered better.
weighted_scores = w1 * pareto_points[:, 0] + w2 * pareto_points[:, 1]

# Identify the Pareto front point with the minimum weighted score.
best_index = np.argmin(weighted_scores)
best_point = pareto_points[best_index]

# Plot only the Pareto front points and highlight the selected optimal point.
fig, ax = plt.subplots()
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], label='Pareto Front Points', alpha=0.7)
ax.scatter(best_point[0], best_point[1], color='red', label='Selected Optimal', s=100)

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_title("Pareto Front with Weighted Sum Selection")
ax.legend()

# Add a text box (without arrow) indicating the optimal score.
ax.text(0.05, 0.95, f"Optimal Score: {weighted_scores[best_index]:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

st.pyplot(fig)
