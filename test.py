import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed so that the Pareto front remains constant.
np.random.seed(42)

# App Title
st.title("Pareto Front Selection via Weighted Sum")

# Sidebar: Preference slider
# When the slider is at 0, full weight is on Objective 2;
# at 1, full weight is on Objective 1.
weight = st.sidebar.slider("Preference Weight for Objective 1", 
                           min_value=0.0, max_value=1.0, value=0.5, step=0.01)
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
            is_efficient[i] = True  # Always keep self.
    return is_efficient

# Compute the Pareto front (non-dominated set) among all points.
pareto_mask = is_pareto_efficient(points)
pareto_points = points[pareto_mask]

# For visualization, sort the Pareto front points by Objective 1
sort_idx = np.argsort(pareto_points[:, 0])
sorted_pareto = pareto_points[sort_idx]

# Compute weighted sums for each Pareto point.
# (A lower weighted sum is considered better.)
weighted_scores = w1 * pareto_points[:, 0] + w2 * pareto_points[:, 1]

# Identify the Pareto front point with the minimum weighted score.
best_index = np.argmin(weighted_scores)
best_point = pareto_points[best_index]

# Create the figure and axis.
fig, ax = plt.subplots()

# Plot all points in light gray.
ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.3, label='All Points')

# Plot the Pareto front points in blue.
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', label='Pareto Front', s=50)
# Connect the Pareto front points with a line.
ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], color='blue', linewidth=1)

# Highlight the optimal Pareto front point based on weighted sum.
ax.scatter(best_point[0], best_point[1], color='red', label='Selected Optimal', s=100)

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_title("Pareto Front (Static) with Optimal Point by Weighted Sum")
ax.legend()

# Adjust the figure to create space on the right for the text box.
plt.subplots_adjust(right=0.75)

# Place a text box outside the plot (to the right) displaying the optimal score.
fig.text(0.78, 0.5, f"Optimal Score:\n{weighted_scores[best_index]:.2f}",
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

st.pyplot(fig)
