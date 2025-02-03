import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed so that the underlying points (and thus the Pareto front) remain constant.
np.random.seed(42)

st.title("Pareto Front with Weighted Sum Optimal Selection")

# Sidebar slider for preference weight.
# At 0.0, full weight is on Objective 2; at 1.0, full weight is on Objective 1.
weight = st.sidebar.slider(
    "Preference Weight for Objective 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)
w1 = weight
w2 = 1 - weight
st.sidebar.write(f"Objective 1 Weight: {w1:.2f} | Objective 2 Weight: {w2:.2f}")

# Generate random points representing two objectives (both assumed to be minimized).
num_points = 200
points = np.random.rand(num_points, 2)

# Function to compute Pareto-efficient (non-dominated) points.
def is_pareto_efficient(costs):
    """
    Returns a boolean array indicating whether each point in `costs` is Pareto efficient.
    For minimization, a point is Pareto efficient if no other point is lower in *both* objectives.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # For all points still marked as efficient, mark as efficient only if at least one objective is lower than c.
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Always keep the current point.
    return is_efficient

# Compute the Pareto front (non-dominated set) from the random points.
pareto_mask = is_pareto_efficient(points)
pareto_points = points[pareto_mask]

# Compute the weighted score for each Pareto point.
# A lower weighted sum is considered better.
weighted_scores = w1 * pareto_points[:, 0] + w2 * pareto_points[:, 1]

# Identify the Pareto point with the minimum weighted score.
best_index = np.argmin(weighted_scores)
best_point = pareto_points[best_index]

# Create the plot.
fig, ax = plt.subplots()

# Plot only the Pareto front points (blue dots).
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', label='Pareto Front', s=50)

# Highlight the selected optimal Pareto point (red dot).
ax.scatter(best_point[0], best_point[1], color='red', label='Optimal Solution', s=100)

ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.set_title("Pareto Front with Optimal Selection by Weighted Sum")
ax.legend()

# Adjust the figure to create space on the right for the text box.
plt.subplots_adjust(right=0.75)

# Place a text box outside the plot (to the right) displaying the optimal weighted score.
fig.text(0.78, 0.5, f"Optimal Score:\n{weighted_scores[best_index]:.2f}",
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

st.pyplot(fig)
