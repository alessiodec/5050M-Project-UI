import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App Title
st.title("Interactive Pareto Front via Weighted Sum")

# Sidebar control: weight slider between the two objectives.
# Weight for Objective 1 is chosen by the user; Objective 2 gets 1 - weight.
weight = st.sidebar.slider("Preference Weight for Objective 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
w1 = weight
w2 = 1 - weight

st.sidebar.write(f"Objective 1 Weight: {w1:.2f} | Objective 2 Weight: {w2:.2f}")

# Generate random points representing two objectives (to be minimized)
num_points = 200
points = np.random.rand(num_points, 2)

# Compute the weighted sum for each point:
# Lower weighted sum is assumed to be better.
weighted_scores = w1 * points[:, 0] + w2 * points[:, 1]

# Identify the best point (i.e. with the minimum weighted score)
best_index = np.argmin(weighted_scores)
best_point = points[best_index]

# Plot all points and highlight the best solution according to the weighted sum.
fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], label='Points', alpha=0.5)
ax.scatter(best_point[0], best_point[1], color='red', label='Best (Weighted Sum)', s=100)

ax.set_title("Pareto Front via Weighted Sum")
ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")
ax.legend()

# Annotate the best point with its weighted sum value.
ax.annotate(f"Weighted Sum = {weighted_scores[best_index]:.2f}",
            xy=(best_point[0], best_point[1]),
            xytext=(best_point[0] + 0.05, best_point[1] + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05))

st.pyplot(fig)
