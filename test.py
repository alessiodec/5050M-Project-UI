import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Generate the Pareto front ---

# f1: First objective values, 100 evenly spaced points between 0 and 1.
f1 = np.linspace(0, 1, 100)

# f2: Second objective values defined as a convex function of f1.
# Here, we use f2 = 1 - sqrt(f1) so that when f1=0, f2=1 and when f1=1, f2=0.
# This gives a convex trade-off curve.
f2 = 1 - np.sqrt(f1)

# --- Create a slider for weighting the objectives ---

# The slider returns a value w between 0 and 1.
# w = 1 means full preference for f1 (first objective),
# while w = 0 means full preference for f2 (second objective).
w = st.slider("Preference weight (0: prefer f2, 1: prefer f1)", 0.0, 1.0, 0.5)

# --- Compute the weighted sum for each point on the Pareto front ---

# The weighted sum objective is: weighted_value = w * f1 + (1 - w) * f2.
# Lower values are better (assuming a minimization problem).
weighted_objectives = w * f1 + (1 - w) * f2

# Find the index of the point with the minimum weighted sum.
optimal_index = np.argmin(weighted_objectives)
optimal_f1 = f1[optimal_index]
optimal_f2 = f2[optimal_index]

# --- Plot the Pareto front and highlight the optimal solution ---

# Create a matplotlib figure and axis.
fig, ax = plt.subplots()

# Plot all the Pareto front points in blue with a line connecting them.
ax.plot(f1, f2, "bo-", label="Pareto Front")

# Highlight the optimal solution (lowest weighted sum) with a red marker.
ax.plot(optimal_f1, optimal_f2, "ro", markersize=10, label="Optimal Solution")

# Label the axes.
ax.set_xlabel("Objective 1")
ax.set_ylabel("Objective 2")

# Add a legend and title to the plot.
ax.legend()
ax.set_title("Convex Pareto Front with Weighted Preference")

# --- Display the plot and the optimal solution ---

# Use Streamlit to show the matplotlib figure.
st.pyplot(fig)

# Display a text box below the plot showing the optimal solution's objective values.
st.text(f"Optimal solution: f1 = {optimal_f1:.3f}, f2 = {optimal_f2:.3f}")
