import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
from . import Engine
from . import config

def run_heatsink_evolution(num_iterations):
    """
    Runs the evolution process for a user-defined number of iterations.

    Args:
        num_iterations (int): Number of iterations to run the evolution process.
    """
    
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data not found! Please load it first.")
        return

    # Ensure X and y exist in config (needed for Engine functions)
    config.X, config.y = st.session_state.heatsink_data[1], st.session_state.heatsink_data[3]

    # Initialize population correctly (instead of using raw X values)
    new_population = Engine.initialize_population(verbose=1)

    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = list(range(num_iterations))

    start_time = time.time()

    # Streamlit placeholder to update graph dynamically
    chart_placeholder = st.empty()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for i in iterations:
            new_population = Engine.generate_new_population(population=new_population, verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)

            elapsed_time = time.time() - start_time

            st.write(f"Iter {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Avg Comp={avg_complexity:.5f}, Iter Time={elapsed_time:.2f}s")

            # --- Clear previous figure and update ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(iterations[: i+1], avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations[: i+1], avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations[: i+1], best_fitness_arr, 'go-', label="Best Fitness")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend()
            ax.set_title("Population Metrics Over Iterations")

            # Update the existing plot dynamically
            chart_placeholder.pyplot(fig)

            time.sleep(0.1)

    st.success("✅ Evolution process completed!")

