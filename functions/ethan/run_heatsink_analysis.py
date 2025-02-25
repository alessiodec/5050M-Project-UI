import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from IPython.display import clear_output

from . import Engine
from . import config

def run_heatsink_analysis(X, standardised_y):
    """
    Runs the full heatsink analysis, including population initialization,
    simplification, evaluation, and real-time Pareto front visualization.

    Parameters:
        X (ndarray): Feature array
        standardised_y (ndarray): Standardized target variable
    """
    st.write("Initializing Heatsink Analysis...")

    # CELL 3: Configuration Setup
    config.X = X
    config.y = standardised_y

    config.POPULATION_SIZE = 1000
    config.POPULATION_RETENTION_SIZE = 20
    config.FIT_THRESHOLD = 10

    st.write("Configuration set up successfully.")

    # CELL 4: Initialize and Evaluate Population
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        init_population = Engine.initialize_population(verbose=1)

    st.write("Initial Population Evaluated:")
    for i, individual in enumerate(init_population[:10]):  # Display only first 10 for readability
        st.write(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)
    st.write(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # CELL 5: Simplify and Evaluate Population
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        simplified_pop = Engine.simplify_and_clean_population(init_population)

    Engine.evaluate_population(simplified_pop)
    st.write("Simplified Population Evaluated.")
    st.write(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # CELL 6: Real-Time Pareto Front Visualization
    pareto_front = Engine.return_pareto_front(init_population)
    pareto_plot_data = np.array([(ind.fitness, ind.complexity) for ind in pareto_front])
    population_plot_data = np.array([(ind.fitness, ind.complexity) for ind in init_population])
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]

    st.write("Generating real-time Pareto front visualization...")

    for i in range(1, len(population_plot_data) + 1):
        clear_output(wait=True)
        plt.figure(figsize=(8, 6))

        # Plot the current subset of population data
        plt.scatter(population_plot_data[:i, 1], population_plot_data[:i, 0], s=15, label="Population")
        plt.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, color='red', label="Pareto Front")
        plt.scatter(utopia_point[0], utopia_point[1], s=50, color='green', label="Utopia Point")

        plt.yscale("log")
        plt.xlabel("Complexity")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Real-Time Update: Pareto Front & Population")
        st.pyplot(plt)  # Display in Streamlit
        time.sleep(0.01)

    st.write("Heatsink Analysis Completed!")
