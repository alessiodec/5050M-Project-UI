import streamlit as st
import numpy as np
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt

from functions.ethan import Engine, config

def run_heatsink_analysis():
    """
    Runs the full heatsink analysis after the dataset is loaded.
    This function initializes a population, simplifies it, and visualizes the Pareto front dynamically.
    """
    st.write("### Running Heatsink Analysis...")

    # --- USER INPUT FOR POPULATION SIZE ---
    config.POPULATION_SIZE = st.number_input(
        "Enter Population Size", min_value=100, max_value=5000, value=1000, step=100
    )

    config.POPULATION_RETENTION_SIZE = st.number_input(
        "Enter Population Retention Size", min_value=5, max_value=500, value=20, step=5
    )

    config.FIT_THRESHOLD = 10  # Keeping the fitness threshold constant

    st.write("**Selected Parameters:**")
    st.write(f"- Population Size: {config.POPULATION_SIZE}")
    st.write(f"- Population Retention Size: {config.POPULATION_RETENTION_SIZE}")
    st.write(f"- Fitness Threshold: {config.FIT_THRESHOLD}")

    # --- INITIALIZE AND EVALUATE POPULATION ---
    st.write("### Initializing Population...")
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        init_population = Engine.initialize_population(verbose=1)

        # Display individuals in Streamlit instead of printing
        population_output = []
        for i, individual in enumerate(init_population):
            population_output.append(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")
        
        st.text("\n".join(population_output[:20]))  # Show first 20 individuals to avoid clutter

    Engine.evaluate_population(init_population)
    st.write(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # --- SIMPLIFY POPULATION ---
    st.write("### Simplifying Population...")
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        simplified_pop = Engine.simplify_and_clean_population(init_population)

    Engine.evaluate_population(simplified_pop)
    st.write(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # --- PARETO FRONT VISUALIZATION ---
    st.write("### Pareto Front Visualization")

    # Calculate Pareto front and prepare data arrays
    pareto_front = Engine.return_pareto_front(init_population)
    pareto_plot_data = np.array([(ind.fitness, ind.complexity) for ind in pareto_front])
    population_plot_data = np.array([(ind.fitness, ind.complexity) for ind in init_population])
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]

    # Streamlit Dynamic Plot
    plot_placeholder = st.empty()  # Placeholder for updating the plot dynamically

    # Iterate over population points and update the scatter plot dynamically
    for i in range(1, len(population_plot_data) + 1):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the current subset of population data
        ax.scatter(population_plot_data[:i, 1], population_plot_data[:i, 0], s=15, label="Population")
        ax.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, color='red', label="Pareto Front")
        ax.scatter(utopia_point[0], utopia_point[1], s=50, color='green', label="Utopia Point")

        ax.set_yscale("log")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.set_title("Real-Time Update: Pareto Front & Population")

        plot_placeholder.pyplot(fig)  # Update the plot in Streamlit

        time.sleep(0.01)  # Small delay for visualization

    st.write("Pareto Front Visualization Complete!")
