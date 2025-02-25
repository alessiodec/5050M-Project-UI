from . import Engine
from . import config
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  # Streamlit-compatible plotting

def run_heatsink_analysis(pop_size, pop_retention):
    """
    Runs the heatsink analysis based on user-defined population parameters.

    Args:
        pop_size (int): The number of individuals in the population.
        pop_retention (int): The number of individuals retained after selection.
    """

    # Update the configuration
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Keeping the threshold constant

    # Ensure required data exists
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded. Run 'Load Heatsink Data' first.")
        return

    # Unpack stored heatsink data
    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y  # Update global config

    # ---- CELL 4: Initialize Population ----
    st.write("🚀 Initializing Population... This may take a moment.")
    start_time = time.time()

    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            init_population = Engine.initialize_population(verbose=1)

    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds")

    # Display population info
    for i, individual in enumerate(init_population[:10]):  # Show only first 10 for brevity
        st.text(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)

    # ---- CELL 5: Simplify Population ----
    st.write("⚙️ Simplifying and Cleaning Population...")
    start_time = time.time()

    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_pop = Engine.simplify_and_clean_population(init_population)

    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds")

    # ---- CELL 6: Real-Time Pareto Front Visualization ----
    st.write("📈 Generating Pareto Front Visualization...")
    pareto_front = Engine.return_pareto_front(init_population)
    pareto_plot_data = np.array([(ind.fitness, ind.complexity) for ind in pareto_front])
    population_plot_data = np.array([(ind.fitness, ind.complexity) for ind in init_population])
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]

    plot_placeholder = st.empty()  # Create a Streamlit placeholder for dynamic updates

    for i in range(1, len(population_plot_data) + 1):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(population_plot_data[:i, 1], population_plot_data[:i, 0], s=15, label="Population")
        ax.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, color='red', label="Pareto Front")
        ax.scatter(utopia_point[0], utopia_point[1], s=50, color='green', label="Utopia Point")

        ax.set_yscale("log")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.set_title("Real-Time Update: Pareto Front & Population")

        plot_placeholder.pyplot(fig)  # Update the plot in Streamlit
        time.sleep(0.01)  # Simulate real-time update delay

    st.success("✅ Heatsink Analysis Completed!")
