from . import Engine
from . import config
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings

def run_heatsink_analysis(pop_size, pop_retention, num_iterations):
    """
    Runs the heatsink analysis based on user-defined population parameters and number of iterations.

    Args:
        pop_size (int): The number of individuals in the population.
        pop_retention (int): The number of individuals retained after selection.
        num_iterations (int): The number of iterations (generations) to run the evolution process.
    """
    # Update the configuration
    config.POPULATION_SIZE = pop_size
    config.POPULATION_RETENTION_SIZE = pop_retention
    config.FIT_THRESHOLD = 10  # Keeping the threshold constant

    # Ensure required data exists in session state
    if "heatsink_data" not in st.session_state:
        st.error("❌ Heatsink data has not been loaded. Run 'Load Heatsink Data' first.")
        return

    # Unpack stored heatsink data and update config.X and config.y
    df, X, y, standardised_y, mean_y, std_y = st.session_state["heatsink_data"]
    config.X, config.y = X, standardised_y

    st.write("🚀 Initializing Population... This may take a moment.")
    start_time = time.time()

    with st.spinner("Generating initial population..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            init_population = Engine.initialize_population(verbose=1)

    st.write(f"✅ Population initialized in {time.time() - start_time:.2f} seconds")

    # Display a few individuals for reference
    for i, individual in enumerate(init_population[:10]):
        st.text(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)

    st.write("⚙️ Simplifying Population...")
    start_time = time.time()

    with st.spinner("Simplifying expressions..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            simplified_pop = Engine.simplify_and_clean_population(init_population)

    st.write(f"✅ Population simplified in {time.time() - start_time:.2f} seconds")

    # ---- Evolution Loop with Real-Time Graph Updates ----
    st.write("📈 Running Evolution Process...")
    # We'll use a placeholder to update the plot in place.
    chart_placeholder = st.empty()

    # Initialize tracking arrays
    avg_fitness_arr = []
    avg_complexity_arr = []
    best_fitness_arr = []
    iterations = []

    evolution_start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num_iterations):
            # Generate new population from current population copy
            new_population = Engine.generate_new_population(population=new_population.copy(), verbose=1)
            avg_fitness, avg_complexity, optimal_fitness = Engine.evaluate_population(new_population)

            avg_fitness_arr.append(avg_fitness)
            avg_complexity_arr.append(avg_complexity)
            best_fitness_arr.append(optimal_fitness)
            iterations.append(i + 1)

            elapsed_time = time.time() - evolution_start
            st.write(f"Iteration {i+1}: Best Fit={optimal_fitness:.8f}, Avg Fit={avg_fitness:.8f}, Elapsed Time={elapsed_time:.2f}s")

            # Update graph dynamically
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(iterations, avg_fitness_arr, 'bo-', label="Avg Fitness")
            ax.plot(iterations, avg_complexity_arr, 'ro-', label="Complexity")
            ax.plot(iterations, best_fitness_arr, 'go-', label="Best Fitness")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness - 1-$R^2$")
            ax.set_yscale("log")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title("Population Metrics Over Iterations")
            chart_placeholder.pyplot(fig)

            time.sleep(0.1)

    st.success("✅ Heatsink Analysis Completed!")
