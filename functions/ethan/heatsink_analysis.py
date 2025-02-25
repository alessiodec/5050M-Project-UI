import Engine
import config
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings

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
    if "heatsink_data" not in config.__dict__:
        raise ValueError("Heatsink data has not been loaded. Run 'load_heatsink_data()' first.")

    config.X, config.y = config.heatsink_data[1], config.heatsink_data[3]

    # ---- CELL 4: Initialize Population ----
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        init_population = Engine.initialize_population(verbose=1)

        for i, individual in enumerate(init_population):
            print(f"{i}: Fitness={individual.fitness:.4f}, Complexity={individual.complexity}, Eq={individual.individual}")

    Engine.evaluate_population(init_population)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # ---- CELL 5: Simplify Population ----
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        simplified_pop = Engine.simplify_and_clean_population(init_population)

    Engine.evaluate_population(simplified_pop)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    # ---- CELL 6: Real-Time Pareto Front Visualization ----
    pareto_front = Engine.return_pareto_front(init_population)
    pareto_plot_data = np.array([(ind.fitness, ind.complexity) for ind in pareto_front])
    population_plot_data = np.array([(ind.fitness, ind.complexity) for ind in init_population])
    utopia_point = [min(population_plot_data[:, 1]), min(population_plot_data[:, 0])]

    for i in range(1, len(population_plot_data) + 1):
        clear_output(wait=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(population_plot_data[:i, 1], population_plot_data[:i, 0], s=15, label="Population")
        plt.scatter(pareto_plot_data[:, 1], pareto_plot_data[:, 0], s=15, color='red', label="Pareto Front")
        plt.scatter(utopia_point[0], utopia_point[1], s=50, color='green', label="Utopia Point")

        plt.yscale("log")
        plt.xlabel("Complexity")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Real-Time Update: Pareto Front & Population")
        plt.show()
        time.sleep(0.01)

    print("✅ Heatsink Analysis Completed!")
