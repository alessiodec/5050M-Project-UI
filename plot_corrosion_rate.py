import numpy as np
import matplotlib.pyplot as plt

def plot_corrosion_rate_contours(model, X, scaler_X, scaler_y):
    """Generates and returns corrosion contour plots."""
    if model is None:
        return None, "Model is not loaded!"

    # Define variable names
    var_names = ['pH', 'T', 'PCO2', 'v', 'd']
    mid_points = np.median(X, axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # Create subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]

            if i == j:
                ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x_vals = np.linspace(mins[j], maxs[j], 25)
            y_vals = np.linspace(mins[i], maxs[i], 25)
            grid_x, grid_y = np.meshgrid(x_vals, y_vals)

            grid_points = np.tile(mid_points, (grid_x.size, 1))
            grid_points[:, j] = grid_x.ravel()
            grid_points[:, i] = grid_y.ravel()

            grid_points_scaled = scaler_X.transform(grid_points)
            predictions_scaled = model.predict(grid_points_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled)
            corrosion_rate = predictions[:, 0].reshape(grid_x.shape)

            cont_fill = ax.contourf(grid_x, grid_y, corrosion_rate, levels=10, cmap='viridis')
            cont_line = ax.contour(grid_x, grid_y, corrosion_rate, levels=10, colors='black', linewidths=0.5)
            ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
            ax.set_xlabel(var_names[j])
            ax.set_ylabel(var_names[i])

    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont_fill, cax=cbar_ax, label='Corrosion Rate')
    plt.suptitle('CR For Different Input Combinations', fontsize=18)

    return fig, None  # No errors
