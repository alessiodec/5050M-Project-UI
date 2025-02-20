import numpy as np
import matplotlib.pyplot as plt

def 5x5_sr(sr_model, X, scaler_X):
    """
    Generate a 5x5 grid of plots showing the Saturation Ratio (SR) predictions
    for different input combinations.
    
    Parameters:
    - sr_model: The trained saturation ratio model.
    - X: The feature data used for predictions (scaled).
    - scaler_X: The StandardScaler used to scale the features.
    
    Outputs:
    - A 5x5 grid of contour plots displaying the predicted saturation ratios.
    """
    
    # List of input names
    var_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d (m)']
    
    # Calculate median for each input (baseline)
    mid_points = np.median(X, axis=0)  # 1D array with 5 median values
    
    # Get lower and upper bounds for each input to define grid ranges
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    # Create a 5x5 grid for subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
    
    # Loop through rows & columns to create plots
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            
            # On the diagonal, display the variable name only
            if i == j:
                ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Generate 25 evenly spaced points for the current two variables
            x_vals = np.linspace(mins[j], maxs[j], 25)
            y_vals = np.linspace(mins[i], maxs[i], 25)
            grid_x, grid_y = np.meshgrid(x_vals, y_vals)
            
            # Create a complete set of input values where the 2 variables vary over the grid,
            # and all other inputs are fixed at their median values
            grid_points = np.tile(mid_points, (grid_x.size, 1))
            grid_points[:, j] = grid_x.ravel()
            grid_points[:, i] = grid_y.ravel()
            
            # Scale grid points as the model was trained on
            grid_points_scaled = scaler_X.transform(grid_points)
            
            # Use the DNN to predict SR for each set of scaled input values
            predictions_scaled = sr_model.predict(grid_points_scaled)
            
            # Use predictions directly (no inverse scaling) and extract the first output for SR
            saturation_ratio = predictions_scaled[:, 0].reshape(grid_x.shape)
            
            # Plot filled contour plot on the current subplot using the scaled predictions
            cont_fill = ax.contourf(grid_x, grid_y, saturation_ratio, levels=10, cmap='viridis')
            # Overlay contour lines for clarity
            cont_line = ax.contour(grid_x, grid_y, saturation_ratio, levels=10, colors='black', linewidths=0.5)
            ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
            
            # Set axis labels
            ax.set_xlabel(var_names[j])
            ax.set_ylabel(var_names[i])
    
    # Adjust layout and add a global colorbar
    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont_fill, cax=cbar_ax, label='Scaled Saturation Ratio')
    
    plt.suptitle('SR For Different Input Combinations', fontsize=18)
    plt.show()
