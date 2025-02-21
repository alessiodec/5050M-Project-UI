import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_5x5_sr(X, scaler_X, sr_model):
    mid_points = np.median(X, axis=0)
    var_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
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
            predictions_scaled = sr_model.predict(grid_points_scaled)
            # For saturation ratio, assume the second output is the desired value
            saturation_ratio = predictions_scaled[:, 0].reshape(grid_x.shape)
            
            cont_fill = ax.contourf(grid_x, grid_y, saturation_ratio, levels=10, cmap='viridis')
            cont_line = ax.contour(grid_x, grid_y, saturation_ratio, levels=10, colors='black', linewidths=0.5)
            ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
            ax.set_xlabel(var_names[j])
            ax.set_ylabel(var_names[i])
    
    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont_fill, cax=cbar_ax, label='Scaled Saturation Ratio')
    plt.suptitle('SR For Different Input Combinations', fontsize=18)
    st.pyplot(fig)
