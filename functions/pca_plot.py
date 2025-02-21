import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

def pca_plot(df_subset):
    columns = [0, 1, 2, 3, 4]  # Indices of the columns for PCA analysis
    
    pca_data = df_subset.iloc[:, columns]  # Select the columns for PCA

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Calculate explained variance ratio
    exp_var_ratio = pca.explained_variance_ratio_

    # Create results dictionary for later plotting
    results = {
        'pca_result': pca_result,
        'loadings': pca.components_,
        'explained_variance_ratio': exp_var_ratio,
        'cumulative_variance_ratio': np.cumsum(exp_var_ratio)
    }

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Scree plot showing explained and cumulative variance
    axes[0].plot(range(1, len(results['explained_variance_ratio']) + 1), results['explained_variance_ratio'], 'bo-', label='Explained Variance')
    axes[0].plot(range(1, len(results['cumulative_variance_ratio']) + 1), results['cumulative_variance_ratio'], 'ro-', label='Cumulative Variance')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].legend()

    # 2. Heatmap of PCA loadings
    loadings = results['loadings']
    features = pca_data.columns
    loadings_df = pd.DataFrame(
        loadings.T,  # Transpose to have variables as rows and PCs as columns
        columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
        index=features
    )

    sns.heatmap(loadings_df, cmap='RdBu', center=0, annot=True, fmt='.2f', ax=axes[1])
    axes[1].set_title('PCA Loadings Heatmap')

    # Use Streamlit to display the plot
    st.pyplot(fig)

    # Return the explained variance ratios for display or further use
    explained_variance = {}
    for i, var in enumerate(results['explained_variance_ratio']):
        explained_variance[f"PC{i+1}"] = {
            'explained_variance_ratio': var,
            'cumulative_variance_ratio': results['cumulative_variance_ratio'][i]
        }

    return explained_variance
