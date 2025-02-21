import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_plot(X, scaler_X):
    """
    Perform PCA and plot the results.
    
    Parameters:
    X (array-like): The preprocessed input data.
    scaler_X (scaler object): The scaler used to preprocess the data.
    """
    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    X_pca = pca.fit_transform(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    # Plotting the results
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='b', marker='o')
    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
    ax.set_title('PCA of Preprocessed Data')

    # Display the plot
    st.pyplot(fig)

    # Display the explained variance
    st.write("Explained variance by each principal component:")
    st.write(f"PC1: {explained_variance[0]*100:.2f}%")
    st.write(f"PC2: {explained_variance[1]*100:.2f}%")
