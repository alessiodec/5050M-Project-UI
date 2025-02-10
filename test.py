import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize the page state if it doesn't exist.
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def show_home():
    """Display the main menu with buttons for each page."""
    st.title("Main Menu")
    st.write("Choose a page:")
    if st.button("Input / Output Relationship Analysis"):
        st.session_state.page = 'page1'
    if st.button("Page 2"):
        st.session_state.page = 'page2'
    if st.button("Page 3"):
        st.session_state.page = 'page3'
    if st.button("Page 4"):
        st.session_state.page = 'page4'

def show_page1():
    """Display content for Page 1."""
    st.title("Page 1")
    st.write("Welcome to Page 1!")
    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_page2():
    """Display content for Page 2: read CSV, show DataFrame head, and generate contour plots."""
    st.title("Page 2")
    st.write("Welcome to Page 2!")
    
    # -----------------------------
    # Step 1: Load the CSV from Google Drive
    # -----------------------------
    file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"  # Your Google Drive file ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        df = pd.read_csv(url)
        st.write("DataFrame Head:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return  # Exit if the file cannot be loaded
    
    # -----------------------------
    # Step 2: Create contour plots using the data from the CSV
    # -----------------------------
    # Define the input variable names. Ensure that your CSV has these columns.
    var_names = ['pH', 'T', 'PCO2', 'v', 'd']
    if not all(col in df.columns for col in var_names):
        st.error("CSV does not contain all required columns: " + ", ".join(var_names))
        return
    
    # Extract the feature matrix X from the dataframe
    X = df[var_names].to_numpy()
    
    # --- Assumptions about your model and scalers ---
    # The code below assumes that you have a pre-trained DNN and corresponding scalers:
    # - scaler_X: used to transform inputs (features)
    # - scaler_y: used to invert-scale the model's predictions
    # - model: your trained DNN model with a .predict() method
    #
    # For demonstration, if these objects are not defined, we create dummy ones.
    try:
        scaler_X
    except NameError:
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler().fit(X)
    try:
        scaler_y
    except NameError:
        # Assume the output is one-dimensional; create a dummy scaler that does nothing.
        scaler_y = StandardScaler().fit(np.zeros((X.shape[0], 1)))
    try:
        model
    except NameError:
        # Create a dummy model that returns the first input column as prediction.
        class DummyModel:
            def predict(self, inputs):
                return inputs[:, [0]]  # Dummy prediction: simply returns the first feature.
        model = DummyModel()
    
    # Calculate the median, minimum, and maximum for each input column
    mid_points = np.median(X, axis=0)  # baseline input values
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
    
    # Loop over each subplot position in the grid
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            
            # For the diagonal plots, display the variable name.
            if i == j:
                ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Generate 25 evenly spaced points between the min and max of the j-th and i-th variables.
            x_vals = np.linspace(mins[j], maxs[j], 25)
            y_vals = np.linspace(mins[i], maxs[i], 25)
            grid_x, grid_y = np.meshgrid(x_vals, y_vals)
            
            # Build a complete set of input points:
            # All inputs are fixed at their median values except the two of interest (i and j).
            grid_points = np.tile(mid_points, (grid_x.size, 1))
            grid_points[:, j] = grid_x.ravel()
            grid_points[:, i] = grid_y.ravel()
            
            # Scale the grid points using the scaler for X.
            grid_points_scaled = scaler_X.transform(grid_points)
            
            # Predict the scaled output using the model.
            predictions_scaled = model.predict(grid_points_scaled)
            
            # Convert predictions back to the original scale.
            predictions = scaler_y.inverse_transform(predictions_scaled)
            
            # Reshape the predictions (assumed to be the corrosion rate, CR) into a 25x25 grid.
            corrosion_rate = predictions[:, 0].reshape(grid_x.shape)
            
            # Plot filled contours and contour lines.
            cont_fill = ax.contourf(grid_x, grid_y, corrosion_rate, levels=10, cmap='viridis')
            cont_line = ax.contour(grid_x, grid_y, corrosion_rate, levels=10, colors='black', linewidths=0.5)
            ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
            
            # Label the axes.
            ax.set_xlabel(var_names[j])
            ax.set_ylabel(var_names[i])
    
    # Adjust layout and add a global colorbar.
    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont_fill, cax=cbar_ax, label='Corrosion Rate')
    plt.suptitle('CR For Different Input Combinations', fontsize=18)
    
    # Use st.pyplot to display the matplotlib figure in the Streamlit app.
    st.pyplot(fig)
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_page3():
    """Display content for Page 3."""
    st.title("Page 3")
    st.write("Welcome to Page 3!")
    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_page4():
    """Display content for Page 4."""
    st.title("Page 4")
    st.write("Welcome to Page 4!")
    if st.button("Back to Home"):
        st.session_state.page = 'home'

# Render the appropriate page based on the current state.
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'page1':
    show_page1()
elif st.session_state.page == 'page2':
    show_page2()
elif st.session_state.page == 'page3':
    show_page3()
elif st.session_state.page == 'page4':
    show_page4()
