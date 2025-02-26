st.session_state.page = 'statistical_analysis'
if st.button('Contour Plots'):
st.session_state.page = 'contour_plots'
    # Return to Data Analysis home (instead of the removed main menu)
if st.button("Go to Home"):
        st.session_state.page = 'main'
        st.session_state.page = 'Data Analysis'

def contour_plots():
st.title('Contour Plots')
@@ -43,7 +44,7 @@ def contour_plots():
st.write("Generating Saturation Ratio Contour Plot...")
plot_5x5_sr(X, scaler_X, sr_model)
if st.button("Go to Home"):
        st.session_state.page = 'main'
        st.session_state.page = 'Data Analysis'

def statistical_analysis():
st.title('Statistical Analysis')
@@ -65,7 +66,7 @@ def statistical_analysis():
st.write("Generating Input Histograms...")
input_histogram()
if st.button("Go to Home"):
        st.session_state.page = 'main'
        st.session_state.page = 'Data Analysis'

def optimisation():
st.title('Optimisation')
@@ -74,7 +75,7 @@ def optimisation():
if st.button("Minimise CR for Given d and PCO₂"):
st.session_state.page = 'minimise_cr'
if st.button("Go to Home"):
        st.session_state.page = 'main'
        st.session_state.page = 'Optimisation'

def minimise_cr_page():
st.title("Minimise Corrosion Rate (CR)")
@@ -101,7 +102,7 @@ def minimise_cr_page():
st.error(f"Error running optimisation: {e}")

if st.button("Go to Optimisation Menu"):
        st.session_state.page = 'optimisation'
        st.session_state.page = 'Optimisation'

def physical_relationship_analysis():
st.title('Physical Relationship Analysis')
@@ -133,43 +134,57 @@ def physical_relationship_analysis():
st.error(f"Error running heatsink analysis: {e}")

if st.button("Go to Home"):
        st.session_state.page = 'main'
        st.session_state.page = 'Physical Relationship Analysis'

################################### MAIN APP ###################################

def main():
    # Initialize the navigation page if not already set.
if 'page' not in st.session_state:
        st.session_state.page = 'main'

        st.session_state.page = 'Data Analysis'
    
    # Load models and data if not already present.
if 'models' not in st.session_state or 'data' not in st.session_state:
cr_model, sr_model = load_models()
df_subset, X, scaler_X = load_preprocess_data()
st.session_state.models = (cr_model, sr_model)
st.session_state.data = (df_subset, X, scaler_X)

    if st.session_state.page == 'main':
        st.title('Main Menu')
        st.write("Select an option to proceed:")

        if st.button('Data Analysis'):
            st.session_state.page = 'data_analysis'
        elif st.button('Optimisation'):
            st.session_state.page = 'optimisation'
        elif st.button('Physical Relationship Analysis'):
            st.session_state.page = 'physical_relationship_analysis'

    elif st.session_state.page == 'data_analysis':
        
    # Map current session_state.page to its main tab
    if st.session_state.page in ["data_analysis", "statistical_analysis", "contour_plots", "Data Analysis"]:
        default_tab = "Data Analysis"
    elif st.session_state.page in ["optimisation", "minimise_cr", "Optimisation"]:
        default_tab = "Optimisation"
    elif st.session_state.page in ["physical_relationship_analysis", "Physical Relationship Analysis"]:
        default_tab = "Physical Relationship Analysis"
    else:
        default_tab = "Data Analysis"
        
    options = ["Data Analysis", "Optimisation", "Physical Relationship Analysis"]
    default_index = options.index(default_tab)
    
    # Sidebar navigation: the radio buttons act as tabs.
    main_tab = st.sidebar.radio("Navigation", options, index=default_index)
    
    # If user selects a different main tab from the sidebar, update the page.
    if main_tab != default_tab:
        st.session_state.page = main_tab

    # Render page content based on the current navigation state.
    if st.session_state.page == "Data Analysis":
data_analysis()
    elif st.session_state.page == 'statistical_analysis':
    elif st.session_state.page == "statistical_analysis":
statistical_analysis()
    elif st.session_state.page == 'contour_plots':
    elif st.session_state.page == "contour_plots":
contour_plots()
    elif st.session_state.page == 'optimisation':
    elif st.session_state.page == "Optimisation":
optimisation()
    elif st.session_state.page == 'minimise_cr':
    elif st.session_state.page == "minimise_cr":
minimise_cr_page()
    elif st.session_state.page == 'physical_relationship_analysis':
    elif st.session_state.page == "Physical Relationship Analysis":
physical_relationship_analysis()
    else:
        st.write("")  # Blank main area if no matching page

if __name__ == "__main__":
main()
