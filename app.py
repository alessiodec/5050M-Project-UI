# app.py
import streamlit as st

st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Data Analysis", 
    "Optimisation", 
    "Physical Relationship Analysis"
])

if option == "Data Analysis":
    import data_analysis
    data_analysis.main()
elif option == "Optimisation":
    import optimisation
    optimisation.main()
elif option == "Physical Relationship Analysis":
    import physical_relationship_analysis
    physical_relationship_analysis.main()
