import streamlit as st

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis", "Settings"])

# Display content based on the selected page
if page == "Home":
    st.title("Home Page")
    st.write("Welcome to the Home page!")

elif page == "Analysis":
    st.title("Analysis Page")
    st.write("This is where you can perform data analysis.")

elif page == "Settings":
    st.title("Settings Page")
    st.write("Customize your app settings here.")
