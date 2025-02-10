import streamlit as st
import pandas as pd

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
    """Display content for Page 2 and print the DataFrame head."""
    st.title("Page 2")
    st.write("Welcome to Page 2!")
    
    # Define the Google Drive file ID and build the direct download URL.
    file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Read the CSV file into a DataFrame.
    try:
        df = pd.read_csv(url)
        st.write("DataFrame Head:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
    
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
