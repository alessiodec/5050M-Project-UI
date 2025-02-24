import streamlit as st
import time

st.title("Loading Bar Example")

# Create a progress bar with an initial value of 0%
progress_bar = st.progress(0)
# Create a placeholder to display status text
status_text = st.empty()

# Loop from 0% to 100%
for percent_complete in range(101):
    progress_bar.progress(percent_complete)  # Update the progress bar.
    status_text.text(f"Loading... {percent_complete}%")  # Update status text.
    time.sleep(0.05)  # Pause to simulate a task taking time.

st.success("Loading complete!")
