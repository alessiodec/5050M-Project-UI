import matplotlib.pyplot as plt
import streamlit as st

def input_histogram(df_subset):
    # Use .iloc to subset columns 1-5
    df_subset.iloc[:, 0:5].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Histograms of Inputs", y=0.95)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)
