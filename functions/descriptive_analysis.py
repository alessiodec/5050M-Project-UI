import streamlit as st

def descriptive_analysis(X):

    st.write("Descriptive Statistics:")
    
    # Convert X to a pandas DataFrame before calling .describe()
    X_df = pd.DataFrame(X, columns=df_subset.columns[:5])  # Add column names based on the first 5 columns of df_subset

    # Now you can call .describe() on the DataFrame
    st.write(X_df.describe())
