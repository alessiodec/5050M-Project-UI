import matplotlib.pyplot as plt
import streamlit as st

def input_histogram():
    csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
    
    df = pd.read_csv(csv_url)
    
    cols_to_keep = list(range(0, 5)) # pH, T, PCO2, v, d
    df_subset = df.iloc[:, cols_to_keep].copy() # make new df
    
    df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]]) # log10 PCO2, v, d
    
    # Use .iloc to subset columns 1-5
    df_subset.hist(bins=30, figsize=(12, 8))
    plt.suptitle("Histograms of Inputs", y=0.95)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)
