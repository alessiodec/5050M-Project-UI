# optimisation.py
import streamlit as st
import pandas as pd
from functions.alfie.optimisation_cr import minimise_cr

def main():
    st.title('Optimisation')
    st.write("This section contains optimisation features.")

    if st.button("Minimise CR for Given d and PCO₂"):
        st.write("Starting Optimisation...")
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        data = pd.read_csv(csv_url)

        d_min, d_max = data["d"].min(), data["d"].max()
        pco2_min, pco2_max = data["PCO2"].min(), data["PCO2"].max()

        d = st.number_input("Enter Pipe Diameter (d):", min_value=d_min, max_value=d_max, step=0.01, value=d_min)
        pco2 = st.number_input("Enter CO₂ Partial Pressure (PCO₂):", min_value=pco2_min, max_value=pco2_max, step=0.001, value=pco2_min)

        if st.button("Run Optimisation"):
            try:
                best_params, min_cr = minimise_cr(d, pco2)
                st.write("✅ **Optimisation Completed!**")
                st.write(f"**Optimal Pipe Diameter (d):** {best_params[0][4]:.3f}")
                st.write(f"**Optimal CO₂ Partial Pressure (PCO₂):** {best_params[0][2]:.3f}")
                st.write(f"**Minimised Corrosion Rate (CR):** {min_cr:.5f}")
            except Exception as e:
                st.error(f"Error running optimisation: {e}")

if __name__ == "__main__":
    main()
