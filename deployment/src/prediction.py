import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st


# Load model pipeline
with open('./src/best_pipe.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def run():
    st.title("Prediksi Kanker Berdasarkan Faktor Risiko")
    st.subheader("Isi data berikut untuk memprediksi kemungkinan tingkat keparahan kanker")

    with st.form(key='cancer_prediction_form'):
        name = st.text_input('Nama Anda', value='nama anda')
        genetic_risk = st.slider('Genetic Risk (0-10)', 0, 10, 5)
        air_pollution = st.slider('Air Pollution Level (0-10)', 0, 10, 5)
        alcohol_use = st.slider('Alcohol Use (0-10)', 0, 10, 5)
        smoking = st.slider('Smoking (0-10)', 0, 10, 5)
        obesity_level = st.slider('Obesity Level (0-10)', 0, 10, 5)
        treatment_cost_usd = st.number_input('Estimated Treatment Cost (USD)', min_value=0.0, value=50000.0, step=1000.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            data_inf = {
                'Name': [name],
                'genetic_risk': [genetic_risk],
                'air_pollution': [air_pollution],
                'alcohol_use': [alcohol_use],
                'smoking': [smoking],
                'obesity_level': [obesity_level],
                'treatment_cost_usd': [treatment_cost_usd]
            }

            df_inf = pd.DataFrame(data_inf)
            st.write("Data Input:")
            st.dataframe(df_inf)

            # Prediction
            prediction = model.predict(df_inf)
            st.success(f"Prediksi Model: {prediction[0]:.2f}")

if __name__ == '__main__':
    run()
