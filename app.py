import streamlit as st
import joblib
import pandas as pd

# Load the models
try:
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    knn_model = joblib.load('knn_model.joblib')
except FileNotFoundError:
    st.error("Error loading model files. Please make sure 'onehot_encoder.joblib', 'minmax_scaler.joblib', and 'knn_model.joblib' are in the same directory.")
    st.stop()

st.title("Aplicación de Predicción con KNN")

st.write("Ingrese los datos para realizar la predicción:")

# Get user input
felder_options = onehot_encoder.categories_[0].tolist()
felder_input = st.selectbox("Seleccione el campo (felder):", felder_options)

examen_admision_input = st.number_input("Ingrese el puntaje del Examen de admisión de Universidad:", min_value=0.0, max_value=100.0, value=50.0)

if st.button("Realizar Predicción"):
    # Create a DataFrame from user input
    data = {'felder': [felder_input], 'Examen_admisión_Universidad': [examen_admision_input]}
    input_df = pd.DataFrame(data)

    # Apply OneHot Encoding
    felder_encoded = onehot_encoder.transform(input_df[['felder']])
    felder_encoded_df = pd.DataFrame(felder_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['felder']))

    # Apply MinMaxScaler
    examen_scaled = minmax_scaler.transform(input_df[['Examen_admisión_Universidad']])
    examen_scaled_df = pd.DataFrame(examen_scaled, columns=['Examen_admisión_Universidad'])

    # Concatenate the processed features
    processed_input = pd.concat([felder_encoded_df, examen_scaled_df], axis=1)

    # Make prediction
    prediction = knn_model.predict(processed_input)

    st.subheader("Resultado de la Predicción:")
    st.write(f"La predicción es: {prediction[0]}")
