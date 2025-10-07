import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the assets (assuming they are in the same directory as the app.py file or specify the full path)
try:
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    knn_model = joblib.load('knn_model.joblib')
except FileNotFoundError:
    st.error("Error: Make sure 'onehot_encoder.joblib', 'minmax_scaler.joblib', and 'knn_model.joblib' are in the same directory as app.py")
    st.stop()

# Set the title of the Streamlit app
st.title('Predicción de Aprobación de Curso')

st.write("Ingrese los datos del estudiante para predecir la aprobación del curso.")

# Get user input
felder_options = onehot_encoder.categories_[0].tolist() # Assuming 'Felder' is the first and only categorical feature
felder_input = st.selectbox('Estilo de Aprendizaje (Felder)', felder_options)
examen_admision_input = st.number_input('Nota Examen de Admisión', min_value=0.0, max_value=5.0, step=0.01)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Felder': [felder_input],
    'Examen_admisión_Universidad': [examen_admision_input]
})

# Preprocess the input data
# Apply one-hot encoding to 'Felder'
input_felder_encoded = onehot_encoder.transform(input_data[['Felder']]).toarray()
input_felder_encoded_df = pd.DataFrame(input_felder_encoded, columns=onehot_encoder.get_feature_names_out(['Felder']))

# Apply minmax scaling to 'Examen_admisión_Universidad'
input_examen_scaled = minmax_scaler.transform(input_data[['Examen_admisión_Universidad']])
input_examen_scaled_df = pd.DataFrame(input_examen_scaled, columns=['Examen_admisión_Universidad_scaled'])


# Concatenate the processed features. Ensure the order of columns matches the training data
# To ensure correct column order, we can create a template of the expected columns from the training data
# (assuming df_encoded from previous steps represents the structure of the training data)
# If the order from onehot_encoder.get_feature_names_out() and the scaled column name is consistent,
# we can concatenate directly. Let's assume the scaled column is the last one as in the notebook.

# Combine the encoded categorical features and the scaled numerical feature
# It's crucial the order of columns here matches the order the model was trained on.
# A robust way is to explicitly list the expected column names in the correct order.
# Based on the notebook, the order was Felder_encoded columns followed by Examen_admisión_Universidad_scaled.
expected_columns = onehot_encoder.get_feature_names_out(['Felder']).tolist() + ['Examen_admisión_Universidad_scaled']

# Create the final processed input DataFrame, ensuring column order
input_processed = pd.concat([input_felder_encoded_df, input_examen_scaled_df], axis=1)

# Reindex to ensure the columns are in the exact order expected by the model
# This assumes the order of columns in expected_columns is correct based on how the model was trained
try:
    input_processed = input_processed[expected_columns]
except KeyError as e:
    st.error(f"Column mismatch during preprocessing: {e}. Ensure the encoder and scaler were trained on data with consistent columns.")
    st.stop()


# Make prediction
if st.button('Predecir'):
    prediction = knn_model.predict(input_processed)

    # Display the prediction
    st.subheader('Resultado de la Predicción:')
    if prediction[0] == 'si':
        st.success('El estudiante probablemente APROBARÁ el curso.')
    else:
        st.error('El estudiante probablemente NO APROBARÁ el curso.')
