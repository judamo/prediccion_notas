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
# Assuming 'Felder' is the first and only categorical feature the encoder was fitted on
try:
    felder_options = onehot_encoder.categories_[0].tolist()
except IndexError:
    st.error("Error: onehot_encoder categories not found. Ensure the encoder was fitted correctly.")
    st.stop()

felder_input = st.selectbox('Estilo de Aprendizaje (Felder)', felder_options)
examen_admision_input = st.number_input('Nota Examen de Admisión', min_value=0.0, max_value=5.0, step=0.01)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Felder': [felder_input],
    'Examen_admisión_Universidad': [examen_admision_input]
})

# Preprocess the input data
# Apply one-hot encoding to 'Felder'
# Ensure the input to transform is always 2D
input_felder_encoded = onehot_encoder.transform(input_data[['Felder']]).toarray()
input_felder_encoded_df = pd.DataFrame(input_felder_encoded, columns=onehot_encoder.get_feature_names_out(['Felder']))

# Apply minmax scaling to 'Examen_admisión_Universidad'
# Ensure the input to transform is always 2D
input_examen_scaled = minmax_scaler.transform(input_data[['Examen_admisión_Universidad']])
input_examen_scaled_df = pd.DataFrame(input_examen_scaled, columns=['Examen_admisión_Universidad_scaled'])


# Combine the encoded categorical features and the scaled numerical feature
# It's crucial the order of columns here matches the order the model was trained on.
# A robust way is to explicitly list the expected column names in the correct order.
# Based on the notebook, the order was Felder_encoded columns followed by Examen_admisión_Universidad_scaled.

# Get expected column names from the onehot encoder and add the scaled numerical column name
expected_columns = onehot_encoder.get_feature_names_out(['Felder']).tolist() + ['Examen_admisión_Universidad_scaled']

# Concatenate the dataframes
input_processed = pd.concat([input_felder_encoded_df, input_examen_scaled_df], axis=1)

# Reindex to ensure the columns are in the exact order expected by the model
# This assumes the order of columns in expected_columns is correct based on how the model was trained
try:
    input_processed = input_processed[expected_columns]
except KeyError as e:
    st.error(f"Column mismatch during preprocessing: {e}. Ensure the encoder and scaler were trained on data with consistent columns.")
    st.stop()
except Exception as e:
     st.error(f"An error occurred during column reindexing: {e}")
     st.stop()


# Make prediction
if st.button('Predecir'):
    try:
        prediction = knn_model.predict(input_processed)

        # Display the prediction
        st.subheader('Resultado de la Predicción:')
        if prediction[0] == 'si':
            st.success('El estudiante probablemente APROBARÁ el curso.')
        else:
            st.error('El estudiante probablemente NO APROBARÁ el curso.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
