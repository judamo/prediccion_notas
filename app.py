import streamlit as st
import pandas as pd
import joblib
import os

# Load the preprocessors and model
@st.cache_resource
def load_resources():
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    standard_scaler = joblib.load('minmax_scaler.joblib')
    model = joblib.load('knn_model.joblib')
    return onehot_encoder, standard_scaler, model

onehot_encoder, standard_scaler, model = load_resources()

st.title('Course Approval Prediction')

st.write('Enter the student information to predict course approval.')

# User inputs
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal'] # Assuming these are the possible values for Felder
felder = st.selectbox('Felder', felder_options)
examen_admision = st.number_input('Examen de admisión Universidad', min_value=0.0, max_value=5.0, step=0.01)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[felder, examen_admision]], columns=['Felder', 'Examen_admisión_Universidad'])

# Preprocess the input data
# Apply one-hot encoding to 'Felder'
felder_encoded = onehot_encoder.transform(input_data[['Felder']])
felder_encoded_df = pd.DataFrame(felder_encoded, columns=onehot_encoder.get_feature_names_out(['Felder']))

# Apply standard scaling to 'Examen_admisión_Universidad'
input_data['Examen_admisión_Universidad_scaled'] = standard_scaler.transform(input_data[['Examen_admisión_Universidad']])

# Concatenate the processed features
processed_input = pd.concat([input_data.drop('Felder', axis=1), felder_encoded_df], axis=1)

# Ensure the columns are in the same order as the training data
# This requires knowing the column order of the training data used for the model
# For simplicity, we will assume the order based on the previous notebook steps
# A more robust solution would save the column order during training
expected_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual','Examen_admisión_Universidad_scaled'] # This should match the order from df_encoded

# Reindex the processed input to match the expected columns, filling missing columns with 0
processed_input = processed_input.reindex(columns=expected_columns, fill_value=0)


# Make prediction
if st.button('Predict'):
    prediction = model.predict(processed_input)
    st.write(f'Predicted Course Approval: {prediction[0]:.2f}')
