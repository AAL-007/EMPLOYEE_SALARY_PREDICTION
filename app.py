import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os # For checking if files exist

# --- My Employee Salary Prediction App for Internship ---
# This Streamlit application allows users to predict employee income based on provided attributes.
# It showcases my ability to deploy a machine learning model into an interactive web interface.

# --- IMPORTANT FILE PATHS ---
# Ensure these files are in the SAME DIRECTORY as this 'app.py' script.
MODEL_PATH = 'model_pipeline.pkl'
DATA_PATH = 'adult.csv' # Used to extract unique categories for dropdowns in the app

# --- 1. Load the Trained Machine Learning Model ---
# This is the core of the app: loading the model I trained and saved in my Jupyter notebook.
model_pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as file:
            model_pipeline = pickle.load(file)
        st.sidebar.success("Machine Learning Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}. Please ensure '{MODEL_PATH}' is a valid pickled model.")
        st.stop() # Stop the app if the model cannot be loaded, as it's essential.
else:
    st.sidebar.error(f"Error: Model file '{MODEL_PATH}' not found. Please train and save your model in the notebook first by running all cells.")
    st.stop()

# --- 2. Load Original Data to Get Unique Categories for Dropdowns ---
# To make the app user-friendly, I need to populate dropdown menus with valid categories
# that my model was trained on. I'll load 'adult.csv' just for this purpose.
workclass_options = []
education_options = []
marital_status_options = []
marital_status_simplified_options = []
occupation_options = []
relationship_options = []
race_options = []
gender_options = []
native_country_options = []

if os.path.exists(DATA_PATH):
    try:
        original_data_for_categories = pd.read_csv(DATA_PATH)

        # Apply the same basic cleaning and feature engineering as in the notebook
        # to ensure the categories are consistent with what the model expects.
        for col in ['workclass', 'occupation', 'native-country']:
            original_data_for_categories[col] = original_data_for_categories[col].replace('?', 'Unknown')
        original_data_for_categories = original_data_for_categories[~original_data_for_categories['workclass'].isin(['Without-pay', 'Never-worked'])]

        original_data_for_categories['marital_status_simplified'] = original_data_for_categories['marital-status'].replace({
            'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent': 'Married',
            'Never-married': 'Single', 'Divorced': 'Separated/Divorced', 'Separated': 'Separated/Divorced',
            'Widowed': 'Widowed'
        })

        # Extract unique sorted options for each categorical feature
        workclass_options = sorted(original_data_for_categories['workclass'].unique().tolist())
        education_options = sorted(original_data_for_categories['education'].unique().tolist()) # Original education, 'education-num' is direct input
        marital_status_options = sorted(original_data_for_categories['marital-status'].unique().tolist())
        marital_status_simplified_options = sorted(original_data_for_categories['marital_status_simplified'].unique().tolist())
        occupation_options = sorted(original_data_for_categories['occupation'].unique().tolist())
        relationship_options = sorted(original_data_for_categories['relationship'].unique().tolist())
        race_options = sorted(original_data_for_categories['race'].unique().tolist())
        gender_options = sorted(original_data_for_categories['gender'].unique().tolist())
        native_country_options = sorted(original_data_for_categories['native-country'].unique().tolist())

        st.sidebar.success("Data categories loaded successfully from 'adult.csv'!")

    except Exception as e:
        st.sidebar.warning(f"Warning: Could not load '{DATA_PATH}' for category options: {e}. Using dummy options for demonstration.")
        # Fallback: Hardcoded dummy options if data loading fails, to keep the app running.
        workclass_options = ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Unknown']
        education_options = ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-voc', 'Assoc-acdm', '11th', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
        marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
        marital_status_simplified_options = ['Married', 'Single', 'Separated/Divorced', 'Widowed']
        occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Sales', 'Adm-clerical', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces', 'Unknown']
        relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
        race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
        gender_options = ['Male', 'Female']
        native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'India', 'El-Salvador', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Portugal', 'Iran', 'Nicaragua', 'Peru', 'Greece', 'Ecuador', 'France', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands', 'Unknown']

else:
    st.sidebar.warning(f"Error: '{DATA_PATH}' not found for category options. Using hardcoded dummy options for demonstration.")
    # Fallback: Hardcoded dummy options if data file is not found.
    workclass_options = ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Unknown']
    education_options = ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-voc', 'Assoc-acdm', '11th', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
    marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    marital_status_simplified_options = ['Married', 'Single', 'Separated/Divorced', 'Widowed']
    occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Sales', 'Adm-clerical', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces', 'Unknown']
    relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
    race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    gender_options = ['Male', 'Female']
    native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'India', 'El-Salvador', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Portugal', 'Iran', 'Nicaragua', 'Peru', 'Greece', 'Ecuador', 'France', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands', 'Unknown']


# --- 3. Streamlit App Layout and User Interface ---

st.set_page_config(page_title="Employee Salary Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("💰 Employee Salary Prediction App")
st.markdown("---")
st.markdown("""
This interactive application is part of my **IBM SkillsBuild 6-Week Virtual Internship**.
It leverages a machine learning model to predict whether an individual's annual income is
**<=50K** or **>50K** based on various demographic and employment attributes from the Adult Census Income Dataset.

**Key Features Demonstrated:**
- **Data Preprocessing & Feature Engineering:** Handling raw data and creating insightful features.
- **Machine Learning Modeling:** Training and evaluating a robust classification model.
- **Interactive Web Deployment:** Presenting the model's capabilities through a user-friendly interface using Streamlit.
""")
st.markdown("---")

# Navigation in the sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose a mode", ["Single Prediction", "Batch Prediction"])
st.sidebar.markdown("---")
st.sidebar.info("This application is a testament to my end-to-end machine learning project development skills.")


# --- Single Prediction Mode ---
if app_mode == "Single Prediction":
    st.header("👤 Predict Single Employee Income")
    st.markdown("Enter an individual's details below to get an instant income prediction.")

    # Using Streamlit columns to organize inputs cleanly
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 17, 90, 30, help="Age of the individual.")
        workclass = st.selectbox("Workclass", workclass_options, help="Type of employer (e.g., Private, Self-emp-not-inc).")
        education_num = st.slider("Education Num (Years of Education)", 1, 16, 9, help="Number of years of education (e.g., 9 for HS-grad, 13 for Bachelors).")

    with col2:
        marital_status = st.selectbox("Marital Status", marital_status_options, help="Marital status (e.g., Married-civ-spouse, Never-married).")
        occupation = st.selectbox("Occupation", occupation_options, help="Specific occupation (e.g., Prof-specialty, Craft-repair).")
        relationship = st.selectbox("Relationship", relationship_options, help="Relationship status (e.g., Husband, Not-in-family).")

    with col3:
        race = st.selectbox("Race", race_options, help="Racial category (e.g., White, Black).")
        gender = st.selectbox("Gender", gender_options, help="Biological sex (Male/Female).")
        hours_per_week = st.slider("Hours Per Week", 1, 99, 40, help="Number of hours worked per week.")
        native_country = st.selectbox("Native Country", native_country_options, help="Country of origin (e.g., United-States, Mexico).")

    st.markdown("---")
    st.subheader("Financial Details (Optional)")
    col_fin1, col_fin2 = st.columns(2)
    with col_fin1:
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0, step=100, help="Capital gains from investments.")
    with col_fin2:
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0, step=100, help="Capital losses.")

    # --- FIX APPLIED: Calculate engineered features explicitly here BEFORE creating input_data dictionary ---
    calculated_capital_gain_loss_diff = capital_gain - capital_loss

    # Define the mapping for marital status simplification
    marital_status_mapping = {
        'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Married',
        'Never-married': 'Single',
        'Divorced': 'Separated/Divorced',
        'Separated': 'Separated/Divorced',
        'Widowed': 'Widowed'
    }
    # Get the simplified marital status. Use .get() with a fallback to the original if not found
    calculated_marital_status_simplified = marital_status_mapping.get(marital_status, marital_status)

    # Create a Pandas DataFrame from the user inputs.
    # It's crucial that the column names and order match what the model's pipeline expects.
    input_data = {
        'age': age,
        'workclass': workclass,
        'educational-num': education_num, # <--- CHANGED TO 'educational-num'
        'marital-status': marital_status, # Original marital status is passed for consistency with X.drop() in notebook
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country,
        # Engineered features are now correctly referenced from pre-calculated variables
        'capital_gain_loss_diff': calculated_capital_gain_loss_diff,
        'marital_status_simplified': calculated_marital_status_simplified
    }
    input_df = pd.DataFrame([input_data])


    if st.button("Predict Income", help="Click to get the income prediction for the entered details."):
        if model_pipeline:
            try:
                # Predict using the loaded pipeline. It handles all preprocessing automatically.
                prediction_encoded = model_pipeline.predict(input_df)[0]
                prediction_proba = model_pipeline.predict_proba(input_df)[0]

                # Convert the numerical prediction (0 or 1) back to meaningful labels.
                prediction_label = ">50K" if prediction_encoded == 1 else "<=50K"

                st.success(f"### Predicted Income: **{prediction_label}**")
                st.info(f"Probability of being <=50K: {prediction_proba[0]:.2f}")
                st.info(f"Probability of being >50K: {prediction_proba[1]:.2f}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input fields are filled correctly and match the model's expected features. Refer to the dataset documentation for valid ranges/categories.")
        else:
            st.warning("Model not loaded. Cannot perform prediction. Please check server logs.")

# --- Batch Prediction Mode ---
elif app_mode == "Batch Prediction":
    st.header("📂 Batch Prediction")
    st.markdown("Upload a CSV file containing employee data for batch income prediction. The CSV must have the same column names as the original dataset (excluding 'income' and 'fnlwgt').")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload a CSV with employee data for bulk predictions. Example format available on GitHub.")

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.dataframe(batch_data.head())

        if st.button("Generate Batch Predictions", help="Click to process the uploaded CSV and get predictions."):
            if model_pipeline:
                try:
                    # Apply the same preprocessing and feature engineering steps to the batch data
                    # as done during model training in the notebook.
                    batch_data_processed = batch_data.copy()

                    # Handle '?' in categorical columns for uploaded batch data
                    for col in ['workclass', 'occupation', 'native-country']:
                        if col in batch_data_processed.columns:
                            batch_data_processed[col] = batch_data_processed[col].replace('?', 'Unknown')

                    # Filter out rows with 'Without-pay'/'Never-worked' if they exist in batch_data_processed
                    if 'workclass' in batch_data_processed.columns:
                        batch_data_processed = batch_data_processed[~batch_data_processed['workclass'].isin(['Without-pay', 'Never-worked'])]

                    # Recreate engineered features for batch data
                    if 'capital-gain' in batch_data_processed.columns and 'capital-loss' in batch_data_processed.columns:
                        batch_data_processed['capital_gain_loss_diff'] = batch_data_processed['capital-gain'] - batch_data_processed['capital-loss']
                    else:
                        st.warning("Warning: 'capital-gain' or 'capital-loss' columns missing in uploaded CSV. Engineered feature 'capital_gain_loss_diff' might be missing.")
                        batch_data_processed['capital_gain_loss_diff'] = 0 # Default to 0 if columns are missing

                    if 'marital-status' in batch_data_processed.columns:
                        batch_data_processed['marital_status_simplified'] = batch_data_processed['marital-status'].replace({
                            'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent': 'Married',
                            'Never-married': 'Single', 'Divorced': 'Separated/Divorced', 'Separated': 'Separated/Divorced',
                            'Widowed': 'Widowed'
                        })
                    else:
                        st.warning("Warning: 'marital-status' column missing in uploaded CSV. Engineered feature 'marital_status_simplified' might be missing.")
                        batch_data_processed['marital_status_simplified'] = 'Unknown' # Default to 'Unknown' if column is missing


                    # Make predictions using the pipeline. It handles preprocessing automatically.
                    batch_preds_encoded = model_pipeline.predict(batch_data_processed)
                    batch_pred_probs = model_pipeline.predict_proba(batch_data_processed)

                    # Map encoded predictions back to original labels for the output CSV.
                    batch_data_processed['Predicted_Income'] = np.where(batch_preds_encoded == 1, '>50K', '<=50K')
                    batch_data_processed['Probability_<=50K'] = batch_pred_probs[:, 0]
                    batch_data_processed['Probability_>50K'] = batch_pred_probs[:, 1]

                    st.subheader("✅ Predictions Generated:")
                    st.dataframe(batch_data_processed.head(10)) # Show top 10 rows of predicted data

                    # Provide a download button for the results CSV.
                    csv_output = batch_data_processed.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_output,
                        file_name='predicted_employee_income.csv',
                        mime='text/csv',
                        help="Download the CSV file with your uploaded data and new predictions."
                    )
                except KeyError as e:
                    st.error(f"Missing expected column in uploaded CSV: {e}. Please ensure your CSV file has all required columns.")
                    st.warning("Required columns for prediction: 'age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'.")
                except Exception as e:
                    st.error(f"An error occurred during batch prediction: {e}")
                    st.warning("Please ensure your uploaded CSV's format matches the original dataset and all required columns are present.")
            else:
                st.warning("Model not loaded. Cannot perform batch prediction. Please check server logs.")

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    Developed with ❤️ for a Freelancing Data Science Portfolio.
    <br>
    Data Source: <a href="https://archive.ics.uci.edu/dataset/2/adult" target="_blank">Adult Census Income Dataset</a>
    <br>
    Explore the code and project details on my <a href="YOUR_GITHUB_REPO_LINK_HERE" target="_blank">GitHub Repository</a>.
    (Remember to replace 'YOUR_GITHUB_REPO_LINK_HERE' with your actual GitHub link!)
</div>
""", unsafe_allow_html=True)