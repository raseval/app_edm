import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import pickle
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gdown
import zipfile
import os


MODEL_ID = "19xMQlN-8_37fXixm1DuazPWI0ORJx9Xj" 
ZIP_NAME = "modelo.zip"

if not os.path.exists("modelo"):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, ZIP_NAME, quiet=False, use_cookies=True)

    # Verifica si el archivo es un zip válido
    if zipfile.is_zipfile(ZIP_NAME):
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall("modelo")
    else:
        raise RuntimeError("The downloaded file is not a valid ZIP archive.")

# ---- Configuración de la página ----
st.set_page_config(
    page_title="Energy Consumption Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Ocultar el menú y footer por defecto para un look más limpio
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, 'modelo', 'modelo')
test_data_path = os.path.join(BASE_DIR, 'test_data.csv')

# ---- Carga de codificadores ----
@st.cache_resource
def load_encoders():
    with open(os.path.join(BASE_DIR, 'le_sector.pkl'), 'rb') as f:
        encoder_sector = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'le_cp.pkl'), 'rb') as f:
        encoder_postal = pickle.load(f)
    return encoder_sector, encoder_postal

encoder_sector, encoder_postal = load_encoders()

# ---- Carga del modelo ----
@st.cache_resource
def load_model():
    return TabularPredictor.load(modelo_path, require_py_version_match=False)

predictor = load_model()

# ---- Funciones auxiliares ----

def plot_feature_importance(predictor, test_data_path):
    test_data = pd.read_csv(test_data_path)
    importancias = predictor.feature_importance(test_data)
    importancias_sorted = importancias.sort_values('importance', ascending=True).reset_index()
    importancias_sorted.rename(columns={'index': 'feature'}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    barplot = sns.barplot(
        x='importance',
        y='feature',
        data=importancias_sorted,
        palette='Blues_d'
    )
    barplot.set_title('Feature Importance')
    barplot.set_xlabel('Importance')
    barplot.set_ylabel('Features')

    for i, v in enumerate(importancias_sorted['importance']):
        barplot.text(v + importancias_sorted['importance'].max() * 0.01, i, f"{v:.3f}", color='black', va='center')

    st.pyplot(plt.gcf())

def plot_prediction_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Consumption (MWh)')
    ax.set_ylabel('Predicted Consumption (MWh)')
    ax.set_title('Prediction vs Actual Values')
    st.pyplot(fig)

# ---- Barra lateral para navegación ----
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a section:", ["Prediction", "Model Info", "Evaluation", "Methodology"])

# ---- Sección Prediction ----
if section == "Prediction":
    st.header("Energy Consumption Prediction (MWh)")
    st.write("Input the following parameters to estimate energy consumption:")

    with st.form(key="prediction_form", clear_on_submit=False):
        sector = st.selectbox("Economic Sector", encoder_sector.classes_)
        postal_code = st.selectbox("Postal Code", encoder_postal.classes_)
        temperature = st.number_input("Temperature (°C)", value=20.0, step=0.1, format="%.2f")
        precipitation_mm = st.number_input("Precipitation (mm)", value=0.0, step=0.1, format="%.2f")
        date_input = st.date_input("Date", value=datetime.date(2023, 1, 1), min_value=datetime.date(2019, 1, 1))
        submit_button = st.form_submit_button(label="Run Prediction")

    if submit_button:
        encoded_sector = encoder_sector.transform([sector])[0]
        encoded_postal = encoder_postal.transform([postal_code])[0]
        tp = precipitation_mm / 1000
        days_since_start = (date_input - datetime.date(2019, 1, 1)).days
        year = date_input.year

        input_df = pd.DataFrame([{
            't2m': temperature,
            'tp': tp,
            'Sector_Economic_encoded': encoded_sector,
            'Codi_Postal_encoded': encoded_postal,
            'Any': year,
            'days_since_start': days_since_start
        }])

        prediction = predictor.predict(input_df)
        st.success(f"Estimated energy consumption: **{prediction.iloc[0]:,.2f} MWh**")

# ---- Sección Model Info ----
elif section == "Model Info":
    st.header("Feature Importance")
    st.write("This chart shows the contribution of each feature to the model’s predictions.")
    try:
        plot_feature_importance(predictor, test_data_path)
    except Exception as e:
        st.error("Could not compute feature importance. Check your test dataset.")
        st.exception(e)

# ---- Sección Evaluation ----
elif section == "Evaluation":
    st.header("Model Evaluation")
    st.write("Assess model performance with standard regression metrics and visualization.")

    try:
        test_data = pd.read_csv(test_data_path)
        y_true = test_data['Valor']
        X_test = test_data.drop(columns=['Valor'])
        y_pred = predictor.predict(X_test)

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
 
        st.markdown(f"**R²:** {r2:.3f}")
        st.markdown(f"**RMSE:** {rmse:,.2f} MWh")
        st.markdown(f"**MAE:** {mae:,.2f} MWh")

        plot_prediction_vs_actual(y_true, y_pred)

        st.info("The red dashed line indicates perfect predictions. Points closer to this line represent better model accuracy.")

    except Exception as e:
        st.error("Failed to evaluate the model. Please verify your test dataset.")
        st.exception(e)

# ---- Sección Methodology ----
elif section == "Methodology":
    st.header("Project Motivation and Methodology")

    st.markdown("""
    ### Motivation

    Accurate energy consumption forecasting enables improved resource management, cost savings, and environmental benefits.
    This project offers an accessible predictive model tailored to Barcelona's socio-economic and meteorological data.

    ### Data Sources

    - Barcelona energy and sector data from [Open Data BCN](https://opendata-ajuntament.barcelona.cat/ca/)
    - Meteorological data from [ERA5 - Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)

    ### Methodology

    - **Data preprocessing:** Label encoding of categorical features, date transformation to numeric features (`days_since_start`, `year`).
    - **Modeling:** AutoGluon AutoML framework used to build robust ensemble models.
    - **Evaluation:** Performance assessed with R², RMSE, and MAE metrics. Feature importance interpreted for model transparency.

    """)