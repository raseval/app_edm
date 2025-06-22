import streamlit as st
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
import pickle
import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gdown
import zipfile
import shutil
import plotly.express as px
import plotly.graph_objects as go

MODEL_ID = "19xMQlN-8_37fXixm1DuazPWI0ORJx9Xj"
ZIP_NAME = "modelo.zip"

# Descargar y descomprimir modelo si no existe
if not os.path.exists("modelo"):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, ZIP_NAME, quiet=False, use_cookies=True)

    if zipfile.is_zipfile(ZIP_NAME):
        # Eliminar carpeta existente si ya hay una anterior incompleta
        if os.path.exists("modelo"):
            shutil.rmtree("modelo")

        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall("modelo")
    else:
        raise RuntimeError("The downloaded file is not a valid ZIP archive.")

st.set_page_config(page_title="Barcelona Energy Forecasting", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, 'modelo', 'modelo')
test_data_path = os.path.join(BASE_DIR, 'test_data.csv')

@st.cache_resource
def load_encoders():
    with open(os.path.join(BASE_DIR, 'le_sector.pkl'), 'rb') as f:
        encoder_sector = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'le_cp.pkl'), 'rb') as f:
        encoder_postal = pickle.load(f)
    return encoder_sector, encoder_postal

@st.cache_resource
def load_model():
    return TabularPredictor.load(modelo_path, require_py_version_match=False)

encoder_sector, encoder_postal = load_encoders()
predictor = load_model()

def plot_multi_day_forecast(df_batch):
    fig = px.line(df_batch, x="ds", y="Predicted", markers=True,
                  title="Forecast for Multiple Days",
                  labels={"ds": "Date", "Predicted": "Energy Consumption (MWh)"})
    fig.update_layout(template="simple_white", hovermode="x unified",
                      margin=dict(l=60, r=40, t=60, b=50))
    return fig

def plot_actual_vs_pred(y_true, y_pred):
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig = px.scatter(x=y_true, y=y_pred,
                     labels={'x': 'Actual Energy Consumption (MWh)',
                             'y': 'Predicted Energy Consumption (MWh)'},
                     title="Actual vs Predicted Energy Consumption")
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                             mode='lines', line=dict(color='red', dash='dash'),
                             name='Ideal Prediction'))
    fig.update_layout(template="simple_white", margin=dict(l=60, r=40, t=60, b=50))
    return fig

def plot_feature_importance(importances):
    df = importances.reset_index().sort_values(by='importance')
    fig = px.bar(df, x='importance', y='index', orientation='h',
                 labels={'index': 'Feature', 'importance': 'Importance'},
                 title='Feature Importance')
    fig.update_layout(template="simple_white", margin=dict(l=60, r=40, t=60, b=50))
    return fig

st.title("Energy Consumption Forecast for Barcelona")
st.markdown("""
This interactive dashboard uses machine learning (AutoML via AutoGluon) to predict electricity consumption (in MWh) across Barcelona using meteorological and socio-economic data.
""")

menu = st.tabs(["Prediction", "Multi-Day Forecast", "Evaluation", "Feature Importance", "Methodology & Data"])

with menu[0]:
    st.header("Single Day Prediction")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sector = st.selectbox("Sector Econòmic", encoder_sector.classes_)
        with col2:
            postal = st.selectbox("Codi Postal", encoder_postal.classes_)
        with col3:
            date = st.date_input("Date", value=datetime.date(2023, 1, 1))
        col4, col5 = st.columns(2)
        with col4:
            temp = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        with col5:
            precip = st.slider("Precipitation (mm)", 0.0, 100.0, 0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_input = pd.DataFrame([{
            't2m': temp,
            'tp': precip / 1000,
            'Sector_Economic_encoded': encoder_sector.transform([sector])[0],
            'Codi_Postal_encoded': encoder_postal.transform([postal])[0],
            'Any': date.year,
            'days_since_start': (date - datetime.date(2019, 1, 1)).days
        }])
        prediction = predictor.predict(df_input)
        st.success(f"Predicted Energy Consumption: **{prediction.values[0]:,.2f} MWh**")

with menu[1]:
    st.header("Multi-day Forecast")
    with st.form("forecast_form"):
        col1, col2 = st.columns(2)
        with col1:
            sector = st.selectbox("Sector Econòmic", encoder_sector.classes_, key="batch_sector")
            temp = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        with col2:
            postal = st.selectbox("Codi Postal", encoder_postal.classes_, key="batch_postal")
            precip = st.slider("Precipitation (mm)", 0.0, 100.0, 0.0)
        start_date = st.date_input("Start Date", value=datetime.date(2023, 1, 1))
        days = st.slider("Forecast Days", 1, 30, 7)
        submit_forecast = st.form_submit_button("Generate Forecast")

    if submit_forecast:
        future_dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        df_batch = pd.DataFrame({
            'ds': future_dates,
            't2m': temp,
            'tp': precip / 1000,
            'Sector_Economic_encoded': encoder_sector.transform([sector])[0],
            'Codi_Postal_encoded': encoder_postal.transform([postal])[0],
            'Any': [d.year for d in future_dates],
            'days_since_start': [(d - datetime.date(2019, 1, 1)).days for d in future_dates]
        })
        y_pred = predictor.predict(df_batch.drop(columns=["ds"]))
        df_batch["Predicted"] = y_pred
        fig = plot_multi_day_forecast(df_batch)
        st.plotly_chart(fig, use_container_width=True)

with menu[2]:
    st.header("Model Evaluation")
    try:
        test = pd.read_csv(test_data_path)
        y_true = test['Valor']
        X_test = test.drop(columns=['Valor'])
        y_pred = predictor.predict(X_test)

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)

        st.metric("R²", f"{r2:.3f}")
        st.metric("RMSE (MWh)", f"{rmse:,.2f}")
        st.metric("MAE (MWh)", f"{mae:,.2f}")

        fig = plot_actual_vs_pred(y_true, y_pred)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Evaluation failed.")
        st.exception(e)

with menu[3]:
    st.header("Feature Importance")
    try:
        test_data = pd.read_csv(test_data_path)
        importances = predictor.feature_importance(test_data)
        fig = plot_feature_importance(importances)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Could not compute feature importance.")
        st.exception(e)

with menu[4]:
    st.header("Methodology & Data")
    st.markdown("""
    ### Motivation
    - Accurate energy consumption forecasting improves resource allocation, infrastructure planning, and sustainability.

    ### Data Sources
    - Energy consumption and sector data: [Open Data BCN](https://opendata-ajuntament.barcelona.cat/ca/)
    - Weather data (temperature, precipitation): [ERA5 - Copernicus CDS](https://cds.climate.copernicus.eu/)

    ### Methodology
    - **Preprocessing**: Encoding categorical variables, creating numeric temporal features.
    - **Model**: AutoGluon AutoML for tabular data with ensembling.
    - **Evaluation**: Metrics used include R², RMSE, MAE.
    - **Deployment**: Interactive dashboard built with Streamlit and Plotly.

    ### Authors & References
    - Inspired by: [Artefactory's Streamlit Prophet](https://github.com/artefactory/streamlit_prophet)
    - [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
    - [Barcelona Open Data Portal](https://opendata-ajuntament.barcelona.cat)

    📬 For suggestions or collaborations, contact the author.
    """)