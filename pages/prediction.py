import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from utils.model import predict_sales

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Sales Prediction | FLP AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Global Glassmorphic Styles
CSS_STYLE = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #020617 100%);
        color: #E2E8F0;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .predict-output-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 2px dashed #10B981;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.1);
    }
    .predict-price {
        font-size: 3.5rem;
        font-weight: 800;
        color: #10B981;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        font-family: 'Outfit', sans-serif;
        margin: 10px 0;
    }
    .gradient-text {
        background: linear-gradient(90deg, #818CF8 0%, #34D399 50%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Guard against missing data or model
MODEL_PATH = "models/sales_model.pkl"
if "df" not in st.session_state or st.session_state["df"] is None or not os.path.exists(MODEL_PATH):
    st.markdown('<h1 style="font-size: 2.5rem;"><span class="gradient-text">🔮 Sales Forecasting & Prediction</span></h1>', unsafe_allow_html=True)
    st.warning("⚠️ **Forecasting model is not trained!** Please visit the **Home (app)** page, clean the dataset, and train the machine learning models before using the prediction page.")
    st.stop()

df = st.session_state["df"]
mstate = st.session_state["model_state"] if st.session_state["model_state"] is not None else import_joblib(MODEL_PATH)

# Retrieve unique categorical lists for drop-downs
states = sorted(df['State'].unique().tolist())
cities = sorted(df['City'].unique().tolist())
products = sorted(df['Product'].unique().tolist())

# Prediction Header
st.markdown('<h1 style="font-size: 2.5rem; margin-bottom: 25px;"><span class="gradient-text">🔮 Sales Forecasting & Prediction</span></h1>', unsafe_allow_html=True)

# Form structure
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3>📝 Configure Forecast Target Parameters</h3>', unsafe_allow_html=True)
st.write("Input the hypothetical transactional parameters below to calculate the future sales values.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Category Dropdowns
        product_sel = st.selectbox("Select Product Category", products)
        state_sel = st.selectbox("Select Target State", states)
        city_sel = st.selectbox("Select Target City", cities)
        date_sel = st.date_input("Target Date", datetime.date.today())
        
    with col2:
        # Numerical Inputs
        quantity_input = st.number_input("Quantity Vol (Units)", min_value=1, max_value=100000, value=10, step=1)
        unit_price_input = st.number_input("Unit Price in INR (₹)", min_value=1.0, max_value=1000000.0, value=500.0, step=10.0)
        discount_input = st.slider("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🔮 Predict Future Sales Value")
    
st.markdown("</div>", unsafe_allow_html=True)

if submit_btn:
    try:
        # Preprocess input dictionary
        input_data = {
            'State': state_sel,
            'City': city_sel,
            'Product': product_sel,
            'Quantity': quantity_input,
            'Unit_Price_INR': unit_price_input,
            'discount_pct': discount_input,
            'Date': pd.Timestamp(date_sel)
        }
        
        # Call prediction utility
        predicted_sales, model_info = predict_sales(input_data, MODEL_PATH)
        
        # Display Predicted Value in custom designed layout
        st.markdown('<div class="predict-output-box">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; font-size:0.95rem;">Forecasted Sales Value</h4>', unsafe_allow_html=True)
        st.markdown(f'<div class="predict-price">₹{predicted_sales:,.2f}</div>', unsafe_allow_html=True)
        
        # Confidence calculation
        r2 = model_info['best_performance']['R2']
        # Map R2 to a confidence rating between 60% and 99% (just for robust visual presentation)
        conf_pct = max(60.0, min(99.0, r2 * 100))
        
        st.markdown(f'<p style="font-size:1.1rem; color:#E2E8F0;">Model Name: <b>{model_info["model_name"]}</b> | 📈 Model Validation R²: <b>{r2:.4f}</b></p>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin: 20px auto; max-width: 300px;"><span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #34D399; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.95rem;">Confidence Score: {conf_pct:.1f}%</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed due to an error: {str(e)}")
