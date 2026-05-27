import streamlit as st
import pandas as pd
import numpy as np
from utils.charts import (
    create_sales_trend_chart,
    create_state_sales_chart,
    create_product_revenue_chart,
    create_monthly_sales_chart,
    create_correlation_matrix_chart,
    create_top_products_chart,
    create_discount_impact_chart
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Analytics Dashboard | FLP AI",
    page_icon="📊",
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
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid #6366F1;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .kpi-card.green { border-left-color: #10B981; }
    .kpi-card.teal { border-left-color: #06B6D4; }
    .kpi-card.orange { border-left-color: #F59E0B; }
    .kpi-card.rose { border-left-color: #F43F5E; }
    
    .kpi-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.075em;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 5px;
        font-family: 'Outfit', sans-serif;
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

# Guard against missing data
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown('<h1 style="font-size: 2.5rem;"><span class="gradient-text">📊 Interactive Analytics Dashboard</span></h1>', unsafe_allow_html=True)
    st.warning("⚠️ **Dataset is not loaded!** Please visit the **Home (app)** page to upload, clean, and process the dataset before viewing the dashboard.")
    st.stop()

df = st.session_state["df"]

# Dashboard Header
st.markdown('<h1 style="font-size: 2.5rem; margin-bottom: 25px;"><span class="gradient-text">📊 Interactive Analytics Dashboard</span></h1>', unsafe_allow_html=True)

# Top KPI Metric Cards (Step 7 Requirement)
k1, k2, k3, k4, k5 = st.columns(5)

# Calculate Metric Values
total_revenue = df['Sales_Value_INR'].sum()
total_quantity = df['Quantity'].sum()
top_product = df['Product'].mode()[0].title()
top_state = df['State'].mode()[0].title()
avg_discount = df['discount_pct'].mean()

with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">💰 Total Revenue</div><div class="kpi-val">₹{total_revenue:,.0f}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card green"><div class="kpi-label">📦 Total Qty Sold</div><div class="kpi-val">{total_quantity:,.0f}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card teal"><div class="kpi-label">🏆 Top Product</div><div class="kpi-val">{top_product}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">📍 Top State</div><div class="kpi-val">{top_state}</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi-card rose"><div class="kpi-label">🏷️ Avg Discount</div><div class="kpi-val">{avg_discount:.1f}%</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Analytics Charts Layout (Grid System)
# Grid 1: Sales Trend & Monthly Sales Trend (2 lines side-by-side or stacked)
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_sales_trend_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_monthly_sales_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# Grid 2: State-wise Sales & Product Revenue Share
c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_state_sales_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_product_revenue_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# Grid 3: Top Products & Discount Impact
c5, c6 = st.columns(2)
with c5:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_top_products_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
with c6:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(create_discount_impact_chart(df), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# Grid 4: Correlation Matrix (Full-Width)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.plotly_chart(create_correlation_matrix_chart(df), width="stretch")
st.markdown('</div>', unsafe_allow_html=True)
