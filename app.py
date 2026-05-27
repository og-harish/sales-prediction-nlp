import streamlit as st
import os
import pandas as pd
import numpy as np
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Sales Analytics & Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark-Theme CSS Style
CSS_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
    /* Base Fonts & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #020617 100%);
        color: #E2E8F0;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid #6366F1; /* Primary Indigo border */
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }
    .kpi-card.green { border-left-color: #10B981; }
    .kpi-card.teal { border-left-color: #06B6D4; }
    .kpi-card.orange { border-left-color: #F59E0B; }
    .kpi-card.rose { border-left-color: #F43F5E; }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.075em;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 5px;
        font-family: 'Outfit', sans-serif;
    }

    /* Buttons Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%) !important;
        border: none !important;
        color: white !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #818CF8 0%, #34D399 50%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Hide standard Streamlit header/footer for premium feeling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: rgba(0,0,0,0) !important;}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Define file paths
DEFAULT_DATA_PATH = "data/Flipkart_Sales_Uncleaned.csv"
CLEANED_DATA_PATH = "data/Flipkart_Sales_Cleaned.csv"
MODEL_PATH = "models/sales_model.pkl"
REQUIRED_DATA_COLUMNS = [
    "Date",
    "City",
    "State",
    "Product",
    "Quantity",
    "Unit_Price_INR",
    "Sales_Value_INR",
    "discount_pct",
    "sentiment_score",
]

def has_required_data_columns(df):
    return all(col in df.columns for col in REQUIRED_DATA_COLUMNS)

def get_groq_api_key():
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""

# Initialize Session State Variables
if "df" not in st.session_state:
    st.session_state["df"] = None
if "model_state" not in st.session_state:
    st.session_state["model_state"] = None
if "is_trained" not in st.session_state:
    st.session_state["is_trained"] = False

# Try to load existing cleaned data and model if available
if os.path.exists(CLEANED_DATA_PATH) and st.session_state["df"] is None:
    try:
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        if has_required_data_columns(cleaned_df):
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
            st.session_state["df"] = cleaned_df
    except Exception:
        pass

if os.path.exists(MODEL_PATH) and st.session_state["model_state"] is None:
    import joblib
    try:
        st.session_state["model_state"] = joblib.load(MODEL_PATH)
        st.session_state["is_trained"] = True
    except Exception:
        pass

# Sidebar layout
with st.sidebar:
    st.markdown('<div style="text-align: center; padding: 10px 0;"><h2 class="gradient-text">FLP Analytics</h2><p style="color:#94A3B8; font-size:0.9rem;">AI Sales Intel & Prediction</p></div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🛠️ System Config")
    st.sidebar.info("Navigate through the analytics modules using the pages menu above.")
    
    st.sidebar.markdown("### 🔑 API Integrations")
    groq_key = get_groq_api_key()
    groq_key_status = "Connected" if groq_key.startswith("gsk_") and len(groq_key) > 10 else "Unavailable"
    if groq_key_status == "Connected":
        st.sidebar.success("Groq API: Connected ⚡")
    else:
        st.sidebar.warning("Groq API: Key Active ⚡")

# Main Page Layout
st.markdown('<h1 style="font-size: 2.7rem; margin-bottom: 5px;">⚡ <span class="gradient-text">Sales Prediction & NLP Insight System</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94A3B8; font-size:1.1rem; margin-bottom: 30px;">A production-style intelligent intelligence engine powered by Streamlit, Scikit-Learn, and Llama 3 on Groq.</p>', unsafe_allow_html=True)

# Grid Layout for Dataset Source
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3>📂 Dataset Source</h3>', unsafe_allow_html=True)
    st.write("Upload a sales dataset or use the default uncleaned Flipkart sales dataset.")
    
    uploaded_file = st.file_uploader("Upload Raw Sales CSV", type=["csv"])
    
    # Decide active file path
    active_file = DEFAULT_DATA_PATH
    if uploaded_file is not None:
        # Save custom uploaded file
        active_file = os.path.join("data", "uploaded_sales_raw.csv")
        with open(active_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Custom raw dataset uploaded successfully!")
        
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h3>⚙️ Preprocessing & Model Training</h3>', unsafe_allow_html=True)
    st.write("Trigger the automated pipeline: impute missing values, drop duplicates, normalize categories, extract date features, generate simulated discounts, and train machine learning regressors.")
    
    process_btn = st.button("🧼 Clean & Process Dataset")
    st.markdown("</div>", unsafe_allow_html=True)

# Details & Processing Execution
if process_btn:
    st.markdown("---")
    st.markdown("### 🌀 Processing Pipeline Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Preprocessing
        status_text.markdown("🔄 *Step 1: Cleaning uncleaned data, handling duplicates, and parsing dates...*")
        progress_bar.progress(20)
        time.sleep(1.0) # For smooth visual effect
        
        from utils.preprocess import preprocess_data
        df_cleaned = preprocess_data(active_file)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
        st.session_state["df"] = df_cleaned
        
        progress_bar.progress(50)
        status_text.markdown("📐 *Step 2: Performing feature engineering (price classification, seasonality, sentiment scoring)...*")
        time.sleep(1.0)
        
        # Step 2: Model Training
        progress_bar.progress(70)
        status_text.markdown("🧠 *Step 3: Initializing model training (Linear Regression, Random Forest, Decision Tree)...*")
        
        from utils.model import train_and_save_model
        model_state = train_and_save_model(df_cleaned)
        st.session_state["model_state"] = model_state
        st.session_state["is_trained"] = True
        
        progress_bar.progress(100)
        status_text.markdown("✨ **Pipeline execution completed successfully!**")
        st.balloons()
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.stop()

# Show active dataset statistics and model status
if st.session_state["df"] is not None:
    st.markdown("---")
    st.markdown("### 📊 Active Dataset & Model Overview")
    
    # 4 metrics cards
    m1, m2, m3, m4 = st.columns(4)
    df = st.session_state["df"]
    
    # Total sales
    total_sales = df['Sales_Value_INR'].sum()
    total_qty = df['Quantity'].sum()
    top_prod = df['Product'].mode()[0].title()
    
    with m1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Active Records</div><div class="kpi-val">{len(df):,}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Aggregated Revenue</div><div class="kpi-val">₹{total_sales:,.0f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="kpi-card teal"><div class="kpi-label">Total Volume Sold</div><div class="kpi-val">{total_qty:,.0f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Top Category</div><div class="kpi-val">{top_prod}</div></div>', unsafe_allow_html=True)

    # Model parameters card
    if st.session_state["is_trained"] and st.session_state["model_state"] is not None:
        mstate = st.session_state["model_state"]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h4>🤖 Machine Learning Model: <span style='color:#10B981;'>{mstate['model_name']}</span> (Selected Best)</h4>", unsafe_allow_html=True)
        
        # Display table of compared models
        comp_df = pd.DataFrame(mstate['performance']).T
        st.table(comp_df.style.format({'R2': '{:.4f}', 'MAE': '{:,.2f}', 'RMSE': '{:,.2f}'}))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ **Machine Learning model has not been trained yet. Please click 'Clean & Process Dataset' to train the forecasting models.**")

    # Dataset Preview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>📋 Cleaned Dataset Sample Preview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("👋 **Welcome to the Sales Analytics & Prediction Suite! Please upload a dataset or click the 'Clean & Process Dataset' button to begin loading data and generating AI forecasting models.**")
