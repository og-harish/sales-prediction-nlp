import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.nlp import generate_ai_insights

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI NLP Insights | FLP AI",
    page_icon="🧠",
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
    .sentiment-badge {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 10px;
    }
    .sentiment-badge.positive {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
    }
    .sentiment-badge.neutral {
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid #94A3B8;
        color: #CBD5E1;
    }
    .sentiment-badge.negative {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid #F43F5E;
        color: #FB7185;
    }
    .insight-status-tag {
        float: right;
        font-size: 0.8rem;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366F1;
        color: #818CF8;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
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

INSIGHTS_DATA_PATH = "data/Flipkart_Sales_Cleaned.csv"
REQUIRED_INSIGHT_COLUMNS = ["Sales_Value_INR", "Product", "State", "discount_pct"]

def load_saved_insights_data():
    try:
        saved_df = pd.read_csv(INSIGHTS_DATA_PATH)
        if "Date" in saved_df.columns:
            saved_df["Date"] = pd.to_datetime(saved_df["Date"], errors="coerce", format="mixed")
        if all(col in saved_df.columns for col in REQUIRED_INSIGHT_COLUMNS):
            return saved_df
    except Exception:
        return None
    return None

if "df" not in st.session_state or st.session_state["df"] is None:
    loaded_df = load_saved_insights_data()
    if loaded_df is not None:
        st.session_state["df"] = loaded_df

# Guard against missing data
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown('<h1 style="font-size: 2.5rem;"><span class="gradient-text">🧠 AI NLP Insight Extraction</span></h1>', unsafe_allow_html=True)
    st.warning("⚠️ **Dataset is not loaded!** Please visit the **Home (app)** page to upload, clean, and process the dataset before viewing the insights.")
    st.stop()

df = st.session_state["df"]

required_columns = REQUIRED_INSIGHT_COLUMNS
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(
        "The active dataset is missing required insight columns: "
        + ", ".join(missing_columns)
        + ". Please clean and process the dataset again from the Home page."
    )
    st.stop()

if "sentiment_score" not in df.columns:
    df = df.copy()
    df["sentiment_score"] = 0.0

# Insights Header
st.markdown('<h1 style="font-size: 2.5rem; margin-bottom: 25px;"><span class="gradient-text">🧠 AI NLP Insight Extraction</span></h1>', unsafe_allow_html=True)

# Grid Layout: Sentiment Analysis Gauge and Sentiment Overview
col1, col2 = st.columns([1, 2])

# Perform statistical sentiment aggregations
avg_sentiment = df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0.0

if avg_sentiment > 0.15:
    sentiment_label = "Positive"
    sentiment_class = "positive"
    sentiment_desc = "Customers show high satisfaction levels in purchases. The sentiment indicates healthy market demand and favorable reviews on transactions."
elif avg_sentiment < -0.15:
    sentiment_label = "Negative"
    sentiment_class = "negative"
    sentiment_desc = "Feedback trends suggest customer dissatisfaction. Issues could relate to high prices, poor discount expectations, or product issues."
else:
    sentiment_label = "Neutral"
    sentiment_class = "neutral"
    sentiment_desc = "Sentiment is highly balanced. Transactions match standard customer expectations with normal response rates."

with col1:
    st.markdown('<div class="glass-card" style="height: 100%; text-align: center;">', unsafe_allow_html=True)
    st.markdown('<h3>🎭 Transactional Sentiment Gauge</h3>', unsafe_allow_html=True)
    
    # Create beautiful gauge chart in Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average Polar Score", 'font': {'size': 14, 'color': '#94A3B8'}},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#6366F1"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.08)",
            'steps': [
                {'range': [-1, -0.15], 'color': 'rgba(244, 63, 94, 0.15)'},
                {'range': [-0.15, 0.15], 'color': 'rgba(148, 163, 184, 0.15)'},
                {'range': [0.15, 1], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#34D399", 'width': 4},
                'thickness': 0.75,
                'value': avg_sentiment
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='sans-serif'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=220
    )
    st.plotly_chart(fig, width="stretch")
    
    st.markdown(f'<div class="sentiment-badge {sentiment_class}">Overall: {sentiment_label}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h3>🎭 Sentiment & Review Context</h3>', unsafe_allow_html=True)
    st.write(sentiment_desc)
    st.write(
        "Sentiment scores are derived by applying NLP sentiment analysis algorithms (via TextBlob) on customer review patterns "
        "reconstructed from product classifications, discount allowances, and transaction frequencies. This allows "
        "predictive sentiment forecasting to understand consumer reaction trends."
    )
    
    # Show small summary stats of sentiment distribution
    positive_count = len(df[df['sentiment_score'] > 0.15])
    neutral_count = len(df[(df['sentiment_score'] >= -0.15) & (df['sentiment_score'] <= 0.15)])
    negative_count = len(df[df['sentiment_score'] < -0.15])
    
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("😊 Positive Trxs", f"{positive_count} ({positive_count/len(df)*100:.1f}%)")
    s_col2.metric("😐 Neutral Trxs", f"{neutral_count} ({neutral_count/len(df)*100:.1f}%)")
    s_col3.metric("☹️ Negative Trxs", f"{negative_count} ({negative_count/len(df)*100:.1f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Section for AI Groq NLP Insights
st.markdown("### 🤖 Generative AI Business Intelligence")

# KPIs for NLP Analysis
total_sales = df['Sales_Value_INR'].sum()
top_product = df['Product'].mode()[0]
top_state = df['State'].mode()[0]
avg_revenue = df['Sales_Value_INR'].mean()
avg_discount = df['discount_pct'].mean()

# Button to trigger AI generation (or load it from session state so we don't request the API key on every page toggle)
if "insights_data" not in st.session_state:
    st.session_state["insights_data"] = None

st.write("Generate modern business insights including executive summaries, trend analysis, positive discoveries, risks, and recommendations using Groq's high-speed AI engine.")
generate_btn = st.button("🧠 Extract AI Business Insights")

if generate_btn or st.session_state["insights_data"] is not None:
    if generate_btn:
        with st.spinner("🤖 *Antigravity AI is querying Llama 3 via Groq API and formulating strategic business recommendations...*"):
            # Call nlp helper
            insights_state = generate_ai_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df)
            st.session_state["insights_data"] = insights_state
            
    insights = st.session_state["insights_data"]
    
    # Insight box layout
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if insights.get('is_ai', False):
        st.markdown('<span class="insight-status-tag">Groq Llama 3 Active ⚡</span>', unsafe_allow_html=True)
        st.markdown('<h3>🤖 Live Generated AI Insights</h3>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="insight-status-tag" style="background:rgba(245,158,11,0.2); border-color:#F59E0B; color:#FBBF24;">Offline Engine ⚙️</span>', unsafe_allow_html=True)
        st.markdown('<h3>📊 Statistical Business Insights</h3>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(insights['content'])
    st.markdown('</div>', unsafe_allow_html=True)
