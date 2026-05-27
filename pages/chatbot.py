import streamlit as st
import pandas as pd
from groq import Groq
from utils.nlp import get_groq_api_key

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Sales Companion | FLP AI",
    page_icon="💬",
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
    .gradient-text {
        background: linear-gradient(90deg, #818CF8 0%, #34D399 50%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    /* Style Chat Avatars and Input */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        padding: 15px !important;
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Chatbot Header
st.markdown('<h1 style="font-size: 2.5rem; margin-bottom: 5px;"><span class="gradient-text">💬 AI Sales Companion</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94A3B8; font-size:1.05rem; margin-bottom: 25px;">Ask data-specific questions about your active sales dataset, forecast patterns, or request general retail business advice.</p>', unsafe_allow_html=True)

# Retrieve Groq API Key
GROQ_API_KEY = get_groq_api_key()

# Initialize Groq Client
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None

client = get_groq_client()

if client is None:
    st.error("⚠️ **Groq API key is invalid or unavailable.** Please set the GROQ_API_KEY environment variable or verify the key.")
    st.stop()

# Build System Prompt Context from live dataset if available
system_prompt = (
    "You are FLP AI Sales Companion, an elite retail strategy consultant and expert data analyst. "
    "You are friendly, professional, analytical, and highly strategic. "
)

if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
    
    # Pre-calculate main metrics to inject in LLM context
    total_sales = df['Sales_Value_INR'].sum()
    total_qty = df['Quantity'].sum()
    top_prod = df['Product'].mode()[0].title()
    top_state = df['State'].mode()[0].title()
    avg_rev = df['Sales_Value_INR'].mean()
    avg_disc = df['discount_pct'].mean()
    unique_products = sorted(df['Product'].unique().tolist())
    unique_states = sorted(df['State'].unique().tolist())
    
    system_prompt += (
        f"\n\nHere is the active sales dataset summary you must use to answer data-specific queries:\n"
        f"- Total Sales Revenue: INR {total_sales:,.2f}\n"
        f"- Total Quantity Sold: {total_qty:,.0f} units\n"
        f"- Best Selling Product Category: {top_prod}\n"
        f"- Leading Sales State (Hub): {top_state}\n"
        f"- Average Revenue per Transaction: INR {avg_rev:,.2f}\n"
        f"- Average Promotional Discount: {avg_disc:.1f}%\n"
        f"- Available Product Catalog: {', '.join(unique_products)}\n"
        f"- Active States covered: {', '.join(unique_states)}\n\n"
        f"Answer all quantitative sales data questions based strictly on these metrics. If the user asks general retail questions, "
        f"use standard industry wisdom, but weave in examples from these active metrics where helpful."
    )
else:
    system_prompt += (
        "\n\nNo active dataset is loaded currently. Prompt the user politely to go to the Home page to "
        "upload, clean, and process the dataset so that you can provide real-time sales calculations. In the meantime, "
        "helpfully answer general retail consulting and business strategy questions."
    )

# Initialize Stateful Chat History
if "messages" not in st.session_state:
    # Set welcome message
    welcome_text = "Greetings! I am your **FLP AI Sales Companion** ⚡. "
    if "df" in st.session_state and st.session_state["df"] is not None:
        welcome_text += "I have successfully analyzed your active sales dataset! Ask me anything about your revenue metrics, best-performing regions, discount strategies, or forecast models."
    else:
        welcome_text += "Please head over to the **Home** page to load and clean your dataset. In the meantime, feel free to ask me general retail consulting or business strategy questions!"
        
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_text}
    ]

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask your sales advisor a question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Query Llama 3 via Groq
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 *Analyzing metrics and formulating response...*")
        
        try:
            # Reconstruct complete conversation history for context
            messages_payload = [{"role": "system", "content": system_prompt}]
            
            # Append last 8 turns of conversation to stay within reasonable context window
            for msg in st.session_state.messages[-8:]:
                messages_payload.append({"role": msg["role"], "content": msg["content"]})
                
            # Call API
            chat_completion = client.chat.completions.create(
                messages=messages_payload,
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=800
            )
            
            response = chat_completion.choices[0].message.content
            message_placeholder.markdown(response)
            
            # Save to session history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"My apologies, I encountered an issue querying the Groq AI service: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
