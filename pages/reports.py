import streamlit as st
import pandas as pd
import numpy as np
import datetime
from fpdf import FPDF
from utils.nlp import generate_local_fallback_insights

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Executive Reports | FLP AI",
    page_icon="📋",
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
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Guard against missing data
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown('<h1 style="font-size: 2.5rem;"><span class="gradient-text">📋 Executive Reports & Exporters</span></h1>', unsafe_allow_html=True)
    st.warning("⚠️ **Dataset is not loaded!** Please visit the **Home (app)** page to upload, clean, and process the dataset before exporting reports.")
    st.stop()

df = st.session_state["df"]

# Retrieve statistical metrics
total_sales = df['Sales_Value_INR'].sum()
total_quantity = df['Quantity'].sum()
top_product = df['Product'].mode()[0].title()
top_state = df['State'].mode()[0].title()
avg_discount = df['discount_pct'].mean()
avg_revenue = df['Sales_Value_INR'].mean()

# Reports Header
st.markdown('<h1 style="font-size: 2.5rem; margin-bottom: 25px;"><span class="gradient-text">📋 Executive Reports & Exporters</span></h1>', unsafe_allow_html=True)

# Grid layout for exporter cards
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h3>📊 Export Cleaned Dataset</h3>', unsafe_allow_html=True)
    st.write(
        "Download the complete preprocessed and feature-engineered sales dataset in CSV format. "
        "The file contains normalized categories, parsed dates, price classifications, seasonal labels, "
        "and sentiment polarity scores, fully prepared for external analysis or visualization."
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Download Cleaned CSV Button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV File",
        data=csv_data,
        file_name=f"Flipkart_Sales_Cleaned_{datetime.date.today()}.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)

class PDFReport(FPDF):
    def header(self):
        # Draw elegant borders and header
        self.set_fill_color(15, 23, 42) # Slate 900
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'B', 15)
        self.cell(0, 10, "FLP SALES FORECAST & BI INTELLIGENCE REPORT", ln=True, align='C')
        self.set_font("Arial", 'I', 9)
        self.cell(0, 5, f"Generated automatically on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Antigravity Engine", ln=True, align='C')
        self.ln(12)
        
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')

def generate_pdf_bytes(df, total_sales, total_quantity, top_product, top_state, avg_discount, avg_revenue):
    """
    Dynamically constructs a beautiful, multi-page business intelligence PDF report.
    Translates unicode characters (like Indian Rupee ₹) into standard Latin counterparts (like INR)
    to prevent FPDF character set crashes.
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Body Styling
    pdf.set_text_color(30, 41, 59) # Slate 800
    
    # 1. Title Block
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Executive KPI Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Create Table of KPIs
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(241, 245, 249) # Light grey header
    
    # Column Headers
    pdf.cell(95, 8, "Performance Metrics Indicator", border=1, fill=True)
    pdf.cell(95, 8, "Recorded Aggregate Value", border=1, fill=True, ln=True)
    
    pdf.set_font("Arial", '', 10)
    # Row data (Replacing ₹ with INR to safeguard FPDF)
    pdf.cell(95, 8, "Total Sales Revenue", border=1)
    pdf.cell(95, 8, f"INR {total_sales:,.2f}", border=1, ln=True)
    
    pdf.cell(95, 8, "Total Quantity Sold", border=1)
    pdf.cell(95, 8, f"{total_quantity:,.0f} Units", border=1, ln=True)
    
    pdf.cell(95, 8, "Average Revenue per Transaction", border=1)
    pdf.cell(95, 8, f"INR {avg_revenue:,.2f}", border=1, ln=True)
    
    pdf.cell(95, 8, "Leading Product Category", border=1)
    pdf.cell(95, 8, f"{top_product.upper()}", border=1, ln=True)
    
    pdf.cell(95, 8, "Primary Geographical Hub (State)", border=1)
    pdf.cell(95, 8, f"{top_state.upper()}", border=1, ln=True)
    
    pdf.cell(95, 8, "Average Transactional Discount Rate", border=1)
    pdf.cell(95, 8, f"{avg_discount:.2f}%", border=1, ln=True)
    
    pdf.ln(10)
    
    # 2. Extract detailed insights
    # Check if AI insights are available, if not, generate fallback
    if "insights_data" in st.session_state and st.session_state["insights_data"] is not None:
        insights = st.session_state["insights_data"]
        raw_text = insights['content']
        # Clean markdown symbols for cleaner print
        cleaned_text = raw_text.replace("**", "").replace("###", "").replace("📊", "").replace("📈", "").replace("✨", "").replace("⚠️", "").replace("🚀", "").replace("•", "-")
        # Replace Rupee symbol
        cleaned_text = cleaned_text.replace("₹", "INR")
    else:
        # Fallback local insights
        local_ins = generate_local_fallback_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df)
        # Format cleanly
        cleaned_text = f"""EXECUTIVE BUSINESS SUMMARY:
{local_ins['summary']}

DETAILED TREND ANALYSIS:
{local_ins['trends']}

KEY POSITIVE FINDINGS:
{local_ins['positives'].replace('**', '').replace('•', '-')}

RISK AREAS AND CONCERNS:
{local_ins['risks'].replace('**', '').replace('•', '-')}

RECOMMENDED ACTION PLAN:
{local_ins['recommendations'].replace('**', '').replace('•', '-')}
"""
        cleaned_text = cleaned_text.replace("₹", "INR")

    # Add section title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Business Intelligence & NLP Insights", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Write multi-line text cleanly
    pdf.set_font("Arial", '', 10)
    
    # Split text into lines to avoid overflow
    for line in cleaned_text.split('\n'):
        if line.strip():
            # Check if this line looks like a header
            if line.isupper() or line.startswith("EXECUTIVE") or line.startswith("DETAILED") or line.startswith("KEY") or line.startswith("RISK") or line.startswith("RECOMMENDED"):
                pdf.ln(3)
                pdf.set_font("Arial", 'B', 11)
                pdf.multi_cell(0, 6, line.strip())
                pdf.set_font("Arial", '', 10)
            else:
                pdf.multi_cell(0, 5, line.strip())
        else:
            pdf.ln(2)
            
    # Return binary string
    try:
        # Compatibility between fpdf versions
        pdf_out = pdf.output(dest='S')
        if isinstance(pdf_out, str):
            return bytes(pdf_out, 'latin1')
        return pdf_out
    except Exception:
        # Fallback to standard bytes
        return pdf.output()

with col2:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h3>📋 Compile PDF Executive Report</h3>', unsafe_allow_html=True)
    st.write(
        "Generate and compile an in-memory PDF business intelligence document. "
        "This document integrates a formatted table of key performance indicators (KPIs) and incorporates "
        "the AI-extracted NLP insights (business summary, positive findings, risks, and recommendations) "
        "arranged in an executive-ready layout."
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Generate in-memory PDF
    try:
        pdf_bytes = generate_pdf_bytes(df, total_sales, total_quantity, top_product, top_state, avg_discount, avg_revenue)
        
        st.download_button(
            label="📥 Download PDF Executive Report",
            data=pdf_bytes,
            file_name=f"FLP_Sales_Executive_Report_{datetime.date.today()}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Failed to generate PDF Report: {str(e)}")
        st.info("Ensure the fpdf package is fully installed.")
    st.markdown('</div>', unsafe_allow_html=True)
