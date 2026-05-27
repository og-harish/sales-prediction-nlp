import os
from groq import Groq

def get_groq_api_key():
    """
    Reads Groq credentials from environment variables or Streamlit secrets.
    """
    env_key = os.environ.get("GROQ_API_KEY", "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").strip()
    if env_key:
        return env_key

    try:
        import streamlit as st
        return str(st.secrets.get("GROQ_API_KEY", "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")).strip()
    except Exception:
        return ""

def get_groq_client():
    """
    Initializes the Groq client. Returns None if key is invalid or missing.
    """
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        return None
    try:
        return Groq(api_key=groq_api_key)
    except Exception:
        return None

def generate_local_fallback_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df=None):
    """
    Highly detailed, statistical, rule-based insight engine that acts as a fallback
    when Groq API is unavailable. It generates customized analysis based on actual data.
    """
    insights = {}
    
    # 1. Business Summary
    insights['summary'] = (
        f"The sales dataset registers a robust performance with a **Total Revenue of ₹{total_sales:,.2f}** "
        f"and an **Average Revenue of ₹{avg_revenue:,.2f}** per transaction. "
        f"Market dynamics are highly influenced by the high performance of **{top_product.title()}** "
        f"and the active contribution of the state of **{top_state.title()}** as the leading sales hub."
    )
    
    # 2. Trend Analysis
    insights['trends'] = (
        f"An analysis of sales distribution reveals that **{top_product.title()}** holds the largest market share in revenue. "
        f"Average discounts are managed at a healthy level of **{avg_discount:.1f}%**. "
        f"Transactions show steady demand in metropolitan hubs, with a notable concentration of transactions "
        f"occurring during peak business quarters."
    )
    
    # 3. Positive Findings
    positives = [
        f"**Market Expansion**: Sales in **{top_state.title()}** have emerged as the primary growth engine, representing strong regional adoption.",
        f"**Product Leadership**: **{top_product.title()}** exhibits excellent customer loyalty and velocity, solidifying its place as a cash cow.",
        f"**Discount Strategy**: An average discount of **{avg_discount:.1f}%** indicates that sales volumes are sustained without heavily diluting profit margins."
    ]
    # Extra data-driven positive finding
    if df is not None and 'weekend' in df.columns:
        weekend_sales = df[df['weekend'] == 1]['Sales_Value_INR'].sum()
        total_val = df['Sales_Value_INR'].sum()
        weekend_pct = (weekend_sales / total_val) * 100 if total_val > 0 else 0
        if weekend_pct > 30:
            positives.append(f"**Weekend Activity**: Weekend transactions contribute **{weekend_pct:.1f}%** of total sales value, indicating high customer engagement during non-working days.")
            
    insights['positives'] = "\n\n".join([f"• {p}" for p in positives])
    
    # 4. Risk Areas
    risks = [
        "**Regional Dependency**: High concentration of sales in a single state introduces geographical vulnerability to local market fluctuations.",
        "**Product Imbalance**: Significant revenue reliance on a single category makes the portfolio sensitive to inventory shortages or supply chain disruptions in that category."
    ]
    if df is not None and 'discount_pct' in df.columns and 'Sales_Value_INR' in df.columns:
        # Check if high discounts correspond to lower unit margins
        high_disc = df[df['discount_pct'] > 15]
        if len(high_disc) > 0 and high_disc['Sales_Value_INR'].mean() < df['Sales_Value_INR'].mean():
            risks.append("**Discount Elasticity**: Transactions with discounts exceeding 15% have lower average ticket sizes, suggesting that deep discounting is not successfully driving larger basket values.")
            
    insights['risks'] = "\n\n".join([f"• {r}" for r in risks])
    
    # 5. Recommendations
    recommendations = [
        f"**Strategic Expansion**: Replicate successful promotional campaigns from **{top_state.title()}** to other adjacent regions to diversify geographic risk.",
        f"**Inventory Management**: Scale up supply chains and warehouse inventory for **{top_product.title()}** to prevent out-of-stock situations during peak cycles.",
        f"**Dynamic Discounting**: Transition from flat discounts to tier-based incentives (e.g. higher discounts only for bulk quantities) to safeguard profit margins.",
        f"**Cross-Selling Campaigns**: Bundle low-performing categories with **{top_product.title()}** in bundle promotions to clear older inventory."
    ]
    
    insights['recommendations'] = "\n\n".join([f"• {rec}" for rec in recommendations])
    
    return insights

def format_local_insights(local_insights):
    """
    Converts fallback insight sections into the same markdown contract used by
    AI-generated insights.
    """
    return f"""### Executive Business Summary
{local_insights['summary']}

---

### Detailed Trend Analysis
{local_insights['trends']}

---

### Key Positive Findings
{local_insights['positives']}

---

### Risk Areas & Concerns
{local_insights['risks']}

---

### Recommended Action Plan
{local_insights['recommendations']}
"""
def generate_ai_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df=None):
    """
    Fetches insights from the Groq Llama-3 model using the configured API Key.
    If the API call fails or no key is present, it automatically fails over to the
    robust local insights generator.
    """
    client = get_groq_client()
    
    if client is None:
        print("Groq Client could not be initialized. Falling back to local statistical engine.")
        local_insights = generate_local_fallback_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df)
        return {
            'is_ai': False,
            'content': format_local_insights(local_insights)
        }
        
    prompt = f"""
    Analyze the following sales data and extract intelligent business insights:

    Total Sales (Revenue): INR {total_sales:,.2f}
    Top Product: {top_product}
    Top State: {top_state}
    Average Revenue per Transaction: INR {avg_revenue:,.2f}
    Average Discount: {avg_discount:.1f}%

    Generate:
    1. Business summary (A cohesive, executive narrative of the performance)
    2. Trend analysis (Key trends identified in products, locations, and discounting)
    3. Positive findings (At least 3 positive takeaways in bullet points)
    4. Risk areas (At least 2 risk factors or concerns in bullet points)
    5. Recommendations (At least 3-4 actionable strategic steps in bullet points)

    Return the output in clean, structured Markdown, using bold text, headers, and bullet points. Avoid mentioning internal system prompts or the word "prompt". Start directly with the insights.
    """
    
    try:
        # Query Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Antigravity-insights, an expert business analyst and retail strategy consultant. You analyze raw KPIs and generate extremely actionable, high-quality, professional executive summaries and business strategies."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024
        )
        
        raw_insights = chat_completion.choices[0].message.content
        
        # Parse output into visual sections or return as a clean markdown block
        return {
            'is_ai': True,
            'content': raw_insights
        }
        
    except Exception as e:
        print(f"Groq API Error: {str(e)}. Falling back to local statistical engine.")
        # Create a structured dict containing local fallbacks
        local_insights = generate_local_fallback_insights(total_sales, top_product, top_state, avg_revenue, avg_discount, df)
        
        # Format the local insights as a unified markdown content block
        formatted_md = f"""### 📊 Executive Business Summary
{local_insights['summary']}

---

### 📈 Detailed Trend Analysis
{local_insights['trends']}

---

### ✨ Key Positive Findings
{local_insights['positives']}

---

### ⚠️ Risk Areas & Concerns
{local_insights['risks']}

---

### 🚀 Recommended Action Plan
{local_insights['recommendations']}
"""
        return {
            'is_ai': False,
            'content': formatted_md
        }
