import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Palette Configuration for UI Visual Harmony
DARK_THEME_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E0E6ED', family='Outfit, Inter, sans-serif'),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor='#1E293B', font_size=13, font_family='sans-serif')
)

def create_sales_trend_chart(df):
    """
    1. Sales trend: Line chart (Date vs Sales)
    """
    # Aggregate sales by date to prevent overcrowding
    df_agg = df.groupby('Date')['Sales_Value_INR'].sum().reset_index()
    df_agg = df_agg.sort_values('Date')
    
    fig = px.line(
        df_agg, 
        x='Date', 
        y='Sales_Value_INR',
        title='📈 Sales Performance Over Time',
        labels={'Sales_Value_INR': 'Sales Value (₹)', 'Date': 'Timeline'},
        color_discrete_sequence=['#3B82F6'] # Neon Blue
    )
    
    fig.update_traces(line=dict(width=2.5), mode='lines+markers', marker=dict(size=4))
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickprefix='₹')
    
    return fig

def create_state_sales_chart(df):
    """
    2. State-wise sales: Bar chart
    """
    df_agg = df.groupby('State')['Sales_Value_INR'].sum().reset_index()
    df_agg = df_agg.sort_values('Sales_Value_INR', ascending=False)
    
    fig = px.bar(
        df_agg,
        x='State',
        y='Sales_Value_INR',
        title='🗺️ Sales Contribution by State',
        labels={'Sales_Value_INR': 'Total Sales (₹)', 'State': 'State'},
        color='Sales_Value_INR',
        color_continuous_scale=px.colors.sequential.Tealgrn
    )
    
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showgrid=False, linecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickprefix='₹')
    
    return fig

def create_product_revenue_chart(df):
    """
    3. Product-wise revenue: Pie/Donut chart
    """
    df_agg = df.groupby('Product')['Sales_Value_INR'].sum().reset_index()
    df_agg = df_agg.sort_values('Sales_Value_INR', ascending=False)
    
    fig = px.pie(
        df_agg,
        values='Sales_Value_INR',
        names='Product',
        title='🛍️ Revenue Share by Product Category',
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0F172A', width=2)))
    fig.update_layout(**DARK_THEME_LAYOUT)
    
    return fig

def create_monthly_sales_chart(df):
    """
    4. Monthly sales trend: Line graph
    """
    # Parse month names for clean X-axis ordering
    # Group by year and month
    df_copy = df.copy()
    df_copy['Year-Month'] = df_copy['Date'].dt.to_period('M').astype(str)
    
    df_agg = df_copy.groupby('Year-Month')['Sales_Value_INR'].sum().reset_index()
    df_agg = df_agg.sort_values('Year-Month')
    
    fig = px.line(
        df_agg,
        x='Year-Month',
        y='Sales_Value_INR',
        title='📅 Aggregated Monthly Revenue Trends',
        labels={'Sales_Value_INR': 'Revenue (₹)', 'Year-Month': 'Month'},
        color_discrete_sequence=['#10B981'] # Emerald Green
    )
    
    fig.update_traces(line=dict(width=3), mode='lines+markers', marker=dict(size=6))
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickprefix='₹')
    
    return fig

def create_correlation_matrix_chart(df):
    """
    5. Correlation matrix: Heatmap
    """
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out year/month/day/weekend from raw correlation matrix for a cleaner output, or keep them
    # Let's keep the major ones
    keep_cols = [c for c in numeric_cols if c not in ['year', 'month', 'day', 'weekday', 'weekend', 'quarter']]
    # Ensure they exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    # If too few columns, include date features
    if len(keep_cols) < 3:
        keep_cols = numeric_cols
        
    corr = df[keep_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='🔗 Feature Correlation Matrix',
        **DARK_THEME_LAYOUT
    )
    
    return fig

def create_top_products_chart(df):
    """
    6. Top products: Horizontal bar chart
    """
    df_agg = df.groupby('Product')['Sales_Value_INR'].sum().reset_index()
    # Take top 10 products
    df_agg = df_agg.sort_values('Sales_Value_INR', ascending=True).tail(10)
    
    fig = px.bar(
        df_agg,
        y='Product',
        x='Sales_Value_INR',
        orientation='h',
        title='🏆 Top Performing Products by Value',
        labels={'Sales_Value_INR': 'Sales Value (₹)', 'Product': 'Product Category'},
        color='Sales_Value_INR',
        color_continuous_scale=px.colors.sequential.Agsunset
    )
    
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickprefix='₹')
    fig.update_yaxes(showgrid=False, linecolor='rgba(255,255,255,0.1)')
    
    return fig

def create_discount_impact_chart(df):
    """
    7. Discount impact: Scatter plot
    """
    # Group or sample if data is too big, to prevent layout slowdown
    if len(df) > 1000:
        df_sample = df.sample(1000, random_state=42)
    else:
        df_sample = df
        
    fig = px.scatter(
        df_sample,
        x='discount_pct',
        y='Sales_Value_INR',
        color='Product',
        size='Quantity',
        hover_data=['City', 'State'],
        title='🏷️ Discount Percentage vs Sales Value Impact',
        labels={'discount_pct': 'Discount (%)', 'Sales_Value_INR': 'Sales Value (₹)', 'Quantity': 'Volume'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(**DARK_THEME_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', ticksuffix='%')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickprefix='₹')
    
    return fig
