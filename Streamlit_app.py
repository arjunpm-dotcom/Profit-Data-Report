"""
Business Intelligence Dashboard - Streamlit Version
Premium design with full analytics functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
from io import StringIO

# Import configuration and data service
from config import BRANCH_RBM_BDM_MAPPING, BRANCH_DISTRICT_MAPPING, DISTRICT_STATE_MAPPING, GOOGLE_SHEET_URL

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS FOR PREMIUM DESIGN ==========
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .kpi-icon {
        font-size: 2rem;
        opacity: 0.8;
    }
    
    /* Revenue Card */
    .kpi-revenue { border-left-color: #3b82f6; }
    .kpi-revenue .kpi-icon { color: #3b82f6; }
    
    /* Profit Card */
    .kpi-profit { border-left-color: #10b981; }
    .kpi-profit .kpi-icon { color: #10b981; }
    
    /* Quantity Card */
    .kpi-quantity { border-left-color: #f59e0b; }
    .kpi-quantity .kpi-icon { color: #f59e0b; }
    
    /* Margin Card */
    .kpi-margin { border-left-color: #8b5cf6; }
    .kpi-margin .kpi-icon { color: #8b5cf6; }
    
    /* Discount Card */
    .kpi-discount { border-left-color: #06b6d4; }
    .kpi-discount .kpi-icon { color: #06b6d4; }
    
    /* Stores Card */
    .kpi-stores { border-left-color: #ef4444; }
    .kpi-stores .kpi-icon { color: #ef4444; }
    
    /* Insight Cards */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        margin-bottom: 0.75rem;
    }
    
    .insight-card.positive { border-left-color: #10b981; }
    .insight-card.trend { border-left-color: #3b82f6; }
    .insight-card.highlight { border-left-color: #f59e0b; }
    .insight-card.alert { border-left-color: #ef4444; }
    
    .insight-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .insight-text {
        font-size: 0.95rem;
        color: #1e293b;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Comparison Cards */
    .comparison-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .comparison-card.positive {
        border: 2px solid #10b981;
    }
    
    .comparison-card.negative {
        border: 2px solid #ef4444;
    }
    
    .comparison-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e293b;
    }
    
    .comparison-growth {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .comparison-growth.positive { color: #10b981; }
    .comparison-growth.negative { color: #ef4444; }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Custom Metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Data Table */
    .dataframe {
        font-size: 0.85rem !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #667eea;
        color: white;
        margin-right: 0.5rem;
    }
    
    .status-badge.success { background: #10b981; }
    .status-badge.warning { background: #f59e0b; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== HELPER FUNCTIONS ==========
def format_indian_currency(value):
    """Format number in Indian currency format"""
    if pd.isna(value) or value == 0:
        return "₹0"
    
    value = float(value)
    sign = "" if value >= 0 else "-"
    abs_value = abs(value)
    
    if abs_value >= 10000000:  # 1 Crore
        return f"{sign}₹{abs_value/10000000:,.2f} Cr"
    elif abs_value >= 100000:  # 1 Lakh
        return f"{sign}₹{abs_value/100000:,.2f} Lakh"
    elif abs_value >= 1000:  # 1 Thousand
        return f"{sign}₹{abs_value/1000:,.2f} K"
    else:
        return f"{sign}₹{abs_value:,.2f}"

def format_indian_number(value):
    """Format any number in Indian format"""
    if pd.isna(value) or value == 0:
        return "0"
    
    value = float(value)
    sign = "" if value >= 0 else "-"
    abs_value = abs(value)
    
    if abs_value >= 10000000:
        return f"{sign}{abs_value/10000000:,.2f} Cr"
    elif abs_value >= 100000:
        return f"{sign}{abs_value/100000:,.2f} Lakh"
    elif abs_value >= 1000:
        return f"{sign}{abs_value/1000:,.2f} K"
    else:
        return f"{sign}{abs_value:,.0f}"

def calculate_growth(current, previous):
    """Calculate percentage growth"""
    if previous == 0:
        return 100 if current > 0 else 0 if current == 0 else -100
    return ((current - previous) / abs(previous)) * 100

def add_rbm_bdm_columns(df):
    """Add RBM and BDM columns"""
    if 'Branch' in df.columns:
        rbm_map = {branch: info['RBM'] for branch, info in BRANCH_RBM_BDM_MAPPING.items()}
        bdm_map = {branch: info['BDM'] for branch, info in BRANCH_RBM_BDM_MAPPING.items()}
        df['RBM'] = df['Branch'].map(rbm_map).fillna('NOT ASSIGNED')
        df['BDM'] = df['Branch'].map(bdm_map).fillna('NOT ASSIGNED')
    return df

def add_location_columns(df):
    """Add District and State columns"""
    if 'Branch' in df.columns:
        df['District'] = df['Branch'].map(BRANCH_DISTRICT_MAPPING).fillna('NOT ASSIGNED')
        df['State'] = df['District'].map(DISTRICT_STATE_MAPPING).fillna('NOT ASSIGNED')
    return df

# ========== DATA LOADING ==========
@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    """Load data from Google Sheets with caching"""
    try:
        response = requests.get(GOOGLE_SHEET_URL, timeout=300)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Process date column
        if 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df['Month'], format='%m/%d/%Y', errors='coerce')
            if df['Date'].isna().all():
                df['Date'] = pd.to_datetime(df['Month'], errors='coerce')
            
            df = df.dropna(subset=['Date'])
            df['Year'] = df['Date'].dt.year.astype(int)
            df['Month_Num'] = df['Date'].dt.month.astype(int)
            df['Month_Short'] = df['Date'].dt.strftime('%b')
            df['Month_Year'] = df['Date'].dt.strftime('%b %Y')
            
            # Financial Year - FIXED for large datasets
            years = df['Date'].dt.year.values
            months = df['Date'].dt.month.values
            fy_start_year = np.where(months >= 4, years, years - 1)
            fy_end_year_short = (fy_start_year + 1) % 100
            
            # Create Financial Year string properly (avoiding numpy string concatenation issues)
            df['Financial_Year'] = ['FY ' + str(int(s)) + '-' + str(int(e)).zfill(2) 
                                    for s, e in zip(fy_start_year, fy_end_year_short)]
            df['FY_Label'] = df['Financial_Year']
            
            # Quarter - fixed for large datasets
            quarter_nums = ((months - 1) // 3) + 1
            df['Quarter'] = ['Q' + str(int(q)) for q in quarter_nums]
            
            # Financial Quarter - fixed
            adjusted_month = np.where(months >= 4, months - 3, months + 9)
            fq_nums = ((adjusted_month - 1) // 3) + 1
            df['Financial_Quarter'] = ['FQ' + str(int(q)) for q in fq_nums]
        
        # Clean numeric columns
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['qty', 'price', 'profit', 'discount', 'value']):
                try:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    pass
        
        # Rename columns
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'qty' in col_lower:
                column_mapping[col] = 'QTY'
            elif 'sold' in col_lower or 'price' in col_lower:
                column_mapping[col] = 'Sold_Price'
            elif 'profit' in col_lower:
                column_mapping[col] = 'Profit'
            elif 'discount' in col_lower:
                column_mapping[col] = 'Discount'
            elif 'branch' in col_lower:
                column_mapping[col] = 'Branch'
            elif 'brand' in col_lower:
                column_mapping[col] = 'Brand'
            elif 'product' in col_lower and 'code' not in col_lower:
                column_mapping[col] = 'Product'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Add RBM/BDM and Location
        df = add_rbm_bdm_columns(df)
        df = add_location_columns(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_kpis(df):
    """Calculate KPI metrics"""
    revenue = float(df['Sold_Price'].sum()) if 'Sold_Price' in df.columns else 0.0
    profit = float(df['Profit'].sum()) if 'Profit' in df.columns else 0.0
    quantity = float(df['QTY'].sum()) if 'QTY' in df.columns else 0.0
    discount = float(df['Discount'].sum()) if 'Discount' in df.columns else 0.0
    
    margin = (profit / revenue * 100) if revenue > 0 else 0.0
    discount_pct = (discount / revenue * 100) if revenue > 0 else 0.0
    
    stores = df['Branch'].nunique() if 'Branch' in df.columns else 0
    states = df['State'].nunique() if 'State' in df.columns else 0
    districts = df['District'].nunique() if 'District' in df.columns else 0
    
    return {
        'revenue': revenue,
        'revenue_formatted': format_indian_currency(revenue),
        'profit': profit,
        'profit_formatted': format_indian_currency(profit),
        'quantity': quantity,
        'quantity_formatted': format_indian_number(quantity),
        'discount': discount,
        'discount_formatted': format_indian_currency(discount),
        'margin': round(margin, 1),
        'discount_pct': round(discount_pct, 1),
        'stores': stores,
        'states': states,
        'districts': districts,
        'records': len(df)
    }

def generate_insights(df):
    """Generate AI-style insights"""
    insights = {
        'top_performer': 'No data available',
        'growth_trend': 'Load data to see trends',
        'highlight': 'Apply filters to view insights',
        'alert': 'No alerts'
    }
    
    if df.empty:
        return insights
    
    try:
        # Top Performer
        if 'Branch' in df.columns and 'Sold_Price' in df.columns:
            branch_revenue = df.groupby('Branch')['Sold_Price'].sum().sort_values(ascending=False)
            if len(branch_revenue) > 0:
                top_branch = branch_revenue.index[0]
                top_revenue = format_indian_currency(branch_revenue.iloc[0])
                insights['top_performer'] = f"{top_branch} leads with {top_revenue} revenue"
        
        # Growth Trend
        if 'Month_Year' in df.columns and 'Sold_Price' in df.columns:
            monthly = df.groupby('Month_Year')['Sold_Price'].sum()
            if len(monthly) >= 2:
                last_month_val = monthly.iloc[-1]
                prev_month_val = monthly.iloc[-2]
                growth = calculate_growth(last_month_val, prev_month_val)
                direction = "up" if growth > 0 else "down"
                insights['growth_trend'] = f"Revenue is {direction} {abs(growth):.1f}% vs previous month"
        
        # Highlight - Best RBM
        if 'RBM' in df.columns and 'Profit' in df.columns:
            rbm_profit = df.groupby('RBM').agg({'Profit': 'sum', 'Sold_Price': 'sum'})
            rbm_profit['Margin'] = (rbm_profit['Profit'] / rbm_profit['Sold_Price'] * 100).round(1)
            rbm_profit = rbm_profit[rbm_profit['Sold_Price'] > 0].sort_values('Margin', ascending=False)
            if len(rbm_profit) > 0:
                best_rbm = rbm_profit.index[0]
                best_margin = rbm_profit['Margin'].iloc[0]
                insights['highlight'] = f"RBM {best_rbm} has the best margin at {best_margin}%"
        
        # Alert
        if 'Branch' in df.columns and 'Profit' in df.columns and 'Sold_Price' in df.columns:
            branch_perf = df.groupby('Branch').agg({'Profit': 'sum', 'Sold_Price': 'sum'})
            branch_perf['Margin'] = (branch_perf['Profit'] / branch_perf['Sold_Price'] * 100).round(1)
            low_margin = branch_perf[branch_perf['Margin'] < 5]
            if len(low_margin) > 0:
                insights['alert'] = f"{len(low_margin)} branches have margin below 5%"
            else:
                insights['alert'] = "All branches performing above minimum margin"
    except:
        pass
    
    return insights

# ========== CHART FUNCTIONS ==========
def create_monthly_trend_chart(df):
    """Create monthly trend chart"""
    if 'Month_Year' not in df.columns:
        return None
    
    monthly = df.groupby('Month_Year').agg({
        'Sold_Price': 'sum',
        'Profit': 'sum',
        'QTY': 'sum'
    }).reset_index()
    
    monthly['Date'] = pd.to_datetime(monthly['Month_Year'], format='%b %Y', errors='coerce')
    monthly = monthly.sort_values('Date')
    monthly['Revenue_Cr'] = monthly['Sold_Price'] / 10000000
    monthly['Profit_Cr'] = monthly['Profit'] / 10000000
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=monthly['Month_Year'], y=monthly['Revenue_Cr'], 
               name='Revenue (₹ Cr)', marker_color='#3b82f6'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=monthly['Month_Year'], y=monthly['Profit_Cr'],
                   name='Profit (₹ Cr)', mode='lines+markers',
                   line=dict(color='#10b981', width=3),
                   marker=dict(size=8)),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Monthly Revenue & Profit Trend',
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#475569')
    )
    fig.update_xaxes(gridcolor='#e2e8f0', linecolor='#e2e8f0')
    fig.update_yaxes(title_text='Revenue (₹ Cr)', secondary_y=False, gridcolor='#e2e8f0')
    fig.update_yaxes(title_text='Profit (₹ Cr)', secondary_y=True)
    
    return fig

def create_rbm_performance_chart(df):
    """Create RBM performance chart"""
    if 'RBM' not in df.columns:
        return None
    
    rbm_data = df.groupby('RBM').agg({
        'Sold_Price': 'sum',
        'Profit': 'sum',
        'QTY': 'sum'
    }).reset_index()
    
    rbm_data = rbm_data.sort_values('Sold_Price', ascending=False)
    rbm_data['Revenue_Cr'] = rbm_data['Sold_Price'] / 10000000
    rbm_data['Profit_Margin'] = (rbm_data['Profit'] / rbm_data['Sold_Price'] * 100).round(1)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = ['#10b981' if m > 15 else '#f59e0b' if m > 10 else '#ef4444' 
              for m in rbm_data['Profit_Margin']]
    
    fig.add_trace(
        go.Bar(x=rbm_data['RBM'], y=rbm_data['Revenue_Cr'],
               name='Revenue (₹ Cr)', marker_color=colors),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=rbm_data['RBM'], y=rbm_data['Profit_Margin'],
                   name='Profit Margin %', mode='lines+markers',
                   line=dict(color='#8b5cf6', width=3),
                   marker=dict(size=10)),
        secondary_y=True
    )
    
    fig.update_layout(
        title='RBM Performance: Revenue vs Margin',
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#475569')
    )
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(title_text='Revenue (₹ Cr)', secondary_y=False, gridcolor='#e2e8f0')
    fig.update_yaxes(title_text='Profit Margin %', secondary_y=True)
    
    return fig

def create_geographic_chart(df):
    """Create geographic pie chart"""
    if 'State' not in df.columns:
        return None
    
    state_data = df.groupby('State')['Sold_Price'].sum().reset_index()
    state_data = state_data.sort_values('Sold_Price', ascending=False)
    state_data['Revenue_Cr'] = state_data['Sold_Price'] / 10000000
    
    fig = px.pie(state_data, values='Revenue_Cr', names='State',
                 title='Revenue Distribution by State',
                 hole=0.4,
                 color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4'])
    
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#475569')
    )
    
    return fig

def create_product_chart(df):
    """Create top products chart"""
    if 'Product' not in df.columns:
        return None
    
    product_data = df.groupby('Product').agg({
        'Sold_Price': 'sum',
        'Profit': 'sum',
        'QTY': 'sum'
    }).reset_index()
    
    product_data = product_data.sort_values('Profit', ascending=False).head(20)
    product_data = product_data.sort_values('Profit', ascending=True)  # For horizontal bar
    product_data['Profit_Lakh'] = product_data['Profit'] / 100000
    product_data['Product_Short'] = product_data['Product'].apply(
        lambda x: str(x)[:30] + '...' if len(str(x)) > 30 else str(x)
    )
    
    colors = ['#10b981' if p > 0 else '#ef4444' for p in product_data['Profit_Lakh']]
    
    fig = go.Figure(go.Bar(
        x=product_data['Profit_Lakh'],
        y=product_data['Product_Short'],
        orientation='h',
        marker_color=colors,
        text=[f'₹{p:.1f}L' for p in product_data['Profit_Lakh']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Top 20 Products by Profit',
        height=550,
        xaxis_title='Profit (₹ Lakh)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#475569'),
        margin=dict(l=200)
    )
    fig.update_xaxes(gridcolor='#e2e8f0')
    
    return fig

def create_district_map_chart(df):
    """Create district performance chart"""
    if 'District' not in df.columns:
        return None
    
    district_data = df.groupby('District').agg({
        'Sold_Price': 'sum',
        'Profit': 'sum',
        'Branch': 'nunique'
    }).reset_index()
    
    district_data = district_data.sort_values('Sold_Price', ascending=False).head(15)
    district_data['Revenue_Cr'] = district_data['Sold_Price'] / 10000000
    district_data['Margin'] = (district_data['Profit'] / district_data['Sold_Price'] * 100).round(1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=district_data['District'],
        y=district_data['Revenue_Cr'],
        name='Revenue (₹ Cr)',
        marker_color=[
            '#10b981' if m > 15 else '#f59e0b' if m > 10 else '#ef4444'
            for m in district_data['Margin']
        ],
        text=[f'{m}%' for m in district_data['Margin']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Top Districts by Revenue (colored by margin)',
        height=450,
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#475569')
    )
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0', title='Revenue (₹ Cr)')
    
    return fig

# ========== MAIN APPLICATION ==========
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 BUSINESS INTELLIGENCE DASHBOARD</h1>
        <p>Advanced Analytics • Multi-Select Filters • Complete Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Data
    with st.spinner('🔄 Loading data from Google Sheets... (This may take 1-2 minutes for large datasets)'):
        df = load_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Please check your internet connection and try again.")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()
        return
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 🎛️ DASHBOARD CONTROLS")
        
        # Mode Toggle
        mode = st.radio("Analysis Mode", ["📈 Analytics", "⚖️ Comparison"], horizontal=True)
        
        st.markdown("---")
        
        if "Analytics" in mode:
            # Time Period Filters
            st.markdown("#### 📅 Time Period")
            period_type = st.selectbox("Period Type", ["All Time", "Calendar Year", "Financial Year", "Quarter"])
            
            year_filter = None
            fy_filter = None
            quarter_filter = None
            
            if period_type == "Calendar Year":
                years = sorted(df['Year'].unique(), reverse=True)
                year_filter = st.selectbox("Year", years)
            elif period_type == "Financial Year":
                fys = sorted(df['FY_Label'].unique(), reverse=True)
                fy_filter = st.selectbox("Financial Year", fys)
            elif period_type == "Quarter":
                years = sorted(df['Year'].unique(), reverse=True)
                year_filter = st.selectbox("Year", years)
                quarters = sorted(df['Quarter'].unique())
                quarter_filter = st.selectbox("Quarter", quarters)
            
            st.markdown("---")
            
            # Location Filters
            st.markdown("#### 🌍 Location Filters")
            states = ['All'] + sorted([s for s in df['State'].unique() if s != 'NOT ASSIGNED'])
            selected_states = st.multiselect("States", states, default=[])
            
            districts = ['All'] + sorted([d for d in df['District'].unique() if d != 'NOT ASSIGNED'])
            selected_districts = st.multiselect("Districts", districts, default=[])
            
            st.markdown("---")
            
            # Hierarchy Filters
            st.markdown("#### 👥 Hierarchy Filters")
            rbms = ['All'] + sorted([r for r in df['RBM'].unique() if r != 'NOT ASSIGNED'])
            selected_rbms = st.multiselect("RBM", rbms, default=[])
            
            bdms = ['All'] + sorted([b for b in df['BDM'].unique() if b != 'NOT ASSIGNED'])
            selected_bdms = st.multiselect("BDM", bdms, default=[])
            
            branches = sorted(df['Branch'].unique())
            selected_branches = st.multiselect("Branches", branches, default=[])
            
            st.markdown("---")
            
            # Product Filters
            st.markdown("#### 🏷️ Product Filters")
            brands = sorted(df['Brand'].dropna().unique())
            selected_brands = st.multiselect("Brands", brands, default=[])
            
        else:
            # Comparison Mode
            st.markdown("#### ⚖️ Comparison Settings")
            comp_type = st.selectbox("Comparison Type", ["Year vs Year", "FY vs FY", "Quarter vs Quarter"])
            
            if comp_type == "Year vs Year":
                years = sorted(df['Year'].unique(), reverse=True)
                period1 = st.selectbox("Period 1 (Baseline)", years, index=min(1, len(years)-1))
                period2 = st.selectbox("Period 2 (Compare)", years, index=0)
            elif comp_type == "FY vs FY":
                fys = sorted(df['FY_Label'].unique(), reverse=True)
                period1 = st.selectbox("Period 1 (Baseline)", fys, index=min(1, len(fys)-1))
                period2 = st.selectbox("Period 2 (Compare)", fys, index=0)
            else:
                years = sorted(df['Year'].unique(), reverse=True)
                quarters = sorted(df['Quarter'].unique())
                col1, col2 = st.columns(2)
                with col1:
                    p1_year = st.selectbox("Year 1", years, key='p1y')
                    p1_quarter = st.selectbox("Quarter 1", quarters, key='p1q')
                with col2:
                    p2_year = st.selectbox("Year 2", years, key='p2y')
                    p2_quarter = st.selectbox("Quarter 2", quarters, key='p2q')
            
            dimension = st.selectbox("Compare By", ["Overall", "RBM", "BDM", "State", "District", "Branch", "Brand"])
        
        st.markdown("---")
        
        # Reset & Refresh
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", ):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("↩️ Reset", ):
                st.rerun()
        
        # Data Summary
        st.markdown("---")
        st.markdown("#### 📊 Data Summary")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Date Range", f"{df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}")
    
    # ========== APPLY FILTERS ==========
    if "Analytics" in mode:
        filtered_df = df.copy()
        
        # Time filters
        if period_type == "Calendar Year" and year_filter:
            filtered_df = filtered_df[filtered_df['Year'] == year_filter]
        elif period_type == "Financial Year" and fy_filter:
            filtered_df = filtered_df[filtered_df['FY_Label'] == fy_filter]
        elif period_type == "Quarter" and year_filter and quarter_filter:
            filtered_df = filtered_df[(filtered_df['Year'] == year_filter) & 
                                      (filtered_df['Quarter'] == quarter_filter)]
        
        # Location filters
        if selected_states and 'All' not in selected_states:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
        if selected_districts and 'All' not in selected_districts:
            filtered_df = filtered_df[filtered_df['District'].isin(selected_districts)]
        
        # Hierarchy filters
        if selected_rbms and 'All' not in selected_rbms:
            filtered_df = filtered_df[filtered_df['RBM'].isin(selected_rbms)]
        if selected_bdms and 'All' not in selected_bdms:
            filtered_df = filtered_df[filtered_df['BDM'].isin(selected_bdms)]
        if selected_branches:
            filtered_df = filtered_df[filtered_df['Branch'].isin(selected_branches)]
        
        # Product filters
        if selected_brands:
            filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
        
        # ========== ANALYTICS VIEW ==========
        # Status Bar
        active_filters = []
        if selected_states and 'All' not in selected_states:
            active_filters.append(f"{len(selected_states)} States")
        if selected_rbms and 'All' not in selected_rbms:
            active_filters.append(f"{len(selected_rbms)} RBMs")
        if selected_branches:
            active_filters.append(f"{len(selected_branches)} Branches")
        
        filter_text = " • ".join(active_filters) if active_filters else "All Data"
        
        st.markdown(f"""
        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <span class="status-badge">{period_type}</span>
            <span class="status-badge success">{len(filtered_df):,} Records</span>
            <span style="color: #64748b; font-size: 0.9rem; padding: 0.4rem;">Filtered by: {filter_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # KPIs
        kpis = calculate_kpis(filtered_df)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card kpi-revenue">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">{kpis['revenue_formatted']}</div>
                <div class="kpi-label">GROSS REVENUE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card kpi-profit">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">{kpis['profit_formatted']}</div>
                <div class="kpi-label">NET PROFIT</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card kpi-quantity">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{kpis['quantity_formatted']}</div>
                <div class="kpi-label">QUANTITY SOLD</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-card kpi-margin">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{kpis['margin']}%</div>
                <div class="kpi-label">PROFIT MARGIN</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="kpi-card kpi-discount">
                <div class="kpi-icon">🎟️</div>
                <div class="kpi-value">{kpis['discount_pct']}%</div>
                <div class="kpi-label">DISCOUNT %</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="kpi-card kpi-stores">
                <div class="kpi-icon">🏪</div>
                <div class="kpi-value">{kpis['states']} / {kpis['districts']} / {kpis['stores']}</div>
                <div class="kpi-label">STATES / DISTRICTS / STORES</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AI Insights
        st.markdown("### 🧠 AI-Powered Insights")
        insights = generate_insights(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="insight-card positive">
                <div class="insight-title">🏆 Top Performer</div>
                <div class="insight-text">{insights['top_performer']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-card trend">
                <div class="insight-title">📈 Growth Trend</div>
                <div class="insight-text">{insights['growth_trend']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="insight-card highlight">
                <div class="insight-title">⭐ Highlight</div>
                <div class="insight-text">{insights['highlight']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="insight-card alert">
                <div class="insight-title">⚠️ Attention Needed</div>
                <div class="insight-text">{insights['alert']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        st.markdown("### 📊 Visualizations")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 Monthly Trends", 
            "👥 RBM Performance", 
            "🗺️ Geographic View",
            "🏷️ Product Analysis",
            "📍 District Analysis"
        ])
        
        with tab1:
            fig = create_monthly_trend_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, )
            else:
                st.info("No monthly trend data available")
        
        with tab2:
            fig = create_rbm_performance_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, )
            else:
                st.info("No RBM data available")
        
        with tab3:
            fig = create_geographic_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, )
            else:
                st.info("No geographic data available")
        
        with tab4:
            fig = create_product_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, )
            else:
                st.info("No product data available")
        
        with tab5:
            fig = create_district_map_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, )
            else:
                st.info("No district data available")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Table
        st.markdown("### 📋 Data Explorer")
        
        display_cols = ['Date', 'RBM', 'BDM', 'Branch', 'State', 'District', 'Brand', 'Product', 'QTY', 'Sold_Price', 'Profit']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        display_df = filtered_df[available_cols].head(100).copy()
        if 'Date' in display_df.columns:
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, height=400)
        st.caption(f"Showing 100 of {len(filtered_df):,} records")
        
        # Export
        csv = filtered_df[available_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name=f"business_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        # ========== COMPARISON VIEW ==========
        st.markdown("### ⚖️ Period Comparison")
        
        if comp_type == "Year vs Year":
            period1_df = df[df['Year'] == period1]
            period2_df = df[df['Year'] == period2]
            st.markdown(f"<h2 style='text-align: center;'>{period1} vs {period2}</h2>", unsafe_allow_html=True)
        elif comp_type == "FY vs FY":
            period1_df = df[df['FY_Label'] == period1]
            period2_df = df[df['FY_Label'] == period2]
            st.markdown(f"<h2 style='text-align: center;'>{period1} vs {period2}</h2>", unsafe_allow_html=True)
        else:
            period1_df = df[(df['Year'] == p1_year) & (df['Quarter'] == p1_quarter)]
            period2_df = df[(df['Year'] == p2_year) & (df['Quarter'] == p2_quarter)]
            st.markdown(f"<h2 style='text-align: center;'>{p1_quarter} {p1_year} vs {p2_quarter} {p2_year}</h2>", unsafe_allow_html=True)
        
        kpis1 = calculate_kpis(period1_df)
        kpis2 = calculate_kpis(period2_df)
        
        revenue_growth = calculate_growth(kpis2['revenue'], kpis1['revenue'])
        profit_growth = calculate_growth(kpis2['profit'], kpis1['profit'])
        qty_growth = calculate_growth(kpis2['quantity'], kpis1['quantity'])
        margin_change = kpis2['margin'] - kpis1['margin']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cls = "positive" if revenue_growth >= 0 else "negative"
            st.markdown(f"""
            <div class="comparison-card {cls}">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Revenue Growth</div>
                <div class="comparison-value">{format_indian_currency(kpis2['revenue'] - kpis1['revenue'])}</div>
                <div class="comparison-growth {cls}">{'↑' if revenue_growth >= 0 else '↓'} {abs(revenue_growth):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cls = "positive" if profit_growth >= 0 else "negative"
            st.markdown(f"""
            <div class="comparison-card {cls}">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Profit Growth</div>
                <div class="comparison-value">{format_indian_currency(kpis2['profit'] - kpis1['profit'])}</div>
                <div class="comparison-growth {cls}">{'↑' if profit_growth >= 0 else '↓'} {abs(profit_growth):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cls = "positive" if qty_growth >= 0 else "negative"
            st.markdown(f"""
            <div class="comparison-card {cls}">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Quantity Growth</div>
                <div class="comparison-value">{format_indian_number(kpis2['quantity'] - kpis1['quantity'])}</div>
                <div class="comparison-growth {cls}">{'↑' if qty_growth >= 0 else '↓'} {abs(qty_growth):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cls = "positive" if margin_change >= 0 else "negative"
            st.markdown(f"""
            <div class="comparison-card {cls}">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Margin Change</div>
                <div class="comparison-value">{margin_change:+.1f} pts</div>
                <div class="comparison-growth {cls}">{kpis1['margin']}% → {kpis2['margin']}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison Chart
        if dimension != "Overall":
            comparison_data = []
            
            p1_grouped = period1_df.groupby(dimension)['Sold_Price'].sum().reset_index()
            p2_grouped = period2_df.groupby(dimension)['Sold_Price'].sum().reset_index()
            
            merged = pd.merge(p1_grouped, p2_grouped, on=dimension, how='outer', suffixes=('_p1', '_p2'))
            merged = merged.fillna(0)
            merged['Total'] = merged['Sold_Price_p1'] + merged['Sold_Price_p2']
            merged = merged.sort_values('Total', ascending=False).head(20)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=merged[dimension],
                y=merged['Sold_Price_p1'] / 10000000,
                name='Period 1',
                marker_color='#3b82f6'
            ))
            fig.add_trace(go.Bar(
                x=merged[dimension],
                y=merged['Sold_Price_p2'] / 10000000,
                name='Period 2',
                marker_color='#10b981'
            ))
            
            fig.update_layout(
                title=f'Comparison by {dimension}',
                barmode='group',
                height=450,
                yaxis_title='Revenue (₹ Cr)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#475569')
            )
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, )
            
            # Comparison Table
            st.markdown("### 📋 Detailed Comparison")
            
            merged['Growth'] = merged.apply(
                lambda x: calculate_growth(x['Sold_Price_p2'], x['Sold_Price_p1']), axis=1
            )
            merged['Period1_Formatted'] = merged['Sold_Price_p1'].apply(format_indian_currency)
            merged['Period2_Formatted'] = merged['Sold_Price_p2'].apply(format_indian_currency)
            merged['Growth_Formatted'] = merged['Growth'].apply(lambda x: f"{x:+.1f}%")
            
            display_table = merged[[dimension, 'Period1_Formatted', 'Period2_Formatted', 'Growth_Formatted']].copy()
            display_table.columns = [dimension, 'Period 1', 'Period 2', 'Growth %']
            
            st.dataframe(display_table, height=400)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #64748b;">
        <p>📊 <strong>Business Intelligence Dashboard</strong> • Streamlit Edition</p>
        <p style="font-size: 0.85rem;">Multi-Select Filters • Complete Visualization • Advanced Analytics</p>
        <p style="font-size: 0.75rem;">© 2024 Enterprise BI Suite • Version 7.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
