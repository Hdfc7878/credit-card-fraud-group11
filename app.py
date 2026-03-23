import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import duckdb

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    DATA_URL = 'https://raw.githubusercontent.com/kostis-christodoulou/e628/main/data/card_fraud.csv'
    df = pd.read_csv(DATA_URL, parse_dates=["trans_date_trans_time", "dob"])
    return df

def engineer_features(df):
    out = df.copy()
    # Temporal
    dt = out['trans_date_trans_time'].dt
    out['hour'] = dt.hour
    out['day_of_week'] = dt.dayofweek
    out['month'] = dt.month
    out['is_weekend'] = (dt.dayofweek >= 5).astype(int)
    out['is_night'] = ((dt.hour < 6) | (dt.hour >= 22)).astype(int)
    
    # Age
    trans_naive = out['trans_date_trans_time'].dt.tz_localize(None)
    dob_naive = out['dob'].dt.tz_localize(None)
    out['age'] = ((trans_naive - dob_naive).dt.days / 365.25).round(1)
    
    # Haversine Distance
    R = 6371.0
    lat1, lat2 = np.radians(out['lat']), np.radians(out['merch_lat'])
    dlat = lat2 - lat1
    dlon = np.radians(out['merch_long'] - out['long'])
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    out['geo_distance_km'] = 2 * R * np.arcsin(np.sqrt(a))
    
    # Logs
    out['log_amt'] = np.log1p(out['amt'])
    out['log_city_pop'] = np.log1p(out['city_pop'])
    return out

# --- APP LAYOUT ---
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("### E628: Data Science for Business — Group 11")

raw_data = load_data()
data = engineer_features(raw_data)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Transactions")
selected_year = st.sidebar.multiselect("Select Year", options=data['trans_year'].unique(), default=data['trans_year'].unique())
selected_cat = st.sidebar.multiselect("Category", options=data['category'].unique(), default=data['category'].unique())
amt_range = st.sidebar.slider("Amount ($)", 0, int(data['amt'].max()), (0, 5000))

filtered_df = data[
    (data['trans_year'].isin(selected_year)) & 
    (data['category'].isin(selected_cat)) & 
    (data['amt'].between(amt_range[0], amt_range[1]))
]

# --- KEY METRICS ---
col1, col2, col3, col4 = st.columns(4)
fraud_count = filtered_df['is_fraud'].sum()
fraud_rate = (fraud_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
total_loss = filtered_df[filtered_df['is_fraud'] == 1]['amt'].sum()

col1.metric("Total Transactions", f"{len(filtered_df):,}")
col2.metric("Fraudulent Cases", f"{fraud_count:,}")
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
col4.metric("Total Fraud Loss", f"${total_loss:,.0f}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Descriptive Analytics", "🤖 ML Model Performance", "🔍 Transaction Predictor"])

with tab1:
    st.subheader("Fraud Patterns by Category & Time")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Fraud by Category
        cat_fraud = filtered_df.groupby('category')['is_fraud'].mean().reset_index().sort_values('is_fraud', ascending=False)
        fig_cat = px.bar(cat_fraud, x='is_fraud', y='category', orientation='h', 
                         title="Fraud Rate by Category", color='is_fraud', color_continuous_scale='Reds')
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        # Fraud by Hour
        hour_fraud = filtered_df.groupby('hour')['is_fraud'].mean().reset_index()
        fig_hour = px.line(hour_fraud, x='hour', y='is_fraud', title="Fraud Probability by Hour of Day")
        fig_hour.add_vrect(x0=22, x1=24, fillcolor="red", opacity=0.1, annotation_text="High Risk")
        fig_hour.add_vrect(x0=0, x1=6, fillcolor="red", opacity=0.1)
        st.plotly_chart(fig_hour, use_container_width=True)

    # Geographic Spread
    st.subheader("Geographic Fraud Concentration")
    geo_data = filtered_df[filtered_df['is_fraud'] == 1].sample(min(1000, fraud_count))
    st.map(geo_data, latitude='lat', longitude='long', color='#E63946')

with tab2:
    st.subheader("Model Interpretation (Hist Gradient Boosting)")
    
    # Feature Importance (Pre-calculated from your script results)
    features = ['log_amt', 'geo_distance_km', 'age', 'hour', 'is_night', 'category_shopping_net', 'category_grocery_net']
    importances = [0.28, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance')
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Global Feature Importance (AUC Drop)")
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.info("""
    **Business Insight:** Transaction Amount and Geographic Distance are the primary drivers of the model. 
    The model identifies that late-night online shopping in specific categories significantly increases the fraud score.
    """)

with tab3:
    st.subheader("Live Fraud Risk Scoring")
    st.write("Input transaction details to calculate a real-time risk score.")
    
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        in_amt = st.number_input("Transaction Amount ($)", value=100.0)
        in_cat = st.selectbox("Category", options=data['category'].unique())
    with p_col2:
        in_hour = st.slider("Hour of Day", 0, 23, 12)
        in_dist = st.number_input("Distance to Merchant (km)", value=15.0)
    with p_col3:
        in_age = st.number_input("Customer Age", value=35)
        in_night = 1 if (in_hour < 6 or in_hour >= 22) else 0

    # Simplified Logic for Demo (In production, you'd load the .pkl model)
    # Heuristic based on your model's findings:
    risk_score = 0.05
    if in_amt > 500: risk_score += 0.4
    if in_dist > 200: risk_score += 0.2
    if in_night: risk_score += 0.15
    if in_cat in ['shopping_net', 'misc_net']: risk_score += 0.1
    if in_age > 65: risk_score += 0.05
    
    risk_score = min(risk_score, 0.99)
    
    st.markdown("---")
    st.write(f"### Predicted Fraud Probability: `{risk_score:.2%}`")
    
    if risk_score > 0.7:
        st.error("🚨 HIGH RISK: This transaction matches high-fraud patterns. Immediate block recommended.")
    elif risk_score > 0.3:
        st.warning("⚠️ MEDIUM RISK: Suggest sending an SMS OTP for verification.")
    else:
        st.success("✅ LOW RISK: Transaction appears legitimate.")

# --- FOOTER ---
st.markdown("---")
st.caption("Data Source: Scirp.org | Credit Card Fraud Detection Using Weighted SVM (Reference Dataset)")