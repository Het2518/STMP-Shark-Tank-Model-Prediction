import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# =================== MODEL LOADING ===================

@st.cache_resource
def load_trained_model(model_keyword):
    """Load best model matching the keyword."""
    files = [f for f in os.listdir() if f.startswith("best_") and model_keyword in f and f.endswith(".pkl")]
    if not files:
        st.error(f"No model found for {model_keyword}")
        st.stop()
    # Prefer XGBoost > GradientBoosting > RandomForest > Logistic > Ridge
    priority = ["XGBoost", "GradientBoosting", "RandomForest", "Logistic", "Ridge", "BayesianRidge"]
    files.sort(key=lambda f: next((i for i, p in enumerate(priority) if p in f), 99))
    model_file = files[0]
    model, scaler = joblib.load(model_file)
    st.sidebar.info(f"✅ Loaded model: {model_file}")
    return model, scaler

deal_model, deal_scaler = load_trained_model("deal")
valuation_model, val_scaler = load_trained_model("valuation")
shark_model, shark_scaler = load_trained_model("sharks")

# =================== ENCODER PREPARATION ===================

@st.cache_data
def build_label_encoders():
    df = pd.read_excel("sharkTankIndia.xlsx")
    encoders = {}
    for col in ['Industry', 'Pitchers City', 'Pitchers State']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le
    return encoders

encoders = build_label_encoders()

# =================== STREAMLIT SETUP ===================

st.set_page_config(page_title="Shark Tank India Deal Predictor", page_icon="🦈", layout="wide")

st.title("🦈 Shark Tank India — True Model-Based Deal Predictor")
st.markdown("#### Predictions strictly from your trained ML models (no manual logic)")

st.divider()

# =================== USER INPUT ===================

col1, col2, col3 = st.columns(3)
with col1:
    season = st.selectbox("Season", [1, 2, 3], index=2)
    num_presenters = st.slider("No. of Presenters", 1, 5, 2)
    male_presenters = st.slider("Male Presenters", 0, num_presenters, 1)
    female_presenters = num_presenters - male_presenters
    couple_presenters = st.selectbox("Couple Presenters?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    avg_age = st.slider("Avg Age", 18, 70, 32)

with col2:
    ask_amount = st.number_input("Ask Amount (₹)", min_value=100000, max_value=100000000, value=5000000, step=100000)
    equity = st.slider("Equity Offered (%)", 1.0, 50.0, 10.0, 0.5)
    valuation = ask_amount / (equity / 100)
    st.metric("Calculated Valuation", f"₹{valuation:,.0f}")

with col3:
    industry = st.selectbox("Industry", [
        'Technology', 'Food & Beverage', 'Fashion', 'Health & Wellness',
        'Education', 'Agriculture', 'E-commerce', 'Services', 'Manufacturing',
        'Beauty & Personal Care', 'Sports & Fitness', 'Home & Kitchen'
    ])
    state = st.selectbox("State", [
        'Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Gujarat',
        'Rajasthan', 'Uttar Pradesh', 'West Bengal', 'Telangana', 'Punjab',
        'Haryana', 'Kerala', 'Madhya Pradesh', 'Andhra Pradesh'
    ])
    city = st.selectbox("City", [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Ahmedabad',
        'Kolkata', 'Pune', 'Jaipur', 'Lucknow', 'Surat', 'Indore', 'Chandigarh', 'Kochi'
    ])

# =================== INPUT PREP ===================

input_df = pd.DataFrame([{
    'Season No': season,
    'Episode No': 1,
    'Pitch No': 1,
    'No of Presenters': num_presenters,
    'Male Presenters': male_presenters,
    'Female Presenters': female_presenters,
    'Couple Presenters': couple_presenters,
    'Pitchers Average Age': avg_age,
    'Pitchers City': city,
    'Pitchers State': state,
    'Original Ask Amount': ask_amount,
    'Original Offered Equity': equity,
    'Valuation Requested': valuation,
    'Industry': industry,
    'Num Sharks Present': 8,
    'Valuation_per_Presenter': valuation / max(num_presenters, 1),
    'Equity_per_Shark': equity / 8
}])

# Encode categorical features
for col, le in encoders.items():
    if col in input_df.columns:
        val = input_df[col].iloc[0]
        if val not in le.classes_:
            input_df[col] = le.transform([le.classes_[0]])
        else:
            input_df[col] = le.transform([val])

# =================== PREDICTION ===================

if st.button("🚀 Predict Using True Trained Models"):
    with st.spinner("Running predictions..."):

        # --- Deal Acceptance ---
        X_scaled = deal_scaler.transform(input_df)
        deal_prob = deal_model.predict_proba(X_scaled)[0][1] * 100
        deal_pred = int(deal_prob >= 60)  # threshold = 60% confidence

        # --- Valuation ---
        Xv_scaled = val_scaler.transform(input_df)
        predicted_val = valuation_model.predict(Xv_scaled)[0]

        # --- Sharks ---
        Xs_scaled = shark_scaler.transform(input_df)
        y_pred = shark_model.predict(Xs_scaled)[0]
        shark_labels = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']
        predicted_sharks = [shark_labels[i] for i, v in enumerate(y_pred) if v == 1]

        # =================== OUTPUT ===================
        st.subheader("🎯 Model Predictions")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Deal Outcome", "✅ Accepted" if deal_pred else "❌ Unlikely", f"{deal_prob:.1f}%")
        with col2:
            st.metric("Predicted Valuation", f"₹{predicted_val:,.0f}")
        with col3:
            st.metric("Model Confidence", f"{deal_prob:.1f}%")

        # Confidence Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=deal_prob,
            title={"text": "Model Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#38ef7d" if deal_prob > 60 else "#f5576c"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Sharks
        st.markdown("### 🦈 Sharks Likely to Invest")
        if predicted_sharks:
            for s in predicted_sharks:
                st.success(f"• {s}")
        else:
            st.warning("No shark match predicted for this pitch.")

        # Data insight section
        st.markdown("### 📊 Insights from Model")
        if deal_prob >= 80:
            st.success("Excellent! Model indicates strong acceptance likelihood.")
        elif deal_prob >= 60:
            st.info("Good probability — competitive offer.")
        elif deal_prob >= 40:
            st.warning("Moderate probability — tweak equity or valuation.")
        else:
            st.error("Low deal chance — try reducing valuation or equity percentage.")

        st.balloons()
