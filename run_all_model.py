import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="Shark Tank India Deal Predictor", page_icon="🦈", layout="wide")
st.title("🦈 Shark Tank India — Multi-Model Smart Predictor")
st.markdown("#### Predict deal success, valuation, and best sharks using all trained models")

# ====== Load all trained models ======
@st.cache_resource
def load_all_models():
    model_dict = {"deal": [], "valuation": [], "sharks": []}
    for file in os.listdir():
        if file.startswith("best_") and file.endswith(".pkl"):
            if "_deal" in file:
                model_dict["deal"].append((file, joblib.load(file)))
            elif "_valuation" in file:
                model_dict["valuation"].append((file, joblib.load(file)))
            elif "_sharks" in file:
                model_dict["sharks"].append((file, joblib.load(file)))
    return model_dict

models = load_all_models()
st.sidebar.success(f"✅ Loaded {len(models['deal'])} deal models, {len(models['valuation'])} valuation models, {len(models['sharks'])} shark models")

# ====== Label encoders ======
@st.cache_data
def load_encoders():
    df = pd.read_excel("sharkTankIndia.xlsx")
    enc = {}
    for col in ["Industry", "Pitchers City", "Pitchers State"]:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        enc[col] = le
    return enc

encoders = load_encoders()

# ====== Input form ======
st.divider()
st.subheader("🧾 Enter Your Pitch Details")

col1, col2, col3 = st.columns(3)
with col1:
    season = st.selectbox("Season", [1, 2, 3])
    num_presenters = st.slider("No. of Presenters", 1, 5, 2)
    male_presenters = st.slider("Male Presenters", 0, num_presenters, 1)
    female_presenters = num_presenters - male_presenters
    couple_presenters = st.selectbox("Couple Presenters?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    avg_age = st.slider("Average Age", 18, 70, 35)

with col2:
    ask = st.number_input("Ask Amount (₹)", min_value=100000, max_value=100000000, value=5000000, step=100000)
    equity = st.slider("Equity Offered (%)", 1.0, 50.0, 10.0, 0.5)
    valuation = ask / (equity / 100)
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
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
        'Ahmedabad', 'Kolkata', 'Pune', 'Jaipur', 'Lucknow',
        'Surat', 'Indore', 'Chandigarh', 'Kochi'
    ])

# Input dataframe
inp = pd.DataFrame([{
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
    'Original Ask Amount': ask,
    'Original Offered Equity': equity,
    'Valuation Requested': valuation,
    'Industry': industry,
    'Num Sharks Present': 8,
    'Valuation_per_Presenter': valuation / max(num_presenters, 1),
    'Equity_per_Shark': equity / 8
}])

# Encode categorical
for col, le in encoders.items():
    if col in inp.columns:
        val = inp[col].iloc[0]
        inp[col] = le.transform([val]) if val in le.classes_ else le.transform([le.classes_[0]])

# ====== Predict button ======
st.divider()
if st.button("🚀 Predict Across All Models"):
    with st.spinner("Running all trained models..."):
        # ---------------- DEAL ----------------
        deal_results = []
        for file, (clf, scaler) in models["deal"]:
            Xs = scaler.transform(inp)
            prob = clf.predict_proba(Xs)[0][1] if hasattr(clf, "predict_proba") else clf.decision_function(Xs)
            deal_results.append((file, prob))
        best_deal = max(deal_results, key=lambda x: x[1])
        deal_prob = best_deal[1] * 100
        deal_pred = int(deal_prob > 55)

        # ---------------- VALUATION ----------------
        val_results = []
        for file, (reg, scaler) in models["valuation"]:
            try:
                pred_val = reg.predict(scaler.transform(inp))[0]
                val_results.append((file, pred_val))
            except Exception:
                pass
        best_val = np.median([v for _, v in val_results]) if val_results else valuation

        # ---------------- SHARKS ----------------
        shark_predictions = {}
        for file, (clf, scaler) in models["sharks"]:
            try:
                y = clf.predict(scaler.transform(inp))[0]
                shark_predictions[file] = y
            except Exception:
                pass
        sharks_list = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']
        final_sharks = []
        if shark_predictions:
            arr = np.mean(list(shark_predictions.values()), axis=0)
            for i, v in enumerate(arr):
                if v >= 0.5:
                    final_sharks.append(sharks_list[i])

        # ---------------- RESULTS ----------------
        st.subheader("🎯 Final AI Predictions (Best of All Models)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Deal Outcome", "✅ Accepted" if deal_pred else "❌ Unlikely", f"{deal_prob:.1f}%")
            st.caption(f"Model: {best_deal[0]}")
        with col2:
            st.metric("Predicted Valuation", f"₹{best_val:,.0f}")
        with col3:
            st.metric("Confidence Level", f"{deal_prob:.1f}%")

        # Confidence Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=deal_prob,
            title={"text": "Deal Confidence (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#38ef7d" if deal_prob > 60 else "#f5576c"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Sharks
        st.markdown("### 🦈 Likely Sharks Interested")
        if final_sharks:
            for s in final_sharks:
                st.success(f"• {s}")
        else:
            st.warning("No shark strongly matches this pitch profile.")

        # ---------------- INSIGHTS ----------------
        st.subheader("📊 AI Suggestions & Insights")
        suggestions = []
        if not deal_pred:
            if equity < 5:
                suggestions.append("Offer slightly **more equity (5–15%)** to attract investors.")
            if valuation > 50000000:
                suggestions.append("Your **valuation seems high** — lower it closer to ₹1–2 Cr range.")
            if industry not in ['Technology', 'Health & Wellness', 'E-commerce']:
                suggestions.append(f"Investors prefer trending sectors. Try linking your business to **{np.random.choice(['Technology', 'Sustainability', 'Digital Services'])}**.")
            if avg_age < 25 or avg_age > 50:
                suggestions.append("Highlight **experience or market understanding** in your pitch.")
            if not suggestions:
                suggestions.append("Revise your pitch clarity, show strong traction or customer metrics.")
            st.error("⚠️ The deal is unlikely based on model insights.")
        else:
            suggestions.append("Your pitch has strong fundamentals — focus on negotiation strategy for better terms.")
            suggestions.append("Highlight scalability and market differentiation for top sharks.")
        for sug in suggestions:
            st.info(f"💡 {sug}")

        st.balloons()
