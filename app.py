# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🦈 Shark Tank India AI Predictor",
    page_icon="🦈",
    layout="wide"
)

st.title("🦈 Shark Tank India AI Predictor")
st.caption("Predict Deal Outcomes, Valuation, and Shark Investments — powered by Machine Learning")

# ==========================
# LOAD DATASET
# ==========================
DATA_FILE = "sharkTankIndia.xlsx"

if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset file 'sharkTankIndia.xlsx' not found.")
    st.stop()

df = pd.read_excel(DATA_FILE)

# Extract dropdown options dynamically
industries = sorted(df["Industry"].dropna().unique().tolist()) if "Industry" in df else []
cities = sorted(df["Pitchers City"].dropna().unique().tolist()) if "Pitchers City" in df else []
states = sorted(df["Pitchers State"].dropna().unique().tolist()) if "Pitchers State" in df else []

# ==========================
# LOAD TRAINED MODELS
# ==========================
def load_model(suffix):
    for f in os.listdir():
        if f.endswith(suffix):
            return joblib.load(f)
    return None

deal_model = load_model("_deal.pkl")
valuation_model = load_model("_valuation.pkl")
shark_model = load_model("_sharks.pkl")

if not deal_model or not valuation_model or not shark_model:
    st.warning("⚠️ Model files missing. Please train models using run_all_models.py first.")
    st.stop()

deal_clf, deal_scaler = deal_model
reg_model, reg_scaler = valuation_model
shark_clf, shark_scaler = shark_model

# ==========================
# INPUT LAYOUT
# ==========================
st.markdown("### 🎯 Pitch Details")

col1, col2, col3 = st.columns(3)
with col1:
    num_presenters = st.number_input("👥 No. of Presenters", 1, 10, 2)
    male_presenters = st.number_input("♂️ Male Presenters", 0, 10, 1)
    female_presenters = st.number_input("♀️ Female Presenters", 0, 10, 1)
with col2:
    couple_presenters = st.number_input("💑 Couple Presenters", 0, 5, 0)
    pitch_age = st.number_input("🎂 Pitchers Average Age", 18, 60, 30)
    num_sharks_present = st.slider("🦈 No. of Sharks Present", 1, 8, 5)
with col3:
    pitch_city = st.selectbox("🏙️ Pitchers City", options=cities or ["Unknown"])
    pitch_state = st.selectbox("🌍 Pitchers State", options=states or ["Unknown"])
    industry = st.selectbox("🏭 Industry", options=industries or ["Other"])

st.markdown("---")

colA, colB, colC = st.columns(3)
with colA:
    ask_amount = st.number_input("💸 Ask Amount (₹ Lakh)", 1, 10000, 100)
with colB:
    offered_equity = st.number_input("📊 Equity Offered (%)", 1, 100, 10)
with colC:
    valuation = (ask_amount / offered_equity) * 100 if offered_equity else 0
    st.metric("🏢 Implied Company Valuation", f"₹{valuation:,.0f} Lakh")

# Derived features
valuation_per_presenter = valuation / num_presenters if num_presenters else 0
equity_per_shark = offered_equity / num_sharks_present if num_sharks_present else 0

# ==========================
# BUILD INPUT DATAFRAME
# ==========================
features = pd.DataFrame({
    # Dummy placeholders for training compatibility
    'Season No': [1],
    'Episode No': [1],
    'Pitch No': [1],
    # Actual pitch features
    'No of Presenters': [num_presenters],
    'Male Presenters': [male_presenters],
    'Female Presenters': [female_presenters],
    'Couple Presenters': [couple_presenters],
    'Pitchers Average Age': [pitch_age],
    'Pitchers City': [pitch_city],
    'Pitchers State': [pitch_state],
    'Original Ask Amount': [ask_amount],
    'Original Offered Equity': [offered_equity],
    'Valuation Requested': [valuation],
    'Industry': [industry],
    'Num Sharks Present': [num_sharks_present],
    'Valuation_per_Presenter': [valuation_per_presenter],
    'Equity_per_Shark': [equity_per_shark]
})

# Encode non-numeric features
for col in features.select_dtypes(include=['object']).columns:
    features[col] = pd.factorize(features[col])[0]

# Auto-align features with scaler expectations (future-proof)
if hasattr(deal_scaler, "feature_names_in_"):
    expected_cols = list(deal_scaler.feature_names_in_)
    missing_cols = [c for c in expected_cols if c not in features.columns]
    for mc in missing_cols:
        features[mc] = 0
    features = features[expected_cols]

# ==========================
# TABS FOR PREDICTION
# ==========================
tab1, tab2, tab3 = st.tabs(["📊 Deal Prediction", "💰 Valuation Prediction", "🦈 Shark Investment"])

# --- TAB 1: DEAL ACCEPTANCE ---
with tab1:
    st.subheader("📊 Deal Acceptance Prediction")

    st.markdown("""
    💡 **Hint:** This prediction estimates whether your pitch is likely to get a deal offer.
    Factors like number of presenters, ask amount, equity, and industry play a big role.
    """)

    X_scaled = deal_scaler.transform(features)
    y_pred = deal_clf.predict(X_scaled)[0]

    prob = None
    if hasattr(deal_clf, "predict_proba"):
        prob = deal_clf.predict_proba(X_scaled)[0][1]

    if y_pred == 1:
        st.success("🎉 The deal is **likely to be accepted!**")
        st.markdown("🟢 *Your pitch looks strong — investors may show interest!*")
    else:
        st.error("❌ The deal is **unlikely to be accepted.**")
        st.markdown("🔴 *Consider revising your equity or ask to improve appeal.*")

    if prob is not None:
        st.progress(int(prob * 100))
        st.markdown(f"**Confidence:** `{prob*100:.2f}%`")

# --- TAB 2: VALUATION PREDICTION ---
with tab2:
    st.subheader("💰 Predicted Negotiated Valuation")

    st.markdown("""
    💡 **Hint:** This shows the expected final negotiated company valuation based on your inputs.
    Compare it with your **Implied Valuation** above to gauge how much investors might agree on.
    """)

    Xv_scaled = reg_scaler.transform(features)
    valuation_pred = reg_model.predict(Xv_scaled)[0]

    # Convert Lakh → Crore for better readability
    valuation_pred_cr = valuation_pred / 100
    valuation_cr = valuation / 100

    st.metric("Predicted Final Valuation", f"₹{valuation_pred:,.0f} Lakh (≈ ₹{valuation_pred_cr:,.2f} Cr)")

    diff = valuation_pred - valuation

    # 💡 Negotiation Range Logic
    if valuation_pred > 0:
        lower_range = valuation_pred * 1.4
        upper_range = valuation_pred * 1.8
        st.markdown(
            f"💬 *Negotiation Tip:* To land around ₹{valuation_pred:,.0f} Lakh "
            f"(≈ ₹{valuation_pred_cr:,.2f} Cr), consider asking between "
            f"**₹{lower_range:,.0f} – ₹{upper_range:,.0f} Lakh** "
            f"(≈ ₹{lower_range/100:,.2f} – ₹{upper_range/100:,.2f} Cr).*"
        )

    # Display difference feedback
    if diff > 0:
        st.info(f"💹 Model suggests a **higher valuation** (+₹{diff:,.0f} Lakh)")
    elif diff < 0:
        st.warning(f"📉 Model suggests a **lower valuation** (−₹{abs(diff):,.0f} Lakh)")
    else:
        st.success("✅ Your ask perfectly matches the predicted valuation!")

    # Visual comparison bar
    st.markdown("#### Deal Alignment")
    pct_alignment = min(100, max(0, (valuation / valuation_pred) * 100)) if valuation_pred > 0 else 0
    st.progress(int(pct_alignment))
    st.caption(f"Your current ask is about {pct_alignment:.1f}% of the model’s predicted valuation.")

# --- TAB 3: SHARK INVESTMENT ---
with tab3:
    st.subheader("🦈 Predict Which Sharks May Invest")

    st.markdown("""
    💡 **Hint:** This prediction identifies which sharks are most likely to invest based on your pitch details.
    Investors are predicted using multi-label classification across past seasons.
    """)

    Xs_scaled = shark_scaler.transform(features)
    pred = shark_clf.predict(Xs_scaled)[0]

    sharks = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']
    df_pred = pd.DataFrame({
        'Shark': sharks,
        'Prediction': ['✅ Invests' if p == 1 else '❌ Skips' for p in pred]
    })

    invested = df_pred[df_pred['Prediction'] == '✅ Invests']
    if not invested.empty:
        st.success(f"Likely Investors: {', '.join(invested['Shark'].tolist())}")
        st.markdown("🟢 *Your pitch aligns well with these sharks’ past investment patterns.*")
    else:
        st.warning("No sharks are expected to invest in this pitch.")
        st.markdown("🔴 *Try adjusting equity or target industry to attract more investors.*")

    st.dataframe(df_pred, use_container_width=True)


st.markdown("---")
st.caption("Made with ❤️ by STMP Developers • Shark Tank India Predictor • Streamlit + ML")
