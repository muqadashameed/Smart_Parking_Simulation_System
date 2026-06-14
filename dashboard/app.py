import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import random
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data"  / "cleaned_parking_data.csv"
RF_PATH     = BASE_DIR / "models" / "outputs" / "random_forest_model.pkl"
XGB_PATH    = BASE_DIR / "models" / "outputs" / "xgboost_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "outputs" / "scaler.pkl"
ASSETS_DIR  = BASE_DIR / "assets"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar nav label */
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #163353 100%);
    border: 1px solid #2a5298;
    border-radius: 12px;
    padding: 12px 18px;
    color: white;
}
div[data-testid="metric-container"] label { color: #90cdf4 !important; font-size: 0.82rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; font-weight: 700; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #68d391 !important; }

/* Page title */
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #4299e1, #63b3ed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.page-sub { color: #718096; font-size: 0.92rem; margin-bottom: 1.5rem; }

/* Section headers */
.section-header {
    font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #4299e1; padding-left: 10px;
    margin: 1.5rem 0 0.8rem 0;
}

/* Slot grid */
.slot-grid { font-family: monospace; line-height: 1.6; }
.slot-free { color: #68d391; }
.slot-busy { color: #fc8181; }

/* Info box */
.info-box {
    background: #1a202c; border: 1px solid #2d3748;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.6rem 0;
}

/* Badge */
.badge-green { background:#276749; color:#c6f6d5; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.badge-red   { background:#742a2a; color:#fed7d7; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.badge-blue  { background:#1a365d; color:#bee3f8; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.badge-orange{ background:#7b341e; color:#feebc8; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD SHARED RESOURCES (cached)
# ─────────────────────────────────────────────────────────────
def encode_df(df):
    """Always encode categorical columns to numeric — safe to call multiple times."""
    df = df.copy()
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    section_map = {"Zone A": 0, "Zone B": 1, "Zone C": 2, "Zone D": 3}
    size_map    = {"Compact": 0, "Standard": 1, "Oversized": 2}
    weekend_map = {True: 1, False: 0, "True": 1, "False": 0}

    df["Nearby_Traffic_Level"] = df["Nearby_Traffic_Level"].map(traffic_map).fillna(df["Nearby_Traffic_Level"]).astype(float).astype(int)
    df["Parking_Lot_Section"]  = df["Parking_Lot_Section"].map(section_map).fillna(df["Parking_Lot_Section"]).astype(float).astype(int)
    df["Spot_Size"]            = df["Spot_Size"].map(size_map).fillna(df["Spot_Size"]).astype(float).astype(int)
    df["Is_Weekend"]           = df["Is_Weekend"].map(weekend_map).fillna(df["Is_Weekend"]).astype(float).astype(int)
    return df

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = encode_df(df)
    return df

@st.cache_resource
def load_models():
    rf  = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH)
    sc  = joblib.load(SCALER_PATH)
    return rf, xgb, sc

df          = load_data()
rf, xgb, sc = load_models()

FEATURES = [
    "Occupancy_Rate","Sensor_Reading_Proximity","Sensor_Reading_Pressure",
    "Sensor_Reading_Ultrasonic","Reserved_Status","Hour","Is_Weekend",
    "Nearby_Traffic_Level","Parking_Lot_Section","Spot_Size",
    "Dynamic_Pricing_Factor","Weather_Temperature","Weather_Precipitation",
]

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🅿️ Smart Parking")
    st.markdown("**AI-Powered Dashboard**")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔴 Live Simulation", "📊 EDA & Insights", "🔮 Predict Occupancy", "🤖 Model Comparison"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("📁 Dataset: IIoT Smart Parking")
    st.caption(f"📈 Records: {len(df):,}")
    st.caption("🤖 Models: Random Forest · XGBoost")
    st.caption("🎓 Phase 4 — Streamlit Dashboard")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="page-title">🅿️ Smart Parking System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AI-powered parking intelligence · Phase 4 Capstone</div>', unsafe_allow_html=True)

    occ = (df["Occupancy_Status"] == "Occupied").mean() * 100
    total = len(df)
    weekend_pct = df["Is_Weekend"].mean() * 100
    avg_price = df["Payment_Amount"].mean()
    avg_dur = df["Parking_Duration"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 Total Records",   f"{total:,}")
    c2.metric("🔴 Avg Occupancy",   f"{occ:.1f}%", delta="+ML Predicted")
    c3.metric("💰 Avg Payment",     f"${avg_price:.2f}")
    c4.metric("⏱ Avg Duration",    f"{avg_dur:.1f} hrs")
    c5.metric("📅 Weekend Share",   f"{weekend_pct:.1f}%")

    st.markdown('<div class="section-header">📅 Occupancy Across Days & Hours</div>', unsafe_allow_html=True)

    # Load raw csv for this view
    raw = pd.read_csv(DATA_PATH)
    raw["Occupancy_Label"] = raw["Occupancy_Label"].map({1:"Occupied",0:"Vacant"}) if raw["Occupancy_Label"].dtype != object else raw["Occupancy_Label"]

    col1, col2 = st.columns(2)
    with col1:
        hour_occ = raw.groupby("Hour")["Occupancy_Label"].apply(lambda x: (x=="Occupied").mean()*100).reset_index()
        hour_occ.columns = ["Hour", "Occupancy %"]
        fig = px.area(hour_occ, x="Hour", y="Occupancy %",
                      title="Occupancy Rate by Hour of Day",
                      color_discrete_sequence=["#4299e1"])
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        day_occ = raw.groupby("Day")["Occupancy_Label"].apply(lambda x: (x=="Occupied").mean()*100).reset_index()
        day_occ.columns = ["Day","Occupancy %"]
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_occ["Day"] = pd.Categorical(day_occ["Day"], categories=day_order, ordered=True)
        day_occ = day_occ.sort_values("Day")
        fig2 = px.bar(day_occ, x="Day", y="Occupancy %",
                      title="Occupancy Rate by Day of Week",
                      color="Occupancy %",
                      color_continuous_scale="Blues")
        fig2.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🗺️ Zone & Vehicle Breakdown</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        zone_counts = raw["Parking_Lot_Section"].value_counts().reset_index()
        zone_counts.columns = ["Zone","Count"]
        fig3 = px.pie(zone_counts, names="Zone", values="Count", title="Slots by Parking Zone",
                      color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
        fig3.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        veh_counts = raw["Vehicle_Type"].value_counts().reset_index()
        veh_counts.columns = ["Vehicle","Count"]
        fig4 = px.bar(veh_counts, x="Vehicle", y="Count", title="Vehicle Type Distribution",
                      color="Count", color_continuous_scale="Teal")
        fig4.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">🚀 Project Phases</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("""<div class="info-box">
        <b>✅ Phase 1</b><br>EDA & Feature Engineering<br><small>Kaggle Notebook · 20 marks</small>
    </div>""", unsafe_allow_html=True)
    p2.markdown("""<div class="info-box">
        <b>✅ Phase 2</b><br>ML Models (RF + XGBoost)<br><small>Classification · 10 marks</small>
    </div>""", unsafe_allow_html=True)
    p3.markdown("""<div class="info-box">
        <b>✅ Phase 3</b><br>3D VPython Simulation<br><small>Parking Grid Demo · 10 marks</small>
    </div>""", unsafe_allow_html=True)
    p4.markdown("""<div class="info-box">
        <b>✅ Phase 4</b><br>Streamlit Dashboard<br><small>4-page App · Bonus</small>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2 — LIVE SIMULATION
# ─────────────────────────────────────────────────────────────
elif page == "🔴 Live Simulation":
    st.markdown('<div class="page-title">🔴 Live 3D Parking Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Interactive browser-based 3D parking system with entry/exit lanes, moving cars, barriers, red/green sensors, and 360 camera controls.</div>',
        unsafe_allow_html=True
    )

    st.info("Use mouse drag to rotate, scroll to zoom, and right-click drag to pan inside the simulation.")

    components.iframe(
        "https://smart-parking-simulation-system-l2b.vercel.app",
        height=780,
        scrolling=True
    )

    st.markdown(
        "[Open simulation in full screen](https://smart-parking-simulation-system-l2b.vercel.app)"
    )

elif page == "📊 EDA & Insights":
    st.markdown('<div class="page-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Dataset deep-dive from Phase 1 — IIoT Smart Parking Management</div>', unsafe_allow_html=True)

    raw = pd.read_csv(DATA_PATH)

    # Dataset summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{len(raw):,}")
    c2.metric("Columns", f"{len(raw.columns)}")
    c3.metric("Null %",  f"{raw.isnull().mean().mean()*100:.2f}%")
    c4.metric("Zones",   raw["Parking_Lot_Section"].nunique())

    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Time Patterns", "🚗 Vehicle & Zone", "💰 Payments", "🔗 Correlations"])

    with tab1:
        st.markdown('<div class="section-header">Occupancy by Hour & Day</div>', unsafe_allow_html=True)
        pivot = raw.pivot_table(values="Occupancy_Label", index="Day", columns="Hour", aggfunc="mean")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        fig = px.imshow(pivot * 100, title="Occupancy Rate Heatmap (Hour × Day)",
                        labels=dict(x="Hour of Day", y="Day", color="Occ %"),
                        color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            hour_df = raw.groupby("Hour")["Occupancy_Label"].mean().reset_index()
            hour_df.columns = ["Hour","Occupancy Rate"]
            fig2 = px.bar(hour_df, x="Hour", y="Occupancy Rate",
                          title="Peak Hours — Average Occupancy",
                          color="Occupancy Rate", color_continuous_scale="Blues")
            fig2.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            wk_df = raw.groupby("Is_Weekend")["Occupancy_Label"].mean().reset_index()
            wk_df["Is_Weekend"] = wk_df["Is_Weekend"].map({0:"Weekday",1:"Weekend"})
            wk_df.columns = ["Type","Occupancy Rate"]
            fig3 = px.bar(wk_df, x="Type", y="Occupancy Rate",
                          title="Weekday vs Weekend Occupancy",
                          color="Type", color_discrete_map={"Weekday":"#4299e1","Weekend":"#ed8936"})
            fig3.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            zone_occ = raw.groupby("Parking_Lot_Section")["Occupancy_Label"].mean().reset_index()
            zone_occ.columns = ["Zone","Occupancy Rate"]
            fig = px.bar(zone_occ, x="Zone", y="Occupancy Rate", title="Occupancy Rate by Zone",
                         color="Occupancy Rate", color_continuous_scale="Teal")
            fig.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            veh_occ = raw.groupby("Vehicle_Type")["Occupancy_Label"].mean().reset_index()
            veh_occ.columns = ["Vehicle","Occupancy Rate"]
            fig2 = px.bar(veh_occ, x="Vehicle", y="Occupancy Rate", title="Occupancy by Vehicle Type",
                          color="Vehicle", color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            size_df = raw["Spot_Size"].value_counts().reset_index()
            size_df.columns = ["Spot Size","Count"]
            fig3 = px.pie(size_df, names="Spot Size", values="Count", title="Spot Size Distribution",
                          hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig3.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            dur_fig = px.histogram(raw, x="Parking_Duration", nbins=30,
                                   title="Parking Duration Distribution",
                                   color_discrete_sequence=["#4299e1"])
            dur_fig.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
            st.plotly_chart(dur_fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            pay_df = raw["Payment_Status"].value_counts().reset_index()
            pay_df.columns = ["Status","Count"]
            fig = px.pie(pay_df, names="Status", values="Count", title="Payment Status Split",
                         hole=0.4, color_discrete_map={"Paid":"#68d391","Unpaid":"#fc8181"})
            fig.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            price_fig = px.histogram(raw, x="Payment_Amount", nbins=40,
                                     title="Payment Amount Distribution",
                                     color_discrete_sequence=["#ed8936"])
            price_fig.update_layout(template="plotly_dark", height=320, margin=dict(t=40,b=20))
            st.plotly_chart(price_fig, use_container_width=True)

        avg_pay = raw.groupby("Parking_Lot_Section")["Payment_Amount"].mean().reset_index()
        avg_pay.columns = ["Zone","Avg Payment"]
        fig_zone = px.bar(avg_pay, x="Zone", y="Avg Payment", title="Average Payment by Zone",
                          color="Avg Payment", color_continuous_scale="Oranges")
        fig_zone.update_layout(template="plotly_dark", height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig_zone, use_container_width=True)

    with tab4:
        num_cols = ["Occupancy_Rate","Sensor_Reading_Proximity","Sensor_Reading_Pressure",
                    "Sensor_Reading_Ultrasonic","Weather_Temperature","Weather_Precipitation",
                    "Dynamic_Pricing_Factor","Parking_Duration","Payment_Amount","Hour","Is_Weekend"]
        corr = raw[num_cols].corr()
        fig = px.imshow(corr, title="Feature Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        text_auto=".2f")
        fig.update_layout(template="plotly_dark", height=520, margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Feature statistics table
        st.markdown('<div class="section-header">📋 Feature Statistics</div>', unsafe_allow_html=True)
        st.dataframe(raw[num_cols].describe().round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4 — PREDICTION TOOL
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Predict Occupancy":
    st.markdown('<div class="page-title">🔮 Occupancy Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Input parking conditions and get an instant ML prediction</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1.2, 1])

    with col_in:
        st.markdown('<div class="section-header">🔧 Input Parameters</div>', unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        hour        = a1.slider("🕐 Hour of Day",          0, 23, 9)
        occ_rate    = a2.slider("📊 Current Occupancy Rate", 0.0, 1.0, 0.5, 0.01)

        b1, b2 = st.columns(2)
        is_weekend  = b1.selectbox("📅 Day Type",   ["Weekday","Weekend"])
        traffic     = b2.selectbox("🚦 Traffic Level", ["Low","Medium","High"])

        c1, c2 = st.columns(2)
        section     = c1.selectbox("🗺️ Parking Zone",  ["Zone A","Zone B","Zone C","Zone D"])
        spot_size   = c2.selectbox("🅿️ Spot Size",      ["Compact","Standard","Oversized"])

        d1, d2 = st.columns(2)
        reserved    = d1.selectbox("🔒 Reserved",    ["No","Yes"])
        weather_tmp = d2.slider("🌡 Temperature (°C)", -10, 45, 22)

        e1, e2 = st.columns(2)
        precipitation = e1.slider("🌧 Precipitation",   0, 5, 0)
        pricing       = e2.slider("💲 Dynamic Pricing", 0.5, 2.0, 1.0, 0.1)

        f1, f2, f3 = st.columns(3)
        proximity   = f1.slider("📡 Proximity", 0.0, 15.0, 5.0)
        pressure    = f2.slider("⚖ Pressure",  0.0, 10.0, 2.0)
        ultrasonic  = f3.slider("🔊 Ultrasonic", 50.0, 150.0, 100.0)

        model_choice = st.radio("🤖 Select Model", ["Random Forest", "XGBoost", "Both"], horizontal=True)
        predict_btn  = st.button("🔮 Predict Now", type="primary", use_container_width=True)

    with col_out:
        st.markdown('<div class="section-header">📤 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Encode inputs
            traffic_enc = {"Low":0,"Medium":1,"High":2}[traffic]
            section_enc = {"Zone A":0,"Zone B":1,"Zone C":2,"Zone D":3}[section]
            size_enc    = {"Compact":0,"Standard":1,"Oversized":2}[spot_size]
            is_wknd     = 1 if is_weekend == "Weekend" else 0
            res_enc     = 1 if reserved == "Yes" else 0

            row = pd.DataFrame([{
                "Occupancy_Rate":               occ_rate,
                "Sensor_Reading_Proximity":     proximity,
                "Sensor_Reading_Pressure":      pressure,
                "Sensor_Reading_Ultrasonic":    ultrasonic,
                "Reserved_Status":              res_enc,
                "Hour":                         hour,
                "Is_Weekend":                   is_wknd,
                "Nearby_Traffic_Level":         traffic_enc,
                "Parking_Lot_Section":          section_enc,
                "Spot_Size":                    size_enc,
                "Dynamic_Pricing_Factor":       pricing,
                "Weather_Temperature":          weather_tmp,
                "Weather_Precipitation":        precipitation,
            }])
            row_scaled = sc.transform(row)

            rf_pred   = rf.predict(row_scaled)[0]
            rf_prob   = rf.predict_proba(row_scaled)[0]
            xgb_pred  = xgb.predict(row_scaled)[0]
            xgb_prob  = xgb.predict_proba(row_scaled)[0]

            label_map  = {1:"🔴 Occupied", 0:"🟢 Vacant"}
            color_map  = {1:"#742a2a", 0:"#1a3a2a"}
            border_map = {1:"#fc8181", 0:"#68d391"}

            def show_result(name, pred, prob):
                p_occ   = prob[1] * 100
                p_vacant= prob[0] * 100
                bg      = color_map[pred]
                br      = border_map[pred]
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {br};border-radius:12px;
                            padding:1rem 1.2rem;margin-bottom:1rem;">
                  <div style="font-size:0.8rem;color:#a0aec0;margin-bottom:4px">{name}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:{br}">{label_map[pred]}</div>
                  <div style="margin-top:8px;font-size:0.85rem;color:#e2e8f0">
                    Occupied confidence: <b>{p_occ:.1f}%</b><br>
                    Vacant confidence: <b>{p_vacant:.1f}%</b>
                  </div>
                </div>""", unsafe_allow_html=True)

            if model_choice in ["Random Forest", "Both"]:
                show_result("🌳 Random Forest Classifier", rf_pred, rf_prob)
            if model_choice in ["XGBoost", "Both"]:
                show_result("⚡ XGBoost Classifier", xgb_pred, xgb_prob)

            # Probability gauge
            prob_to_show = rf_prob[1] if model_choice != "XGBoost" else xgb_prob[1]
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_to_show * 100,
                title={"text":"Occupied Probability (%)"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar": {"color":"#4299e1"},
                    "steps":[
                        {"range":[0,40],"color":"#1a3a2a"},
                        {"range":[40,70],"color":"#7b341e"},
                        {"range":[70,100],"color":"#742a2a"},
                    ],
                    "threshold":{"line":{"color":"white","width":2},"value":70},
                }
            ))
            gauge.update_layout(template="plotly_dark", height=280, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(gauge, use_container_width=True)

        else:
            st.markdown("""<div class="info-box" style="text-align:center;padding:2.5rem 1rem;">
                <div style="font-size:3rem">🤖</div>
                <div style="color:#90cdf4;font-size:1rem;margin-top:0.5rem">
                    Adjust the parameters on the left and click <b>Predict Now</b>
                </div>
            </div>""", unsafe_allow_html=True)

        # Bulk prediction over all 24 hours
        st.markdown('<div class="section-header">📈 Hourly Forecast</div>', unsafe_allow_html=True)

        traffic_enc = {"Low":0,"Medium":1,"High":2}[traffic]
        section_enc = {"Zone A":0,"Zone B":1,"Zone C":2,"Zone D":3}[section]
        size_enc    = {"Compact":0,"Standard":1,"Oversized":2}[spot_size]
        is_wknd     = 1 if is_weekend == "Weekend" else 0
        res_enc     = 1 if reserved == "Yes" else 0

        hours = list(range(24))
        probs = []
        for h in hours:
            r = pd.DataFrame([{
                "Occupancy_Rate": occ_rate, "Sensor_Reading_Proximity": proximity,
                "Sensor_Reading_Pressure": pressure, "Sensor_Reading_Ultrasonic": ultrasonic,
                "Reserved_Status": res_enc, "Hour": h, "Is_Weekend": is_wknd,
                "Nearby_Traffic_Level": traffic_enc, "Parking_Lot_Section": section_enc,
                "Spot_Size": size_enc, "Dynamic_Pricing_Factor": pricing,
                "Weather_Temperature": weather_tmp, "Weather_Precipitation": precipitation,
            }])
            probs.append(rf.predict_proba(sc.transform(r))[0][1] * 100)

        fig_hr = px.area(x=hours, y=probs, title="Predicted Occupied Probability Across 24 Hours",
                         labels={"x":"Hour","y":"Occupied Prob (%)"},
                         color_discrete_sequence=["#4299e1"])
        fig_hr.add_hline(y=50, line_dash="dash", line_color="#fc8181", annotation_text="50% threshold")
        fig_hr.update_layout(template="plotly_dark", height=280, margin=dict(t=40,b=20))
        st.plotly_chart(fig_hr, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 5 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown('<div class="page-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Evaluate Random Forest vs XGBoost on the full test set</div>', unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, roc_curve, auc
    )

    X = encode_df(df.copy())[FEATURES].astype(float)
    y = df["Occupancy_Label"]
    X_scaled = sc.transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_pred   = rf.predict(X_test)
    xgb_pred  = xgb.predict(X_test)
    rf_prob   = rf.predict_proba(X_test)[:,1]
    xgb_prob  = xgb.predict_proba(X_test)[:,1]

    def metrics(preds, probs, label):
        return {
            "Model":     label,
            "Accuracy":  round(accuracy_score(y_test, preds)*100, 2),
            "F1 Score":  round(f1_score(y_test, preds)*100, 2),
            "Precision": round(precision_score(y_test, preds)*100, 2),
            "Recall":    round(recall_score(y_test, preds)*100, 2),
        }

    rf_m  = metrics(rf_pred, rf_prob, "Random Forest")
    xgb_m = metrics(xgb_pred, xgb_prob, "XGBoost")

    # Top KPI row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🌳 RF Accuracy",   f"{rf_m['Accuracy']}%")
    m2.metric("🌳 RF F1",         f"{rf_m['F1 Score']}%")
    m3.metric("🌳 RF Precision",  f"{rf_m['Precision']}%")
    m4.metric("⚡ XGB Accuracy",  f"{xgb_m['Accuracy']}%")
    m5.metric("⚡ XGB F1",        f"{xgb_m['F1 Score']}%")
    m6.metric("⚡ XGB Precision", f"{xgb_m['Precision']}%")

    # Comparison bar chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📊 Side-by-Side Metrics</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame([rf_m, xgb_m])
        fig = go.Figure()
        metrics_list = ["Accuracy","F1 Score","Precision","Recall"]
        colors_rf  = "#4299e1"
        colors_xgb = "#ed8936"
        fig.add_trace(go.Bar(name="Random Forest", x=metrics_list,
                             y=[rf_m[m] for m in metrics_list], marker_color=colors_rf))
        fig.add_trace(go.Bar(name="XGBoost", x=metrics_list,
                             y=[xgb_m[m] for m in metrics_list], marker_color=colors_xgb))
        fig.update_layout(barmode="group", template="plotly_dark", height=360,
                          yaxis=dict(range=[0,105]), margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📈 ROC Curves</div>', unsafe_allow_html=True)
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
        rf_auc  = auc(rf_fpr, rf_tpr)
        xgb_auc = auc(xgb_fpr, xgb_tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=rf_fpr,  y=rf_tpr,  name=f"RF (AUC={rf_auc:.3f})",  line_color=colors_rf))
        fig_roc.add_trace(go.Scatter(x=xgb_fpr, y=xgb_tpr, name=f"XGB (AUC={xgb_auc:.3f})", line_color=colors_xgb))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        fig_roc.update_layout(template="plotly_dark", height=360, margin=dict(t=20,b=20),
                               xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
    cm1_col, cm2_col = st.columns(2)

    def cm_fig(preds, title, cmap):
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Vacant","Occupied"], y=["Vacant","Occupied"],
                        title=title, color_continuous_scale=cmap)
        fig.update_layout(template="plotly_dark", height=340, margin=dict(t=50,b=20))
        return fig

    with cm1_col:
        st.plotly_chart(cm_fig(rf_pred, "🌳 Random Forest Confusion Matrix", "Blues"), use_container_width=True)
    with cm2_col:
        st.plotly_chart(cm_fig(xgb_pred, "⚡ XGBoost Confusion Matrix", "Oranges"), use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">🔍 Feature Importances — Random Forest</div>', unsafe_allow_html=True)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig_imp = px.bar(x=imp.values, y=imp.index, orientation="h",
                     title="RF Feature Importances",
                     color=imp.values, color_continuous_scale="Blues")
    fig_imp.update_layout(template="plotly_dark", height=420, margin=dict(t=40,b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">📋 Summary Table</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([
        {"Model":"🌳 Random Forest", "Accuracy":f"{rf_m['Accuracy']}%", "F1":f"{rf_m['F1 Score']}%",
         "Precision":f"{rf_m['Precision']}%", "Recall":f"{rf_m['Recall']}%",
         "AUC":f"{rf_auc:.4f}", "Notes":"Robust, good generalization"},
        {"Model":"⚡ XGBoost",       "Accuracy":f"{xgb_m['Accuracy']}%", "F1":f"{xgb_m['F1 Score']}%",
         "Precision":f"{xgb_m['Precision']}%", "Recall":f"{xgb_m['Recall']}%",
         "AUC":f"{xgb_auc:.4f}", "Notes":"Boosting, handles imbalance well"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)