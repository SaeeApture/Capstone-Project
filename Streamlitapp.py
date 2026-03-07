"""
⚡ Battery Health Intelligence Dashboard
Competition-Level Streamlit Web App
Based on NASA Battery Discharge Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Battery Health AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .main .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.8rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #58a6ff; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1f2937, #111827);
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #0d2137, #0a1628);
        border: 1px solid #1f6feb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-box h3 { color: #58a6ff; margin-bottom: 0.5rem; }

    /* Health badge */
    .badge-excellent { background:#0f4c2a; color:#3fb950; border:1px solid #238636; border-radius:6px; padding:4px 12px; font-weight:700; }
    .badge-good      { background:#2d2f0f; color:#d29922; border:1px solid #9e6a03; border-radius:6px; padding:4px 12px; font-weight:700; }
    .badge-fair      { background:#2d1c0f; color:#f0883e; border:1px solid #b46e00; border-radius:6px; padding:4px 12px; font-weight:700; }
    .badge-poor      { background:#2d0f0f; color:#f85149; border:1px solid #b62324; border-radius:6px; padding:4px 12px; font-weight:700; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: #161b22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; padding: 0.4rem 1rem; }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #58a6ff !important; }

    /* Slider */
    .stSlider > div { color: #58a6ff; }
    div[data-testid="stNumberInput"] input { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
INITIAL_CAPACITY = 2.0

def calculate_soh(capacity):
    return (capacity / INITIAL_CAPACITY) * 100

def estimate_rul(soh):
    return (soh / 100) * 5

def recommend_use(soh):
    if soh >= 80:   return "⚡ Electric Vehicle Reuse"
    elif soh >= 70: return "☀️ Solar Energy Storage"
    elif soh >= 60: return "🔋 Backup Power Systems"
    else:           return "♻️ Recycle Battery"

def pack_group(soh):
    if soh >= 80:   return "🟢 High Performance Pack"
    elif soh >= 70: return "🟡 Solar Storage Pack"
    elif soh >= 60: return "🟠 Backup Pack"
    else:           return "🔴 Do Not Group"

def health_badge(soh):
    if soh >= 80:   return "excellent", "Excellent"
    elif soh >= 70: return "good",      "Good"
    elif soh >= 60: return "fair",      "Fair"
    else:           return "poor",      "Poor"

def dark_plotly(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="monospace"),
        title=dict(text=title, font=dict(size=16, color="#58a6ff")),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig

# ─────────────────────────────────────────
#  DATA GENERATION (SYNTHETIC – replace with CSV load)
# ─────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n=5000):
    """Fallback synthetic data if CSV not found."""
    np.random.seed(42)
    cycles      = np.random.randint(1, 168, n)
    voltage     = np.random.uniform(1.74, 4.04, n)
    current     = np.random.uniform(-2.03, -1.97, n)
    temperature = 22 + cycles * 0.08 + np.random.normal(0, 1.5, n)
    base_cap    = 2.035 - cycles * 0.005 + np.random.normal(0, 0.04, n)
    capacity    = np.clip(base_cap, 1.15, 2.04)
    df = pd.DataFrame({
        "id_cycle":             cycles,
        "Voltage_measured":     voltage,
        "Current_measured":     current,
        "Temperature_measured": temperature,
        "Capacity":             capacity,
        "ambient_temperature":  24,
    })
    return df

@st.cache_data
def load_data(uploaded_file=None):
    """
    Priority:
    1. File uploaded via Streamlit uploader
    2. discharge.csv sitting next to this script
    3. Synthetic fallback
    """
    import os

    # 1. Uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, "✅ Loaded from uploaded file"

    # 2. CSV next to the notebook / script
    # Notebook was saved at: C:\Users\student\Downloads\Untitled10.ipynb
    # So discharge.csv is expected at: C:\Users\student\Downloads\discharge.csv
    candidate_paths = [
        r"C:\Users\student\Downloads\discharge.csv",   # same folder as notebook
        "discharge.csv",                                # current working directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "discharge.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, f"✅ Loaded from `{path}`"

    # 3. Synthetic fallback
    df = generate_synthetic_data()
    return df, "⚠️ discharge.csv not found — using synthetic data (place discharge.csv next to this script)"

@st.cache_resource
def train_models(df):
    features = ["id_cycle", "Voltage_measured", "Current_measured", "Temperature_measured"]
    X = df[features]
    y = df["Capacity"]

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":               SVR(kernel="rbf", C=10),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "MAE":   round(mean_absolute_error(y_test, pred), 4),
            "R2":    round(r2_score(y_test, pred), 4),
            "pred":  pred,
            "true":  y_test.values,
        }

    # K-Means clustering on per-cycle aggregated data
    cycle_data = df.groupby("id_cycle").agg(
        Capacity=("Capacity", "mean"),
        Temperature=("Temperature_measured", "mean"),
        Voltage=("Voltage_measured", "mean"),
    ).reset_index()
    cycle_data["SOH"] = calculate_soh(cycle_data["Capacity"])

    km_features = cycle_data[["Capacity", "Temperature", "Voltage"]].values
    km_scaled   = StandardScaler().fit_transform(km_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cycle_data["Cluster"] = kmeans.fit_predict(km_scaled)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(km_scaled)
    cycle_data["PCA1"] = pca_coords[:, 0]
    cycle_data["PCA2"] = pca_coords[:, 1]

    return scaler, results, cycle_data, X_test, y_test

# ─────────────────────────────────────────
#  LOAD DATA & MODELS
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
# ── SIDEBAR (defined before load so uploader feeds load_data) ─────────────────
with st.sidebar:
    st.markdown("## ⚡ Battery AI")
    st.markdown("---")

    # Data source
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload discharge.csv (optional)",
        type=["csv"],
        help=(
            "If not uploaded, the app looks for:\n"
            r"C:\Users\student\Downloads\discharge.csv"
        ),
    )
    st.markdown("---")

    # Input sliders
    st.markdown("### 🔬 Input Parameters")
    cycle     = st.slider("Charge Cycles",    min_value=0,    max_value=3000, value=500,  step=10)
    voltage   = st.slider("Voltage (V)",      min_value=2.5,  max_value=4.2,  value=3.7,  step=0.01, format="%.2f")
    current   = st.slider("Current (A)",      min_value=-3.0, max_value=0.0,  value=-2.0, step=0.01, format="%.2f")
    temp      = st.slider("Temperature (°C)", min_value=0.0,  max_value=60.0, value=30.0, step=0.5,  format="%.1f")
    st.markdown("---")
    st.caption("📡 NASA Battery Dataset · sklearn · Plotly · Seaborn")

# ── LOAD DATA & MODELS ────────────────────────────────────────────────────────
df, data_status = load_data(uploaded_file)
scaler, results, cycle_data, X_test, y_test = train_models(df)
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model      = results[best_model_name]["model"]

# Model selector rendered after models are trained
with st.sidebar:
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose ML Model",
        list(results.keys()),
        index=list(results.keys()).index(best_model_name),
    )

# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────
user_arr    = np.array([[cycle, voltage, current, temp]])
user_scaled = scaler.transform(user_arr)
chosen_model = results[model_choice]["model"]
cap_pred    = chosen_model.predict(user_scaled)[0]
soh         = calculate_soh(cap_pred)
rul         = estimate_rul(soh)
reuse       = recommend_use(soh)
group       = pack_group(soh)
badge_cls, badge_lbl = health_badge(soh)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
    <h1 style='color:#58a6ff; font-size:2.2rem; margin-bottom:0;'>⚡ Battery Health Intelligence</h1>
    <p style='color:#8b949e; font-size:1rem;'>AI-Powered Degradation Analysis · NASA Discharge Data</p>
</div>
""", unsafe_allow_html=True)

# Data status banner
if "✅" in data_status:
    st.success(data_status)
else:
    st.warning(data_status)

# ─────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔋 Predicted Capacity", f"{cap_pred:.3f} Ah")
k2.metric("💚 State of Health",    f"{soh:.1f}%")
k3.metric("⏳ Remaining Life",      f"{rul:.2f} yrs")
k4.metric("🏆 Best Model R²",       f"{results[best_model_name]['R2']:.4f}")
k5.metric("📊 Dataset Rows",        f"{len(df):,}")

st.markdown("---")

# ─────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔋 Prediction",
    "📉 Degradation Curve",
    "🤖 Health Prediction",
    "🗺️ Cluster Analysis",
    "🌡️ Temperature Monitor",
])

# ═══════════════════════════════════════
#  TAB 1 · PREDICTION RESULT
# ═══════════════════════════════════════
with tab1:
    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        st.markdown(f"""
        <div class='result-box'>
            <h3>🔍 Diagnosis Report</h3>
            <p><b>Model:</b> {model_choice}</p>
            <p><b>Capacity:</b> {cap_pred:.4f} Ah &nbsp;&nbsp;
               <span class='badge-{badge_cls}'>{badge_lbl}</span></p>
            <p><b>State of Health:</b> {soh:.2f}%</p>
            <p><b>Remaining Useful Life:</b> {rul:.2f} years</p>
            <p><b>Recommended Use:</b> {reuse}</p>
            <p><b>Pack Grouping:</b> {group}</p>
            <p><b>IoT Monitoring:</b> {'⚠️ Required for safety' if soh < 80 else '✅ Standard monitoring'}</p>
        </div>
        """, unsafe_allow_html=True)

        # Model comparison table
        st.markdown("<div class='section-header'>📊 Model Leaderboard</div>", unsafe_allow_html=True)
        lb = pd.DataFrame({
            "Model": list(results.keys()),
            "MAE":   [results[k]["MAE"] for k in results],
            "R² Score": [results[k]["R2"] for k in results],
        }).sort_values("R² Score", ascending=False).reset_index(drop=True)
        lb.index += 1
        st.dataframe(lb.style.highlight_max("R² Score", color="#0f4c2a")
                              .highlight_min("MAE", color="#0f4c2a")
                              .format({"MAE": "{:.4f}", "R² Score": "{:.4f}"}),
                     use_container_width=True)

    with col_gauge:
        # SOH Gauge (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=soh,
            delta={"reference": 80, "increasing": {"color": "#3fb950"}, "decreasing": {"color": "#f85149"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar":  {"color": "#1f6feb"},
                "steps": [
                    {"range": [0,  60],  "color": "#2d0f0f"},
                    {"range": [60, 70],  "color": "#2d1c0f"},
                    {"range": [70, 80],  "color": "#2d2f0f"},
                    {"range": [80, 100], "color": "#0f4c2a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 80},
            },
            number={"suffix": "%", "font": {"size": 40, "color": "#58a6ff"}},
            title={"text": "State of Health", "font": {"size": 18, "color": "#8b949e"}},
        ))
        dark_plotly(fig_gauge)
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # RUL bar
        fig_rul = go.Figure(go.Bar(
            x=[rul], y=["RUL"], orientation="h",
            marker=dict(color="#1f6feb", line=dict(color="#58a6ff", width=1)),
            text=[f"{rul:.2f} yrs"], textposition="inside",
        ))
        fig_rul.add_vline(x=5, line_dash="dash", line_color="#f85149", annotation_text="Max Life")
        dark_plotly(fig_rul, "Remaining Useful Life")
        fig_rul.update_layout(height=160, showlegend=False,
                              xaxis=dict(range=[0, 5.5]))
        st.plotly_chart(fig_rul, use_container_width=True)

# ═══════════════════════════════════════
#  TAB 2 · BATTERY DEGRADATION CURVE
# ═══════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📉 Battery Degradation Curve (matplotlib + seaborn)</div>", unsafe_allow_html=True)

    per_cycle = df.groupby("id_cycle").agg(
        Capacity=("Capacity", "mean"),
        Temp=("Temperature_measured", "mean"),
    ).reset_index()
    per_cycle["SOH"] = calculate_soh(per_cycle["Capacity"])

    # --- matplotlib figure ---
    fig_deg, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")

    # Left: Capacity vs Cycles
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.plot(per_cycle["id_cycle"], per_cycle["Capacity"],
             color="#58a6ff", linewidth=1.5, alpha=0.7, label="Measured")

    # Trend line
    z = np.polyfit(per_cycle["id_cycle"], per_cycle["Capacity"], 2)
    p = np.poly1d(z)
    xs = np.linspace(per_cycle["id_cycle"].min(), per_cycle["id_cycle"].max(), 300)
    ax1.plot(xs, p(xs), color="#f85149", linewidth=2.5, linestyle="--", label="Trend")
    ax1.axhline(y=0.8 * INITIAL_CAPACITY, color="#f0883e", linewidth=1.5,
                linestyle=":", label="80% EOL threshold")

    # Mark current prediction
    ax1.scatter([cycle], [cap_pred], color="#3fb950", s=120, zorder=5, label="Your Battery")

    ax1.set_xlabel("Cycle Number", color="#8b949e")
    ax1.set_ylabel("Capacity (Ah)", color="#8b949e")
    ax1.set_title("Capacity vs Charge Cycles", color="#58a6ff", fontweight="bold")
    ax1.tick_params(colors="#8b949e")
    ax1.spines[:].set_color("#30363d")
    legend = ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax1.grid(alpha=0.15, color="#8b949e")

    # Right: SOH distribution (seaborn KDE)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    sns.kdeplot(per_cycle["SOH"], ax=ax2, fill=True,
                color="#58a6ff", alpha=0.4, linewidth=2)
    ax2.axvline(x=soh, color="#3fb950", linewidth=2.5, linestyle="--",
                label=f"Your battery: {soh:.1f}%")
    ax2.axvline(x=80, color="#f85149", linewidth=1.5, linestyle=":",
                label="EOL threshold 80%")
    ax2.set_xlabel("State of Health (%)", color="#8b949e")
    ax2.set_ylabel("Density", color="#8b949e")
    ax2.set_title("SOH Distribution (KDE)", color="#58a6ff", fontweight="bold")
    ax2.tick_params(colors="#8b949e")
    ax2.spines[:].set_color("#30363d")
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.grid(alpha=0.15, color="#8b949e")

    plt.tight_layout(pad=2)
    st.pyplot(fig_deg, use_container_width=True)
    plt.close()

    # --- Plotly interactive degradation ---
    st.markdown("<div class='section-header'>📈 Interactive Degradation Explorer (Plotly)</div>", unsafe_allow_html=True)
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(
        x=per_cycle["id_cycle"], y=per_cycle["Capacity"],
        mode="markers", name="Measured Capacity",
        marker=dict(color=per_cycle["SOH"], colorscale="RdYlGn",
                    size=7, showscale=True, colorbar=dict(title="SOH%")),
        hovertemplate="Cycle: %{x}<br>Capacity: %{y:.3f} Ah<extra></extra>",
    ))
    fig_int.add_trace(go.Scatter(
        x=xs, y=p(xs), mode="lines", name="Trend",
        line=dict(color="#f85149", width=2.5, dash="dash"),
    ))
    fig_int.add_hline(y=0.8 * INITIAL_CAPACITY, line_dash="dot",
                      line_color="#f0883e", annotation_text="80% EOL")
    fig_int.add_vline(x=cycle, line_dash="dash", line_color="#3fb950",
                      annotation_text=f"Cycle {cycle}")
    dark_plotly(fig_int, "Battery Capacity Degradation Over Cycles")
    st.plotly_chart(fig_int, use_container_width=True)

# ═══════════════════════════════════════
#  TAB 3 · HEALTH PREDICTION GRAPH
# ═══════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🤖 Model Prediction vs Actual (Plotly)</div>", unsafe_allow_html=True)

    chosen_res = results[model_choice]
    true_vals  = chosen_res["true"]
    pred_vals  = chosen_res["pred"]

    fig_pred = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted vs Actual Capacity", "Residual Distribution"),
    )

    # Scatter: pred vs actual
    fig_pred.add_trace(go.Scatter(
        x=true_vals, y=pred_vals, mode="markers",
        marker=dict(color="#58a6ff", size=5, opacity=0.6),
        name="Predictions",
        hovertemplate="True: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    # Perfect line
    lo, hi = true_vals.min(), true_vals.max()
    fig_pred.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#f85149", dash="dash", width=2),
        name="Perfect Fit",
    ), row=1, col=1)

    # Residuals histogram (seaborn-style via plotly)
    residuals = pred_vals - true_vals
    fig_pred.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="#58a6ff", opacity=0.75,
        name="Residuals",
    ), row=1, col=2)
    fig_pred.add_vline(x=0, line_dash="dash", line_color="#f85149", row=1, col=2)

    dark_plotly(fig_pred, f"{model_choice} · R²={chosen_res['R2']:.4f}  MAE={chosen_res['MAE']:.4f}")
    fig_pred.update_layout(height=420, showlegend=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    # SOH prediction over future cycles (seaborn)
    st.markdown("<div class='section-header'>🔮 SOH Forecast – All Models (seaborn)</div>", unsafe_allow_html=True)
    future_cycles = np.arange(1, 300, 5)
    fig_soh, ax = plt.subplots(figsize=(13, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    colors_map = {
        "Linear Regression": "#58a6ff",
        "Decision Tree":     "#3fb950",
        "Random Forest":     "#f0883e",
        "SVM":               "#d2a8ff",
    }
    for mname, mres in results.items():
        preds = []
        for c in future_cycles:
            inp = np.array([[c, voltage, current, temp]])
            inp_s = scaler.transform(inp)
            preds.append(calculate_soh(mres["model"].predict(inp_s)[0]))
        ax.plot(future_cycles, preds,
                label=f"{mname} (R²={mres['R2']:.3f})",
                color=colors_map[mname], linewidth=2)

    ax.axhline(80, color="#f85149", linestyle="--", linewidth=1.5, label="EOL 80%")
    ax.axvline(cycle, color="#3fb950", linestyle=":", linewidth=1.5, label=f"Cycle {cycle}")
    ax.set_xlabel("Cycle Number", color="#8b949e")
    ax.set_ylabel("SOH (%)", color="#8b949e")
    ax.set_title("Predicted State of Health – All Models", color="#58a6ff", fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
    ax.grid(alpha=0.15, color="#8b949e")
    st.pyplot(fig_soh, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════
#  TAB 4 · CLUSTER VISUALIZATION
# ═══════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🗺️ K-Means Battery Cluster Map (Plotly 3D + PCA)</div>", unsafe_allow_html=True)

    cluster_labels = {0: "High-Health", 1: "Moderate", 2: "Degraded", 3: "Critical"}
    cluster_colors = ["#3fb950", "#58a6ff", "#f0883e", "#f85149"]
    cycle_data["ClusterLabel"] = cycle_data["Cluster"].map(cluster_labels)

    col_3d, col_pca = st.columns(2)

    with col_3d:
        fig_3d = px.scatter_3d(
            cycle_data, x="Capacity", y="Temperature", z="Voltage",
            color="ClusterLabel",
            color_discrete_sequence=cluster_colors,
            size_max=8, opacity=0.75,
            hover_data={"SOH": ":.1f", "id_cycle": True},
            title="3D Cluster Map",
        )
        fig_3d.update_traces(marker=dict(size=4))
        dark_plotly(fig_3d, "3D Cluster Map – Capacity / Temp / Voltage")
        fig_3d.update_layout(height=420)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_pca:
        fig_pca = px.scatter(
            cycle_data, x="PCA1", y="PCA2",
            color="ClusterLabel",
            color_discrete_sequence=cluster_colors,
            hover_data={"SOH": ":.1f", "id_cycle": True},
            opacity=0.75,
        )
        dark_plotly(fig_pca, "PCA Projection of Battery Clusters")
        fig_pca.update_layout(height=420)
        st.plotly_chart(fig_pca, use_container_width=True)

    # Seaborn cluster heatmap (mean features per cluster)
    st.markdown("<div class='section-header'>🔥 Cluster Feature Heatmap (seaborn)</div>", unsafe_allow_html=True)
    cluster_means = cycle_data.groupby("ClusterLabel")[
        ["Capacity", "Temperature", "Voltage", "SOH"]
    ].mean()

    fig_hm, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        cluster_means.T, annot=True, fmt=".2f", cmap="Blues",
        linewidths=0.5, linecolor="#30363d",
        annot_kws={"color": "#e6edf3", "size": 11},
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Mean Feature Values per Cluster", color="#58a6ff", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    plt.tight_layout()
    st.pyplot(fig_hm, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════
#  TAB 5 · TEMPERATURE MONITORING
# ═══════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🌡️ Temperature Monitoring Dashboard (Plotly + matplotlib)</div>", unsafe_allow_html=True)

    sample = df.sample(n=min(2000, len(df)), random_state=99).sort_values("id_cycle")

    # Plotly line chart – temperature vs cycle (colored by SOH zone)
    sample["SOH_zone"] = pd.cut(
        calculate_soh(sample["Capacity"]),
        bins=[0, 60, 70, 80, 100],
        labels=["Critical <60%", "Fair 60-70%", "Good 70-80%", "Excellent >80%"],
    )
    fig_temp = px.scatter(
        sample, x="id_cycle", y="Temperature_measured",
        color="SOH_zone",
        color_discrete_map={
            "Excellent >80%":  "#3fb950",
            "Good 70-80%":     "#58a6ff",
            "Fair 60-70%":     "#f0883e",
            "Critical <60%":   "#f85149",
        },
        opacity=0.5, size_max=4,
        labels={"Temperature_measured": "Temperature (°C)", "id_cycle": "Cycle"},
    )
    # Running mean
    rm = sample.groupby("id_cycle")["Temperature_measured"].mean().reset_index()
    fig_temp.add_trace(go.Scatter(
        x=rm["id_cycle"], y=rm["Temperature_measured"],
        mode="lines", name="Running Mean",
        line=dict(color="white", width=2.5),
    ))
    fig_temp.add_hline(y=45, line_dash="dash", line_color="#f85149",
                       annotation_text="⚠️ Warning 45°C")
    dark_plotly(fig_temp, "Temperature vs Charge Cycles (colored by SOH zone)")
    fig_temp.update_layout(height=380)
    st.plotly_chart(fig_temp, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Seaborn boxplot – temp per SOH zone
        st.markdown("<div class='section-header'>📦 Temperature by Health Zone (seaborn)</div>", unsafe_allow_html=True)
        sample["SOH_pct"] = calculate_soh(sample["Capacity"])
        sample["Zone"] = pd.cut(
            sample["SOH_pct"],
            bins=[0, 60, 70, 80, 100],
            labels=["Critical", "Fair", "Good", "Excellent"],
        )
        fig_box, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        palette = {"Excellent": "#3fb950", "Good": "#58a6ff",
                   "Fair": "#f0883e", "Critical": "#f85149"}
        sns.boxplot(data=sample, x="Zone", y="Temperature_measured",
                    palette=palette, order=["Excellent","Good","Fair","Critical"],
                    linewidth=1.5, flierprops={"marker":"o","markersize":3,"alpha":0.4}, ax=ax)
        ax.set_xlabel("Health Zone", color="#8b949e")
        ax.set_ylabel("Temperature (°C)", color="#8b949e")
        ax.set_title("Temperature Distribution by SOH Zone", color="#58a6ff", fontweight="bold")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.grid(axis="y", alpha=0.15, color="#8b949e")
        plt.tight_layout()
        st.pyplot(fig_box, use_container_width=True)
        plt.close()

    with col_b:
        # Plotly heatmap – temp vs voltage correlation
        st.markdown("<div class='section-header'>🔥 Temp × Voltage Density (Plotly)</div>", unsafe_allow_html=True)
        fig_dens = go.Figure(go.Histogram2dContour(
            x=sample["Voltage_measured"],
            y=sample["Temperature_measured"],
            colorscale="Blues",
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            line=dict(width=0.5),
            hovertemplate="Voltage: %{x:.2f}V<br>Temp: %{y:.1f}°C<extra></extra>",
        ))
        dark_plotly(fig_dens, "Voltage × Temperature Density")
        fig_dens.update_layout(
            height=400,
            xaxis_title="Voltage (V)",
            yaxis_title="Temperature (°C)",
        )
        st.plotly_chart(fig_dens, use_container_width=True)

    # Alert banner
    alert_color = "#2d0f0f" if temp > 45 else "#0f2d0f"
    alert_icon  = "⚠️ HIGH TEMP ALERT" if temp > 45 else "✅ Temperature Normal"
    alert_txt   = "#f85149" if temp > 45 else "#3fb950"
    st.markdown(f"""
    <div style='background:{alert_color}; border:1px solid {alert_txt};
                border-radius:10px; padding:0.8rem 1.5rem; margin-top:1rem;'>
        <span style='color:{alert_txt}; font-size:1.1rem; font-weight:700;'>{alert_icon}</span>
        &nbsp;&nbsp;Entered temperature: <b>{temp}°C</b>
        {'— Exceeds safe operating threshold of 45°C. Check thermal management.' if temp > 45 else '— Within safe operating range.'}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:0.8rem; padding: 0.5rem;'>
    ⚡ Battery Health AI · Streamlit · sklearn · Plotly · Matplotlib · Seaborn
    &nbsp;|&nbsp; Models: LinearRegression · DecisionTree · RandomForest · SVR · KMeans
</div>
""", unsafe_allow_html=True)

