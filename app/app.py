"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         🏥 AI-Powered Diabetes Prediction System — Healthcare Dashboard      ║
║         Built with Streamlit + Plotly | Random Forest ML Model               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import warnings
warnings.filterwarnings("ignore")
# This finds the exact folder where app.py is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI | Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — PREMIUM MEDICAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #273549;
    --border: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --gradient-main: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0f172a 50%, #0d1b2e 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Main Content Padding ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0c1f3a 0%, #0f172a 40%, #1a0533 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(14,165,233,0.15);
    border: 1px solid rgba(14,165,233,0.3);
    color: #38bdf8;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--primary);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.blue::before  { background: var(--gradient-main); }
.metric-card.green::before { background: var(--gradient-success); }
.metric-card.red::before   { background: var(--gradient-danger); }
.metric-card.amber::before { background: var(--gradient-warning); }
.metric-card.purple::before { background: linear-gradient(135deg, #8b5cf6, #6d28d9); }

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-icon {
    font-size: 2rem;
    position: absolute;
    top: 1.2rem;
    right: 1.4rem;
    opacity: 0.25;
}

/* ── Result Cards ── */
.result-card-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-card-negative {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%);
    border: 1.5px solid rgba(16,185,129,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-card-moderate {
    background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(217,119,6,0.06) 100%);
    border: 1.5px solid rgba(245,158,11,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-sub {
    font-size: 0.9rem;
    color: var(--text-secondary);
}
.confidence-pill {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 1rem;
}

/* ── Info Card ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.info-card h4 {
    color: var(--primary);
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-pill {
    background: rgba(14,165,233,0.15);
    border: 1px solid rgba(14,165,233,0.2);
    color: #38bdf8;
    padding: 3px 12px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Health Tip Cards ── */
.tip-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: background 0.2s;
}
.tip-card:hover { background: var(--bg-card-hover); }
.tip-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.tip-text { color: var(--text-secondary); font-size: 0.88rem; line-height: 1.5; }
.tip-title { color: var(--text-primary); font-weight: 600; font-size: 0.9rem; margin-bottom: 3px; }

/* ── Disclaimer ── */
.disclaimer-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 2rem;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.disclaimer-icon { font-size: 1.4rem; flex-shrink: 0; }
.disclaimer-text { font-size: 0.82rem; color: #fbbf24; line-height: 1.6; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    color: var(--text-secondary);
    font-size: 0.8rem;
}
.footer a { color: var(--primary); text-decoration: none; }

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSlider"] > div { color: var(--text-primary) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-main) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gradient-main) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.4) !important;
}

/* ── Number inputs ── */
div[data-testid="stNumberInput"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] > div > div > div {
    background: var(--gradient-main) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: var(--gradient-main) !important;
}

/* ── Spinbox ── */
div[data-testid="stSpinner"] * { color: var(--primary) !important; }

/* ── Alert / Info ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Sidebar radio ── */
div[data-testid="stSidebar"] .stRadio > label { color: var(--text-primary) !important; }
div[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-weight: 500 !important;
}

/* ── Input label color ── */
label[data-testid="stWidgetLabel"] * { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#   UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    try:
        dataset_path = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
        df = pd.read_csv(dataset_path)
        return df

    except FileNotFoundError:
        st.error("Dataset file not found.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """Load trained model and scaler."""
    
    model = None
    scaler = None

    model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Scaler file not found.")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")

    return model, scaler


@st.cache_resource(show_spinner=False)
def load_metrics():
    # Construct the absolute path
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.pkl")
    
    try:
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
        return metrics
    
    except FileNotFoundError:
        # It's good practice to return None so your safety check can catch it
        st.warning(f"Metrics file not found at {metrics_path}")
        return None
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


def validate_inputs(pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, dpf, age):
    """Return list of (field_name, message) validation errors."""
    errors = []
    if pregnancies < 0 or pregnancies > 20:
        errors.append(("Pregnancies", "Must be between 0 and 20."))
    if glucose < 40 or glucose > 400:
        errors.append(("Glucose", "Must be between 40 and 400 mg/dL."))
    if blood_pressure < 40 or blood_pressure > 250:
        errors.append(("Blood Pressure", "Must be between 40 and 250 mmHg."))
    if skin_thickness < 0 or skin_thickness > 100:
        errors.append(("Skin Thickness", "Must be between 0 and 100 mm."))
    if insulin < 0 or insulin > 1000:
        errors.append(("Insulin", "Must be between 0 and 1000 μU/mL."))
    if bmi < 10.0 or bmi > 70.0:
        errors.append(("BMI", "Must be between 10.0 and 70.0 kg/m²."))
    if dpf < 0.0 or dpf > 3.0:
        errors.append(("Diabetes Pedigree Function", "Must be between 0.0 and 3.0."))
    if age < 1 or age > 120:
        errors.append(("Age", "Must be between 1 and 120 years."))
    return errors


def risk_level(prob):
    """Return (label, colour_hex, emoji) from probability 0-1."""
    if prob < 0.30:
        return "Low Risk", "#10b981", "🟢"
    elif prob < 0.60:
        return "Moderate Risk", "#f59e0b", "🟡"
    else:
        return "High Risk", "#ef4444", "🔴"



# ═════════════════════════════════════════════════════════
#   PLOTLY CHART BUILDERS
# ═════════════════════════════════════════════════════════

PLOT_BG    = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#94a3b8"
FONT_FAMILY = "Inter, sans-serif"

BASE_LAYOUT = dict(
    plot_bgcolor=PLOT_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
    margin=dict(l=10, r=10, t=40, b=10),
)


def gauge_chart(prob: float) -> go.Figure:
    label, color, _ = risk_level(prob)
    pct = round(prob * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#10b981"}},
        number={"suffix": "%", "font": {"size": 44, "color": "#f1f5f9"}},
        title={"text": f"<b>Diabetes Risk Score</b><br><span style='color:{color};font-size:0.9em'>{label}</span>",
               "font": {"size": 15, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(16,185,129,0.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.8,
                "value": pct
            },
        }
    ))
    fig.update_layout(**BASE_LAYOUT, height=300)
    return fig


def donut_chart(prob: float) -> go.Figure:
    label, color, _ = risk_level(prob)
    neg_color = "#10b981" if prob < 0.5 else "#334155"
    fig = go.Figure(go.Pie(
        values=[prob * 100, (1 - prob) * 100],
        labels=["Diabetic Probability", "Non-Diabetic Probability"],
        hole=0.68,
        marker=dict(
            colors=[color, "#1e293b"],
            line=dict(color="#0f172a", width=2)
        ),
        textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{round(prob*100,1)}%</b>",
        x=0.5, y=0.55, font=dict(size=26, color="#f1f5f9"), showarrow=False
    )
    fig.add_annotation(
        text=label,
        x=0.5, y=0.38, font=dict(size=12, color=color), showarrow=False
    )
    fig.update_layout(**BASE_LAYOUT, height=300,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                                  xanchor="center", x=0.5,
                                  font=dict(color="#94a3b8", size=11)))
    return fig


def feature_importance_chart(model) -> go.Figure:
    features = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age"
    ]

    importance_values = getattr(model, "feature_importances_", None)

    if importance_values is None:
        return go.Figure()

    sorted_pairs = sorted(zip(importance_values, features), reverse=True)
    vals, feats = zip(*sorted_pairs)
    colors = ["#0ea5e9" if v == max(vals) else
              "#6366f1" if v > 0.12 else
              "#8b5cf6" if v > 0.08 else "#334155"
              for v in vals]

    fig = go.Figure(go.Bar(
        x=list(vals),
        y=list(feats),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Feature Importance — Random Forest", font=dict(color="#f1f5f9", size=14), x=0.02),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                   tickfont=dict(color=TEXT_COLOR), range=[0, max(vals)*1.25]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=12)),
        height=360,
    )
    return fig


def model_metrics_chart(metrics_data) -> go.Figure:

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

    values = [
        metrics_data["accuracy"],
        metrics_data["precision"],
        metrics_data["recall"],
        metrics_data["f1"],
        metrics_data["roc_auc"]
    ]
    colors   = ["#0ea5e9", "#6366f1", "#10b981", "#f59e0b", "#ec4899"]

    fig = go.Figure()
    for i, (m, v, c) in enumerate(zip(metrics, values, colors)):
        fig.add_trace(go.Bar(
            name=m, x=[m], y=[v],
            marker=dict(
                color=c,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"{v*100:.1f}%"],
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=12),
            hovertemplate=f"<b>{m}</b>: {v*100:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Model Performance Metrics", font=dict(color="#f1f5f9", size=14), x=0.02),
        showlegend=False,
        yaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_COLOR)),
        xaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=12)),
        height=340,
        bargap=0.35,
    )
    return fig


def analytics_scatter(input_glucose: float, input_bmi: float) -> go.Figure:
    """Synthetic population scatter to give analytics context."""
    rng = np.random.default_rng(42)
    n = 600
    glu = rng.normal(120, 35, n).clip(60, 300)
    bmi_v = rng.normal(30, 8, n).clip(15, 65)
    labels = ((glu > 140) & (bmi_v > 28)).astype(int)

    fig = go.Figure()
    for lbl, name, color in [(0, "Non-Diabetic", "#10b981"), (1, "Diabetic", "#ef4444")]:
        mask = labels == lbl
        fig.add_trace(go.Scatter(
            x=glu[mask], y=bmi_v[mask],
            mode="markers",
            name=name,
            marker=dict(color=color, size=5, opacity=0.45,
                        line=dict(width=0)),
            hovertemplate=f"Glucose: %{{x:.0f}}<br>BMI: %{{y:.1f}}<br>{name}<extra></extra>",
        ))

    # Current patient
    fig.add_trace(go.Scatter(
        x=[input_glucose], y=[input_bmi],
        mode="markers",
        name="Your Values",
        marker=dict(color="#fbbf24", size=16, symbol="star",
                    line=dict(color="#f59e0b", width=2)),
        hovertemplate=f"<b>Your Patient</b><br>Glucose: {input_glucose:.0f}<br>BMI: {input_bmi:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Population Analytics — Glucose vs BMI", font=dict(color="#f1f5f9", size=14), x=0.02),
        xaxis=dict(title="Glucose (mg/dL)", gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR)),
        yaxis=dict(title="BMI (kg/m²)", gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR)),
        height=360,
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color="#94a3b8")),
    )
    return fig


# ═════════════════════════════════════════════════════════
#   SIDEBAR
# ═════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1.2rem 0 0.5rem'>
            <div style='font-size:2.8rem'>🏥</div>
            <div style='font-size:1.1rem;font-weight:800;color:#38bdf8;margin:4px 0 2px'>DiabetesAI</div>
            <div style='font-size:0.72rem;color:#64748b;letter-spacing:0.08em;text-transform:uppercase'>
                Healthcare Analytics Platform
            </div>
        </div>
        <hr style='border:none;border-top:1px solid #1e293b;margin:1rem 0'>
        """, unsafe_allow_html=True)

        st.markdown("**🗂️ Navigation**")
        page = st.radio(
            "Navigate",
            ["🔬 Prediction Engine", "📊 Analytics Dashboard", "ℹ️ About & Help"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:1rem 0'>",
                    unsafe_allow_html=True)

        with st.expander("⚙️ Model Configuration", expanded=False):
            st.markdown("<p style='color:#94a3b8;font-size:0.82rem'>Settings for inference</p>",
                        unsafe_allow_html=True)
            threshold = st.slider("Decision Threshold", 0.30, 0.70, 0.50, 0.01,
                                  help="Probability threshold for positive prediction")
            st.caption(f"Current threshold: **{threshold:.2f}**")

        with st.expander("📋 Quick Reference", expanded=False):
            st.markdown("""
            <div style='font-size:0.8rem;color:#94a3b8;line-height:1.8'>
            <b style='color:#38bdf8'>Glucose:</b> 70–140 mg/dL (normal)<br>
            <b style='color:#38bdf8'>BMI:</b> 18.5–24.9 (healthy)<br>
            <b style='color:#38bdf8'>Blood Pressure:</b> &lt;120 mmHg (optimal)<br>
            <b style='color:#38bdf8'>Insulin:</b> 16–166 μU/mL (normal)<br>
            <b style='color:#38bdf8'>Skin Thickness:</b> 10–50 mm (typical)<br>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:1rem 0'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.73rem;color:#475569;text-align:center;line-height:1.8'>
            🤖 Powered by Random Forest ML<br>
            🩺 Pima Indians Diabetes Dataset<br>
            v2.0 · 2025
        </div>
        """, unsafe_allow_html=True)

    return page, threshold


# ═════════════════════════════════════════════════════════
#   PAGE 1 — PREDICTION ENGINE
# ═════════════════════════════════════════════════════════

def page_prediction(model, scaler,metrics, threshold: float):
    # Hero Banner
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>🤖 AI Powered · Random Forest</div>
        <h1 class='hero-title'>Diabetes Risk Prediction Engine</h1>
        <p class='hero-sub'>
            Enter patient biomarkers below to receive an AI-driven diabetes risk assessment
            powered by a trained Random Forest classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab_input, tab_results, tab_metrics, tab_tips = st.tabs([
        "🩺 Patient Inputs",
        "📈 Prediction Results",
        "🏆 Model Metrics",
        "💡 Health Recommendations",
    ])

    # ── TAB 1: INPUTS ──────────────────────────────────
    with tab_input:
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.5rem'>🩺</span>
            <h2>Patient Biomarker Entry</h2>
            <span class='section-pill'>Step 1 of 2</span>
        </div>
        """, unsafe_allow_html=True)

        with st.form("patient_form", clear_on_submit=False):
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("##### 👶 Demographic & Obstetric")
                pregnancies = st.number_input(
                    "Pregnancies", min_value=0, max_value=20, value=1, step=1,
                    help="Number of times pregnant (0 for males / non-pregnant females)"
                )
                age = st.number_input(
                    "Age (years)", min_value=1, max_value=120, value=30, step=1,
                    help="Patient's age in years"
                )

                st.markdown("##### 🩸 Blood & Metabolic Panel")
                glucose = st.number_input(
                    "Glucose (mg/dL)", min_value=0, max_value=500, value=110, step=1,
                    help="Plasma glucose concentration (2-hour OGTT)"
                )
                insulin = st.number_input(
                    "Insulin (μU/mL)", min_value=0, max_value=900, value=80, step=1,
                    help="2-hour serum insulin level"
                )

            with col2:
                st.markdown("##### 💪 Physical Measurements")
                bmi = st.number_input(
                    "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=26.5,
                    step=0.1, format="%.1f",
                    help="Body Mass Index = weight(kg) / height²(m)"
                )
                blood_pressure = st.number_input(
                    "Blood Pressure (mmHg)", min_value=0, max_value=200, value=72, step=1,
                    help="Diastolic blood pressure"
                )
                skin_thickness = st.number_input(
                    "Skin Thickness (mm)", min_value=0, max_value=100, value=23, step=1,
                    help="Triceps skin fold thickness"
                )

                st.markdown("##### 🧬 Genetic Risk Factor")
                dpf = st.number_input(
                    "Diabetes Pedigree Function", min_value=0.000, max_value=3.000,
                    value=0.450, step=0.001, format="%.3f",
                    help="Likelihood of diabetes based on family history (0–2.42 typical)"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                submitted = st.form_submit_button(
                    "🔬 Run Diabetes Risk Analysis",
                    use_container_width=True,
                )
            with c2:
                reset = st.form_submit_button("🔄 Reset", use_container_width=True)
                if reset:
                    st.session_state.clear()
                    st.rerun()

        # Store inputs in session state
        if submitted:
            errors = validate_inputs(pregnancies, glucose, blood_pressure,
                                     skin_thickness, insulin, bmi, dpf, age)
            if errors:
                for field, msg in errors:
                    st.error(f"❌ **{field}:** {msg}")
            else:
                st.session_state["inputs"] = {
                    "pregnancies": pregnancies, "glucose": glucose,
                    "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
                    "insulin": insulin, "bmi": bmi, "dpf": dpf, "age": age,
                }
                st.session_state["threshold"] = threshold
                st.success("✅ Inputs validated. Click the **Prediction Results** tab to see your analysis.")

    # ── TAB 2: RESULTS ─────────────────────────────────
    with tab_results:
        if "inputs" not in st.session_state:
            st.info("💡 Complete the **Patient Inputs** form and submit to see prediction results here.")
            return

        inp = st.session_state["inputs"]
        thr = st.session_state.get("threshold", 0.50)

        # Feature vector
        features = np.array([[
            inp["pregnancies"], inp["glucose"], inp["blood_pressure"],
            inp["skin_thickness"], inp["insulin"], inp["bmi"],
            inp["dpf"], inp["age"]
        ]])

        # Predict
        with st.spinner("🧠 Running AI inference..."):

            if model is not None and scaler is not None:
                try:
                    scaled = scaler.transform(pd.DataFrame(features))
                    prob = float(model.predict_proba(scaled)[0][1])
                    pred = 1 if prob >= thr else 0

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    return

            else:
                st.error("❌ Model or scaler not loaded properly.")
                return
        
        label, color, emoji = risk_level(prob)
        confidence = round(prob * 100, 1) if pred == 1 else round((1 - prob) * 100, 1)

        # ── Result card ──
        st.markdown("<br>", unsafe_allow_html=True)
        card_class = ("result-card-positive" if pred == 1 and prob >= 0.60
                      else "result-card-moderate" if pred == 1
                      else "result-card-negative")
        result_title = "⚠️ Diabetes Risk Detected" if pred == 1 else "✅ Low Diabetes Risk"
        pill_bg = color
        pill_text = "white"

        st.markdown(f"""
        <div class='{card_class}'>
            <div class='result-title' style='color:{color}'>{result_title}</div>
            <div class='result-sub'>Based on the provided biomarkers and trained Random Forest model</div>
            <span class='confidence-pill' style='background:{pill_bg}22;color:{color};border:1px solid {color}55'>
                {emoji} {label} &nbsp;·&nbsp; Confidence: {confidence:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ──
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.plotly_chart(donut_chart(prob), use_container_width=True,
                            config={"displayModeBar": False})

        # ── Patient Summary ──
        st.markdown("---")
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.4rem'>📋</span>
            <h2>Patient Summary</h2>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4, gap="medium")
        summary_data = [
            ("Glucose", f"{inp['glucose']} mg/dL", "blue",
             "🔴" if inp['glucose'] > 140 else "🟢"),
            ("BMI", f"{inp['bmi']:.1f} kg/m²", "green",
             "🔴" if inp['bmi'] > 30 else "🟡" if inp['bmi'] > 25 else "🟢"),
            ("Age", f"{inp['age']} yrs", "purple", ""),
            ("Diabetes Pedigree", f"{inp['dpf']:.3f}", "amber",
             "🔴" if inp['dpf'] > 1.0 else "🟡" if inp['dpf'] > 0.5 else "🟢"),
        ]
        for col, (lbl, val, clr, flag) in zip(cols, summary_data):
            with col:
                st.markdown(f"""
                <div class='metric-card {clr}'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl} {flag}</div>
                </div>
                """, unsafe_allow_html=True)

        cols2 = st.columns(4, gap="medium")
        summary_data2 = [
            ("Pregnancies", f"{inp['pregnancies']}", "blue", ""),
            ("Blood Pressure", f"{inp['blood_pressure']} mmHg", "green",
             "🔴" if inp['blood_pressure'] > 90 else "🟢"),
            ("Insulin", f"{inp['insulin']} μU/mL", "red",
             "🔴" if inp['insulin'] > 166 else "🟢"),
            ("Skin Thickness", f"{inp['skin_thickness']} mm", "amber", ""),
        ]
        st.markdown("<br>", unsafe_allow_html=True)
        for col, (lbl, val, clr, flag) in zip(cols2, summary_data2):
            with col:
                st.markdown(f"""
                <div class='metric-card {clr}'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl} {flag}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Personalized Health Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📝 Personalized Health Summary", expanded=True):
            high_glucose = inp["glucose"] > 140
            high_bmi     = inp["bmi"] > 30
            high_bp      = inp["blood_pressure"] > 90
            high_ins     = inp["insulin"] > 166
            high_dpf     = inp["dpf"] > 0.8

            flags = []
            if high_glucose: flags.append("**Elevated Glucose** (> 140 mg/dL): Possible impaired glucose tolerance.")
            if high_bmi:     flags.append("**High BMI** (> 30): Classified as obese; consider weight management.")
            if high_bp:      flags.append("**Elevated Diastolic BP** (> 90 mmHg): Monitor for hypertension.")
            if high_ins:     flags.append("**High Insulin** (> 166 μU/mL): Possible insulin resistance.")
            if high_dpf:     flags.append("**High Genetic Risk** (DPF > 0.8): Strong family history indication.")
            if inp["age"] > 45: flags.append("**Age > 45**: Risk of Type 2 diabetes increases with age.")

            if flags:
                st.warning("The following risk factors were identified:")
                for f in flags:
                    st.markdown(f"• {f}")
            else:
                st.success("✅ No critical risk flags detected in the biomarker profile.")

            st.markdown(f"""
            <div class='info-card' style='margin-top:1rem'>
                <h4>🩺 Clinical Interpretation</h4>
                <p style='color:#94a3b8;font-size:0.88rem;line-height:1.7'>
                The AI model assessed this patient's risk as <b style='color:{color}'>{label}</b> with a 
                diabetes probability of <b style='color:{color}'>{prob*100:.1f}%</b> 
                (decision threshold: {thr:.2f}). 
                {'An endocrinology referral is recommended for confirmatory HbA1c testing.' if pred == 1 
                 else 'Routine annual screening is advised. Maintain a healthy lifestyle to reduce long-term risk.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Analytics scatter ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            analytics_scatter(inp["glucose"], inp["bmi"]),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # ── TAB 3: METRICS ─────────────────────────────────
    with tab_metrics:
        if metrics is None:
            st.error("Metrics not loaded.")
            return
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.5rem'>🏆</span>
            <h2>Model Performance Metrics</h2>
            <span class='section-pill'>Random Forest</span>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        kpis = [
        ("Accuracy",  f"{metrics['accuracy']*100:.1f}%", "blue",   "📊"),
        ("Precision", f"{metrics['precision']*100:.1f}%", "green", "🎯"),
        ("Recall",    f"{metrics['recall']*100:.1f}%", "amber", "📡"),
        ("F1 Score",  f"{metrics['f1']*100:.1f}%", "purple", "⚖️"),
        ("ROC AUC",   f"{metrics['roc_auc']*100:.1f}%", "red", "📈"),
        ]
        cols = st.columns(5, gap="small")
        for col, (lbl, val, clr, icon) in zip(cols, kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card {clr}'>
                    <div class='metric-icon'>{icon}</div>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.plotly_chart(model_metrics_chart(metrics), use_container_width=True,
                            config={"displayModeBar": False})
        with col_b:

            if model is not None:
                st.plotly_chart(
                feature_importance_chart(model),
                use_container_width=True,
                config={"displayModeBar": False}
                )
            else:
                st.warning("Feature importance unavailable because model failed to load.")
        
        with st.expander("📖 Metrics Explanation", expanded=False):
            st.markdown("""
            | Metric | Formula | Interpretation |
            |--------|---------|---------------|
            | **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correct predictions |
            | **Precision** | TP/(TP+FP) | Of predicted positives, how many are truly diabetic |
            | **Recall (Sensitivity)** | TP/(TP+FN) | Of true diabetics, how many did we catch |
            | **F1 Score** | 2·P·R/(P+R) | Harmonic mean of Precision & Recall |
            | **ROC AUC** | Area under ROC curve | Discrimination ability across thresholds |

            > *Metrics are computed on a held-out 20% test split of the Pima Indians Diabetes Dataset.*
            """)

        with st.expander("⚙️ Model Configuration", expanded=False):
            cfg = {
                "Algorithm": "Random Forest Classifier",
                "n_estimators": 100,
                "max_depth": "None (unlimited)",
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "Feature Scaling": "StandardScaler",
                "Train/Test Split": "80% / 20%",
                "Random State": 42,
                "Dataset": "Pima Indians Diabetes Database (NIDDK)",
                "Samples": "768 (500 non-diabetic, 268 diabetic)",
            }
            df_cfg = pd.DataFrame(cfg.items(), columns=["Parameter", "Value"])
            st.dataframe(
                df_cfg,
                use_container_width=True,
                hide_index=True,
            )

    # ── TAB 4: HEALTH TIPS ─────────────────────────────
    with tab_tips:
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.5rem'>💡</span>
            <h2>Health Recommendations & Tips</h2>
        </div>
        """, unsafe_allow_html=True)

        tips = [
            ("🥗", "Balanced Diet", "Prioritise whole grains, lean proteins, and fibre-rich vegetables. Limit refined carbohydrates and added sugars to help maintain stable blood glucose levels."),
            ("🏃", "Regular Exercise", "Aim for at least 150 minutes of moderate aerobic activity weekly. Physical activity improves insulin sensitivity and aids in weight management."),
            ("⚖️", "Weight Management", "Achieving and maintaining a healthy BMI (18.5–24.9) significantly reduces Type 2 diabetes risk. Even a 5–7% weight loss lowers risk by ~58%."),
            ("💧", "Stay Hydrated", "Drinking adequate water (8–10 glasses/day) helps kidneys flush out excess glucose through urine and supports metabolic function."),
            ("😴", "Quality Sleep", "Poor sleep disrupts insulin regulation. Target 7–9 hours of quality sleep per night to support hormonal balance and metabolic health."),
            ("🚭", "Avoid Smoking & Alcohol", "Smoking increases insulin resistance by up to 44%. Excessive alcohol impairs glucose regulation and liver function."),
            ("🩺", "Regular Monitoring", "Routine HbA1c, fasting glucose, and lipid panel checks every 6–12 months enable early detection and timely intervention."),
            ("🧘", "Stress Reduction", "Chronic stress elevates cortisol, which raises blood glucose. Practice mindfulness, yoga, or deep-breathing techniques daily."),
            ("👨‍⚕️", "Medical Consultation", "Always consult a certified endocrinologist or diabetologist for personalised management plans, medication adjustments, and lifestyle guidance."),
        ]

        col1, col2 = st.columns(2, gap="large")
        for i, (icon, title, text) in enumerate(tips):
            target = col1 if i % 2 == 0 else col2
            with target:
                st.markdown(f"""
                <div class='tip-card'>
                    <div class='tip-icon'>{icon}</div>
                    <div>
                        <div class='tip-title'>{title}</div>
                        <div class='tip-text'>{text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='disclaimer-box'>
            <div class='disclaimer-icon'>⚠️</div>
            <div class='disclaimer-text'>
                <b>Medical Disclaimer:</b> This tool is designed for <b>educational and research purposes only</b>. 
                It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
                Always seek the advice of a qualified healthcare provider with any questions regarding a 
                medical condition. Do not disregard professional medical advice or delay seeking it because 
                of information provided by this tool. The predictions generated are probabilistic estimates 
                and should be interpreted with clinical judgment.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#   PAGE 2 — ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════

def page_analytics(df):
    
    if df.empty:
        st.error("Dataset unavailable.")
        return
    df = df.copy()
    df["Label"] = df["Outcome"].map({
        0: "Non-Diabetic",
        1: "Diabetic"
    })
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>📊 Population Analytics</div>
        <h1 class='hero-title'>Diabetes Analytics Dashboard</h1>
        <p class='hero-sub'>
            Explore population-level diabetes trends, feature distributions, 
            and model performance on the Pima Indians dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset overview KPIs ──
    kpi_data = [
        ("Total Patients", "768", "blue", "👥"),
        ("Diabetic Cases", "268 (34.9%)", "red", "🔴"),
        ("Non-Diabetic", "500 (65.1%)", "green", "🟢"),
        ("Features Used", "8", "purple", "🧬"),
        ("Model Accuracy", "87.01%", "amber", "🏆"),
    ]
    cols = st.columns(5, gap="small")
    for col, (lbl, val, clr, icon) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class='metric-card {clr}'>
                <div class='metric-icon'>{icon}</div>
                <div class='metric-value' style='font-size:1.6rem'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation heatmap ──
    st.markdown("<br>", unsafe_allow_html=True)
    num_df = df[["Glucose","BMI","Age","BloodPressure","Insulin","DiabetesPedigreeFunction","SkinThickness","Pregnancies","Outcome"]]
    corr = num_df.corr().round(2)
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="%{y} × %{x}: %{z}<extra></extra>",
    ))
    fig_hm.update_layout(
        **BASE_LAYOUT,
        height=420,
        title=dict(text="Feature Correlation Matrix", font=dict(color="#f1f5f9", size=14), x=0.02),
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    # ── Descriptive stats ──
    
    with st.expander("📊 Descriptive Statistics by Class", expanded=False):

        stats_df = (
                df.groupby("Label")[[
                    "Glucose",
                    "BMI",
                    "Age",
                    "BloodPressure",
                    "Insulin",
                    "DiabetesPedigreeFunction",
                    "SkinThickness",
                    "Pregnancies"
                ]]
                .describe()
                .round(2)
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════
#   PAGE 3 — ABOUT & HELP
# ═════════════════════════════════════════════════════════

def page_about():
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>ℹ️ About This System</div>
        <h1 class='hero-title'>About DiabetesAI</h1>
        <p class='hero-sub'>
            A production-grade AI-powered clinical decision support tool for early diabetes risk stratification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>🎯 Project Overview</h4>
            <p style='color:#94a3b8;font-size:0.88rem;line-height:1.8'>
            DiabetesAI leverages a Random Forest classifier trained on the 
            <b style='color:#38bdf8'>Pima Indians Diabetes Database</b> (NIDDK) to predict 
            the likelihood of diabetes based on eight clinical biomarkers. 
            The system is designed as a clinical decision-support aid, providing 
            probabilistic risk scores with interpretable feature importance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬 How the Model Works", expanded=True):
            st.markdown("""
            **1. Data Collection**  
            The model is trained on 768 patient records from the Pima Indians Diabetes Dataset,
            featuring 8 clinical measurements.

            **2. Preprocessing**  
            - Missing/zero values handled via median imputation  
            - Features standardized using `StandardScaler`  

            **3. Model Architecture**  
            - **Algorithm:** Random Forest Classifier  
            - **Ensemble of 100 decision trees** trained with bootstrapped samples  
            - **Majority voting** aggregates individual tree predictions  

            **4. Inference Pipeline**  
            ```
            Patient Input → Validation → StandardScaler → RF Model → Probability → Risk Level
            ```

            **5. Output Interpretation**  
            - `< 30%` → Low Risk  
            - `30–60%` → Moderate Risk  
            - `> 60%` → High Risk  
            """)

        with st.expander("📚 Dataset Information", expanded=False):
            st.markdown("""
            **Pima Indians Diabetes Database**  
            *Source:* National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)  
            *Hosted by:* UCI Machine Learning Repository / Kaggle  

            | Feature | Unit | Description |
            |---------|------|-------------|
            | Pregnancies | count | Number of pregnancies |
            | Glucose | mg/dL | 2-hr plasma glucose (OGTT) |
            | Blood Pressure | mmHg | Diastolic blood pressure |
            | Skin Thickness | mm | Triceps skin fold |
            | Insulin | μU/mL | 2-hr serum insulin |
            | BMI | kg/m² | Body Mass Index |
            | DPF | — | Diabetes pedigree function |
            | Age | years | Patient age |
            | Outcome | 0/1 | 0 = No diabetes, 1 = Diabetes |
            """)

    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🛠️ Technology Stack</h4>
            <div style='color:#94a3b8;font-size:0.85rem;line-height:2.2'>
            🐍 <b>Python 3.9+</b><br>
            🚀 <b>Streamlit</b> — Web framework<br>
            🤖 <b>Scikit-learn</b> — ML model<br>
            📊 <b>Plotly</b> — Interactive charts<br>
            🔢 <b>NumPy</b> — Numerical computing<br>
            🐼 <b>Pandas</b> — Data manipulation<br>
            🥒 <b>Pickle</b> — Model serialisation<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card' style='margin-top:1rem'>
            <h4>👩‍💻 Developer</h4>
            <div style='font-size:0.85rem;color:#94a3b8;line-height:1.8'>
            Developed by <b style='color:#38bdf8'>Gargi Ghosh</b><br>
            AI & Full Stack Developer · Healthcare Analytics Enthusiast
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: #111827;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #1f2937;
            color: #e5e7eb;
            font-family: monospace;
            font-size: 16px;
            line-height: 1.6;
        ">

        📂 <b style="color:#93c5fd">Project Structure</b><br><br>

        Diabetes Prediction System/<br>
        ├── app/<br>
        │  └── app.py<br>
        ├── dataset/<br>
        |   └── diabetes.csv<br>
        └── models/<br>
        &nbsp;&nbsp;&nbsp;├── random_forest_model.pkl<br>
        &nbsp;&nbsp;&nbsp;├── scaler.pkl<br>
        &nbsp;&nbsp;&nbsp;└── model_metrics.pkl<br>

        </div>
        """, unsafe_allow_html=True)
         
        st.markdown("""
        <div style="
            background: #0f172a;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #334155;
        ">

        <div style="color:#38bdf8; font-weight:600; margin-bottom:10px;">
        🚀 Launch Application
        </div>

        <div style="color:#cbd5e1; font-size:13px; margin-bottom:8px;">
        Run these commands:
        </div>

        <code style="
            display:block;
            background:#1e293b;
            padding:8px;
            border-radius:6px;
            color:#e2e8f0;
            margin-bottom:6px;
        ">cd app</code>

        <code style="
            display:block;
            background:#1e293b;
            padding:8px;
            border-radius:6px;
            color:#e2e8f0;
        ">streamlit run app.py</code>

        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card' style='margin-top:1rem'>
            <h4>⚠️ Limitations</h4>
            <div style='font-size:0.82rem;color:#94a3b8;line-height:1.8'>
            • Trained on a single demographic (Pima women)<br>
            • Small dataset (768 samples)<br>
            • No temporal/longitudinal features<br>
            • Not validated in clinical settings<br>
            • Binary outcome only (no prediabetes)<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer-box'>
        <div class='disclaimer-icon'>⚠️</div>
        <div class='disclaimer-text'>
            <b>Medical Disclaimer:</b> DiabetesAI is strictly for <b>educational and research purposes only</b>.
            It does not constitute medical advice, diagnosis, or treatment recommendations.
            All patient data entered remains local and is never stored or transmitted.
            Consult a licensed healthcare professional for any clinical decisions.
            This tool complies with educational fair-use principles. 
            The developers accept no liability for clinical use of this tool.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#   FOOTER
# ═════════════════════════════════════════════════════════

def render_footer():
    st.markdown("""
    <div class='footer'>
        <p>
            🏥 <b>DiabetesAI</b> · AI-Powered Healthcare Analytics Platform ·
            Built with ❤️ using Streamlit & Scikit-learn
        </p>
        <p style='margin-top:0.3rem'>
            Dataset: Pima Indians Diabetes Database (NIDDK) ·
            Model: Random Forest Classifier ·
            <span style='color:#ef4444'>Not for clinical use</span>
        </p>
        <p style='margin-top:0.3rem;color:#334155'>© 2025 DiabetesAI · All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#   MAIN
# ═════════════════════════════════════════════════════════

def main():
    # Load model assets
    model, scaler = load_model_and_scaler()
    
    # Load the metrics
    metrics = load_metrics()
    
    # Load the real dataset
    df = load_dataset()


    # Sidebar returns selected page and threshold
    page, threshold = render_sidebar()

    # Route to selected page
    if page == "🔬 Prediction Engine":
        page_prediction(model, scaler, metrics, threshold)
    elif page == "📊 Analytics Dashboard":
        page_analytics(df)
    elif page == "ℹ️ About & Help":
        page_about()

    # Persistent footer
    render_footer()


if __name__ == "__main__":
    main()
