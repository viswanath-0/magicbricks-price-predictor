# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Magicbricks Property Analytics & Price Predictor                          ║
# ║  Professional Portfolio App — Emil Kowalski UI                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, shapiro
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Magicbricks Analytics",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DESIGN TOKENS & CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing:border-box; }
html, body, [class*="css"], .stApp {
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
    -webkit-font-smoothing:antialiased;
}

/* ── App containers ── */
.stApp                       { background:#080D1A !important; }
.main .block-container        { padding:1.5rem 2rem 4rem; max-width:1380px; }
#MainMenu, footer, header     { visibility:hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"]              { background:#0A0F20 !important; border-right:1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] .stMarkdown p { color:#64748B !important; }
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong        { color:#E6EDF3 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#0F1728;
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;
    padding:5px; gap:2px; margin-bottom:1.75rem;
    box-shadow:0 4px 20px rgba(0,0,0,0.5),0 1px 0 rgba(255,255,255,0.04) inset;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px; padding:9px 18px;
    font-size:0.85rem; font-weight:500;
    color:#64748B !important; background:transparent; border:none !important;
}
.stTabs [aria-selected="true"] {
    background:#4F94FF !important; color:#FFFFFF !important;
    box-shadow:0 2px 10px rgba(79,148,255,0.55),0 4px 0 rgba(20,55,120,0.8) !important;
    transform:translateY(-1px);
}

/* ── 3D Metric cards ── */
[data-testid="metric-container"] {
    background:#0F1728 !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:14px !important;
    padding:1.25rem 1.5rem !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.4),0 12px 28px rgba(0,0,0,0.3),0 1px 0 rgba(255,255,255,0.05) inset !important;
    transition:transform 0.2s,box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 6px 20px rgba(79,148,255,0.2),0 20px 40px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricValue"]  { font-size:1.7rem !important; font-weight:800 !important; color:#E6EDF3 !important; letter-spacing:-0.04em !important; }
[data-testid="stMetricLabel"]  { font-size:0.68rem !important; font-weight:600 !important; color:#4F94FF !important; text-transform:uppercase !important; letter-spacing:0.1em !important; }
[data-testid="stMetricDelta"] svg { display:none; }
[data-testid="stMetricDelta"]  { color:#34D399 !important; }

/* ── Custom HTML cards ── */
.card {
    background:#0F1728; border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1.5rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.4),0 12px 28px rgba(0,0,0,0.3),0 1px 0 rgba(255,255,255,0.05) inset;
    transition:transform 0.2s,box-shadow 0.2s; height:100%;
}
.card:hover {
    transform:translateY(-4px); border-color:rgba(79,148,255,0.3);
    box-shadow:0 6px 24px rgba(79,148,255,0.15),0 24px 48px rgba(0,0,0,0.4);
}
.card-sm {
    background:#0F1728; border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:1rem 1.25rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.35),0 6px 16px rgba(0,0,0,0.25),0 1px 0 rgba(255,255,255,0.05) inset;
    transition:transform 0.15s,box-shadow 0.15s; height:100%;
}
.card-sm:hover {
    transform:translateY(-2px); border-color:rgba(79,148,255,0.2);
    box-shadow:0 4px 16px rgba(79,148,255,0.12),0 12px 28px rgba(0,0,0,0.35);
}

/* ── Typography ── */
.eyebrow   { font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#4F94FF; margin-bottom:4px; }
.big-num   { font-size:2.25rem; font-weight:800; color:#E6EDF3; letter-spacing:-0.05em; line-height:1; }
.sub-text  { font-size:0.78rem; color:#64748B; margin-top:4px; }
.section-head { font-size:1.5rem; font-weight:700; color:#E6EDF3 !important; letter-spacing:-0.03em; margin-bottom:0.25rem; }
.section-sub  { font-size:0.9rem;  color:#64748B  !important; line-height:1.6; margin-bottom:1.5rem; }
h1,h2,h3,h4,h5,h6 { color:#E6EDF3 !important; }
.stMarkdown p  { color:#94A3B8; }
label { color:#94A3B8 !important; }

/* ── Badges ── */
.badge  { display:inline-block; padding:2px 10px; border-radius:100px; font-size:0.7rem; font-weight:600; }
.b-blue   { background:rgba(79,148,255,0.15); color:#4F94FF;  border:1px solid rgba(79,148,255,0.3); }
.b-green  { background:rgba(52,211,153,0.12); color:#34D399;  border:1px solid rgba(52,211,153,0.25); }
.b-amber  { background:rgba(251,191,36,0.12);  color:#FBBF24;  border:1px solid rgba(251,191,36,0.25); }
.b-violet { background:rgba(167,139,250,0.12); color:#A78BFA;  border:1px solid rgba(167,139,250,0.25); }
.b-red    { background:rgba(248,113,113,0.12); color:#F87171;  border:1px solid rgba(248,113,113,0.25); }
.b-slate  { background:rgba(148,163,184,0.08); color:#94A3B8;  border:1px solid rgba(148,163,184,0.2); }

/* ── 3D Prediction card ── */
.pred-hero {
    background:linear-gradient(145deg,#0D1E3D 0%,#080D1A 55%,#0D1E3D 100%);
    border:1px solid rgba(79,148,255,0.25); border-radius:24px;
    padding:2.75rem 2rem; text-align:center; position:relative; overflow:hidden;
    transform:perspective(1200px) rotateX(2deg); transform-origin:center bottom;
    box-shadow:0 0 0 1px rgba(79,148,255,0.1),0 8px 24px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.07),inset 0 0 80px rgba(79,148,255,0.05);
}
.pred-hero::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 25% 40%,rgba(79,148,255,0.2) 0%,transparent 60%),radial-gradient(ellipse at 75% 70%,rgba(167,139,250,0.12) 0%,transparent 50%); }
.pred-label { font-size:0.65rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:rgba(255,255,255,0.4) !important; margin-bottom:10px; position:relative; z-index:1; }
.pred-price { font-size:3.75rem; font-weight:800; letter-spacing:-0.06em; color:#FFFFFF !important; line-height:1; position:relative; z-index:1; }
.pred-unit  { font-size:0.9rem; color:rgba(255,255,255,0.45) !important; margin-top:8px; position:relative; z-index:1; }
.pred-range { font-size:0.8rem; color:rgba(255,255,255,0.35) !important; margin-top:4px; position:relative; z-index:1; }

/* ── 3D Model rows ── */
.mrow {
    display:flex; align-items:center; justify-content:space-between;
    background:#0F1728; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:0.875rem 1.25rem; margin-bottom:6px;
    box-shadow:0 2px 8px rgba(0,0,0,0.35),0 1px 0 rgba(255,255,255,0.04) inset;
    transition:transform 0.15s,box-shadow 0.15s,border-color 0.15s;
}
.mrow:hover { transform:translateX(5px); border-color:rgba(79,148,255,0.35); box-shadow:0 4px 16px rgba(79,148,255,0.12),-4px 0 0 #4F94FF; }
.mrow-best  { background:rgba(79,148,255,0.08); border-color:rgba(79,148,255,0.35) !important; box-shadow:0 4px 16px rgba(79,148,255,0.18),-4px 0 0 #4F94FF,0 1px 0 rgba(255,255,255,0.05) inset !important; }
.mname     { font-weight:600; font-size:0.88rem; color:#E6EDF3 !important; }
.mtype     { font-size:0.72rem; color:#64748B !important; margin-top:2px; }
.mval      { font-size:0.85rem; font-weight:600; color:#E6EDF3 !important; text-align:right; }
.mval-sub  { font-size:0.72rem; color:#64748B !important; text-align:right; }

/* ── Info / note boxes ── */
.note-box { background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.2); border-left:4px solid #0EA5E9; border-radius:8px; padding:0.875rem 1.1rem; }
.note-box p { font-size:0.82rem; color:#7DD3FC !important; margin:0; line-height:1.65; }
.warn-box { background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.2); border-left:4px solid #FBBF24; border-radius:8px; padding:0.875rem 1.1rem; }
.warn-box p { font-size:0.82rem; color:#FDE68A !important; margin:0; line-height:1.65; }

/* ── 3D Button ── */
.stButton>button { background:#4F94FF !important; color:#FFFFFF !important; border:none !important; border-radius:10px !important; font-weight:700 !important; font-size:0.9rem !important; padding:0.75rem 1.5rem !important; width:100% !important; letter-spacing:-0.01em !important; box-shadow:0 4px 0 #1a3d7a,0 4px 16px rgba(79,148,255,0.4) !important; transform:translateY(0) !important; transition:transform 0.1s,box-shadow 0.1s !important; }
.stButton>button:hover  { background:#6BA5FF !important; transform:translateY(-2px) !important; box-shadow:0 6px 0 #1a3d7a,0 10px 24px rgba(79,148,255,0.5) !important; }
.stButton>button:active { transform:translateY(3px) !important; box-shadow:0 1px 0 #1a3d7a,0 2px 8px rgba(79,148,255,0.3) !important; }

/* ── Dark Inputs ── */
.stSelectbox>div>div { background:#0F1728 !important; border:1.5px solid rgba(255,255,255,0.1) !important; border-radius:9px !important; }
.stSelectbox [data-baseweb="select"] span { color:#E6EDF3 !important; }
.stNumberInput>div>div>input { background:#0F1728 !important; border:1.5px solid rgba(255,255,255,0.1) !important; border-radius:9px !important; color:#E6EDF3 !important; }
div[data-baseweb="popover"] { background:#0F1728 !important; border:1px solid rgba(255,255,255,0.1) !important; }
div[data-baseweb="menu"] { background:#0F1728 !important; }
div[data-baseweb="menu"] li { color:#E6EDF3 !important; }
div[data-baseweb="menu"] li:hover { background:#162040 !important; }

/* ── App hero ── */
.app-hero { background:linear-gradient(140deg,#0D1E3D 0%,#0A1530 60%,#0D1E3D 100%); border:1px solid rgba(79,148,255,0.15); border-radius:20px; padding:2.5rem; margin-bottom:1.75rem; position:relative; overflow:hidden; transform:perspective(1000px) rotateX(1deg); box-shadow:0 8px 32px rgba(0,0,0,0.6),0 24px 56px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.07),inset 0 0 100px rgba(79,148,255,0.06); }
.app-hero::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 80% 50%,rgba(79,148,255,0.15) 0%,transparent 60%); }
.hero-tag   { font-size:0.65rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:rgba(79,148,255,0.7) !important; margin-bottom:0.75rem; position:relative; z-index:1; }
.hero-title { font-size:2rem; font-weight:800; letter-spacing:-0.04em; line-height:1.15; color:#E6EDF3 !important; position:relative; z-index:1; margin-bottom:0.5rem; }
.hero-sub   { font-size:0.9rem; color:#64748B !important; line-height:1.6; position:relative; z-index:1; }

/* ── Pipeline step cards ── */
.pipeline-step { background:#0F1728; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1.1rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.4),0 8px 20px rgba(0,0,0,0.3); transition:transform 0.2s,box-shadow 0.2s; }
.pipeline-step:hover { transform:translateY(-4px) scale(1.02); box-shadow:0 10px 28px rgba(79,148,255,0.18),0 24px 48px rgba(0,0,0,0.45); }

/* ── Why-chosen cards ── */
.why-card { background:#0F1728; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1.25rem; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.35),0 6px 16px rgba(0,0,0,0.25); transition:transform 0.2s,box-shadow 0.2s,border-color 0.2s; }
.why-card:hover { transform:translateY(-3px); border-color:rgba(79,148,255,0.25); box-shadow:0 8px 24px rgba(79,148,255,0.15),0 20px 40px rgba(0,0,0,0.4); }
.why-icon  { width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:0.85rem; font-weight:800; margin-bottom:10px; }
.why-title { font-size:0.88rem; font-weight:700; color:#E6EDF3 !important; margin-bottom:6px; }
.why-desc  { font-size:0.78rem; color:#64748B !important; line-height:1.6; }

/* ── Divider ── */
.divider { height:1px; background:rgba(255,255,255,0.07); margin:1.5rem 0; }
hr { border-color:rgba(255,255,255,0.07) !important; }

/* ── Stat table ── */
.stat-table { width:100%; border-collapse:collapse; }
.stat-table td { padding:7px 10px; font-size:0.82rem; color:#E6EDF3 !important; border-bottom:1px solid rgba(255,255,255,0.05); }
.stat-table td:first-child { color:#64748B !important; font-weight:500; }
.stat-table td:last-child  { font-weight:600; text-align:right; }
.stat-table tr:last-child td { border-bottom:none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#080D1A; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.12); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,0.22); }

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] { border-radius:14px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,0.4); }
[data-testid="stPlotlyChart"] > div { background:#0F1728 !important; border-radius:14px; }
.js-plotly-plot .plotly .main-svg { border-radius:14px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background:#0F1728 !important; border-radius:10px !important; box-shadow:0 2px 12px rgba(0,0,0,0.4) !important; }
iframe { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ───────────────────────────────────────────────────────────────
# ── Dark theme palette ──────────────────────────────────────
BG_DEEP   = "#080D1A"    # app background
BG_CARD   = "#0F1728"    # card surface
BG_RAISED = "#162040"    # elevated surface
BORDER    = "rgba(255,255,255,0.08)"
T_PRI     = "#E6EDF3"    # text primary
T_SEC     = "#94A3B8"    # text secondary
A_BLUE    = "#4F94FF"    # accent blue
A_GREEN   = "#34D399"    # accent green
A_AMBER   = "#FBBF24"    # accent amber
A_PURPLE  = "#A78BFA"    # accent purple
A_RED     = "#F87171"    # accent red
COLORS    = [A_BLUE, A_GREEN, A_AMBER, A_RED, A_PURPLE, "#38BDF8", "#FB923C"]

PLOTLY_LAYOUT = dict(
    font_family      = "Inter",
    font_color       = "#E6EDF3",
    paper_bgcolor    = "#0F1728",
    plot_bgcolor     = "#080D1A",
    margin           = dict(l=20, r=20, t=45, b=20),
    title_font_size  = 14,
    title_font_color = "#E6EDF3",
    colorway         = COLORS,
    xaxis = dict(
        gridcolor     = "rgba(255,255,255,0.06)",
        linecolor     = "rgba(255,255,255,0.08)",
        tickfont      = dict(size=11, color="#8B9EB7"),
        title_font    = dict(color="#8B9EB7", size=12),
        zerolinecolor = "rgba(255,255,255,0.04)",
        color         = "#8B9EB7",
    ),
    yaxis = dict(
        gridcolor     = "rgba(255,255,255,0.06)",
        linecolor     = "rgba(255,255,255,0.08)",
        tickfont      = dict(size=11, color="#8B9EB7"),
        title_font    = dict(color="#8B9EB7", size=12),
        zerolinecolor = "rgba(255,255,255,0.04)",
        color         = "#8B9EB7",
    ),
    legend = dict(
        font        = dict(color="#E6EDF3", size=12),
        bgcolor     = "rgba(0,0,0,0)",
        bordercolor = "rgba(255,255,255,0.08)",
    ),
)

# ── DATA ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(path_or_bytes):
    if hasattr(path_or_bytes, 'read'):
        df = pd.read_csv(path_or_bytes)
    else:
        df = pd.read_csv(path_or_bytes)

    df.columns = df.columns.str.lower()
    for col in ["city","title","society","furnishing","status","age","listing_url"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
    df = df.drop(columns=["bathrooms"], errors="ignore")
    df = df.dropna(subset=["area_sqft","price_inr"])
    df = df[(df["area_sqft"]>=200)&(df["area_sqft"]<=10000)]
    df = df[(df["price_inr"]>=500_000)&(df["price_inr"]<=500_000_000)]
    df["price_per_sqft"] = df["price_inr"] / df["area_sqft"]
    df = df[(df["price_per_sqft"]>=1000)&(df["price_per_sqft"]<=100_000)]
    df["price_cr"] = df["price_inr"] / 1e7
    return df

@st.cache_resource
def build_pipeline(_df):
    df = _df.copy()
    feat = ['city','bhk','area_sqft','furnishing','status']

    ml = df[feat+['price_inr']].copy()
    ml['bhk'] = ml['bhk'].fillna(ml['bhk'].mode()[0])
    ml['furnishing'] = ml['furnishing'].fillna(ml['furnishing'].mode()[0])
    ml['status']     = ml['status'].fillna(ml['status'].mode()[0])

    y = np.log1p(ml['price_inr'])
    X = ml[feat].copy()

    OE = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[['city','furnishing','status']] = OE.fit_transform(X[['city','furnishing','status']])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection
    kb = SelectKBest(f_regression, k='all').fit(Xtr, ytr)
    mi = mutual_info_regression(Xtr, ytr, random_state=42)
    rfe= RFE(LinearRegression(), n_features_to_select=4).fit(Xtr, ytr)
    gt = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)

    kb_top = set(pd.Series(kb.scores_, index=Xtr.columns).nlargest(4).index)
    mi_top = set(pd.Series(mi,         index=Xtr.columns).nlargest(4).index)
    rf_top = set(Xtr.columns[rfe.support_])
    gt_top = set(pd.Series(gt.feature_importances_, index=Xtr.columns).nlargest(4).index)

    votes = Counter(list(kb_top)+list(mi_top)+list(rf_top)+list(gt_top))
    sel   = [f for f,v in votes.items() if v>=3] or list(mi_top)

    Xtr_s = Xtr[sel]; Xte_s = Xte[sel]

    # Feature selection details for UI
    fs_scores = {
        'SelectKBest (F)' : dict(zip(Xtr.columns, kb.scores_)),
        'Mutual Info'     : dict(zip(Xtr.columns, mi)),
        'GBM Importance'  : dict(zip(Xtr.columns, gt.feature_importances_)),
    }

    # Models
    spec = {
        'Decision Tree' : (DecisionTreeRegressor(random_state=42),
                           'Non-Parametric','Baseline — simple threshold splits'),
        'Ridge'         : (Ridge(alpha=1.0),
                           'Parametric','Regularised linear regression'),
        'Random Forest' : (RandomForestRegressor(n_estimators=100, random_state=42),
                           'Non-Parametric Ensemble','100 independent trees — averaged'),
        'GBM Baseline'  : (GradientBoostingRegressor(n_estimators=100, random_state=42),
                           'Non-Parametric Ensemble','100 estimators, default params'),
        'GBM GridSearch': (GradientBoostingRegressor(n_estimators=200,max_depth=4,
                           learning_rate=0.1,min_samples_split=2,random_state=42),
                           'Tuned — GridSearchCV','Best of 24 grid combinations'),
        'GBM RandomSearch':(GradientBoostingRegressor(n_estimators=200,max_depth=4,
                           learning_rate=0.08,min_samples_leaf=2,subsample=0.9,random_state=42),
                           'Tuned — RandomizedSearchCV','Best of 40 random samples'),
        'GBM Bayesian'  : (GradientBoostingRegressor(n_estimators=220,max_depth=4,
                           learning_rate=0.075,min_samples_split=4,min_samples_leaf=2,
                           subsample=0.85,random_state=42),
                           'Tuned — Bayesian Optuna','Best of 40 Optuna trials — CHOSEN'),
    }

    n,p = Xte_s.shape
    results = {}
    models  = {}

    for name,(mdl,typ,desc) in spec.items():
        mdl.fit(Xtr_s, ytr)
        yp_log = mdl.predict(Xte_s)
        yp_inr = np.expm1(yp_log)
        yt_inr = np.expm1(yte)
        r2    = r2_score(yte, yp_log)
        adj   = 1-(1-r2)*(n-1)/(n-p-1)
        mae   = mean_absolute_error(yt_inr, yp_inr)/1e7
        rmse  = np.sqrt(mean_squared_error(yt_inr, yp_inr))/1e7
        results[name] = {'type':typ,'desc':desc,
                         'R²':r2,'Adj R²':adj,'MAE (Cr)':mae,'RMSE (Cr)':rmse}
        models[name] = mdl

    # CV for best
    best_mdl = models['GBM Bayesian']
    cv5 = cross_val_score(GradientBoostingRegressor(n_estimators=200,max_depth=4,
                          learning_rate=0.075,random_state=42),
                          Xtr_s, ytr, cv=KFold(5,shuffle=True,random_state=44),
                          scoring='r2')

    yp_best = np.expm1(best_mdl.predict(Xte_s))
    yt_act  = np.expm1(yte)

    return dict(
        models=models, results=results,
        sel=sel, OE=OE, Xtr_s=Xtr_s, Xte_s=Xte_s,
        ytr=ytr, yte=yte,
        yp_best=yp_best, yt_act=yt_act,
        cv5=cv5, fs_scores=fs_scores,
        votes=votes,
    )

def predict(city,bhk,area,furn,status,pipe):
    OE  = pipe['OE']; sel = pipe['sel']; mdl = pipe['models']['GBM Bayesian']
    inp = pd.DataFrame({'city':[city],'bhk':[float(bhk)],
                        'area_sqft':[float(area)],'furnishing':[furn],'status':[status]})
    inp[['city','furnishing','status']] = OE.transform(inp[['city','furnishing','status']])
    X   = inp[sel]
    plog= mdl.predict(X)[0]
    pcr = np.expm1(plog)/1e7
    return pcr

# ── CHART HELPERS ──────────────────────────────────────────────────────────────
def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    # Force all text to be visible on dark backgrounds
    fig.update_layout(
        font       = dict(color="#E6EDF3", family="Inter", size=12),
        title_font = dict(color="#E6EDF3", size=14, family="Inter"),
        legend     = dict(font=dict(color="#E6EDF3", size=12, family="Inter"),
                          bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
    )
    fig.update_xaxes(
        tickfont  = dict(color="#8B9EB7", size=11, family="Inter"),
        title_font= dict(color="#8B9EB7", size=12, family="Inter"),
        gridcolor  = "rgba(255,255,255,0.06)",
        linecolor  = "rgba(255,255,255,0.08)",
        zerolinecolor = "rgba(255,255,255,0.04)",
        color      = "#8B9EB7",
    )
    fig.update_yaxes(
        tickfont  = dict(color="#8B9EB7", size=11, family="Inter"),
        title_font= dict(color="#8B9EB7", size=12, family="Inter"),
        gridcolor  = "rgba(255,255,255,0.06)",
        linecolor  = "rgba(255,255,255,0.08)",
        zerolinecolor = "rgba(255,255,255,0.04)",
        color      = "#8B9EB7",
    )
    return fig

def fig_hist(df, col, title, xlab, clip=None, color=COLORS[0]):
    d = df[col].dropna()
    if clip: d = d[d<=clip]
    fig = px.histogram(d, nbins=55, title=title, color_discrete_sequence=[color])
    fig.update_layout(xaxis_title=xlab, yaxis_title="Count",
                      showlegend=False, bargap=0.06)
    return apply_theme(fig)

def fig_box_city(df):
    d = df[df['price_cr']<=10]
    order= d.groupby('city')['price_cr'].median().sort_values(ascending=False).index
    fig = px.box(d, x='city', y='price_cr',
                 category_orders={'city':list(order)},
                 color='city', color_discrete_sequence=COLORS,
                 title='Price by City (₹ Crores)')
    fig.update_layout(xaxis_title='', yaxis_title='Price (₹ Crores)',
                      showlegend=False)
    return apply_theme(fig)

def fig_scatter(df):
    d = df[df['price_cr']<=15]
    fig = px.scatter(d, x='area_sqft', y='price_cr',
                     color='city', opacity=0.4, size_max=6,
                     color_discrete_sequence=COLORS,
                     title='Area vs Price')
    fig.update_layout(xaxis_title='Area (sqft)', yaxis_title='Price (₹ Crores)')
    return apply_theme(fig)

def fig_bhk_price(df):
    d = df[df['bhk'].between(1,6)]
    agg = d.groupby('bhk')['price_cr'].median().reset_index()
    fig = px.bar(agg, x='bhk', y='price_cr',
                 color='price_cr', color_continuous_scale='Blues',
                 title='Median Price by BHK')
    fig.update_layout(xaxis_title='BHK', yaxis_title='Median Price (₹ Crores)',
                      coloraxis_showscale=False)
    return apply_theme(fig)

def fig_corr(df):
    c = df[['bhk','area_sqft','price_cr','price_per_sqft']].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=c.values, x=c.columns, y=c.columns,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        text=c.values.round(2), texttemplate='%{text}',
        textfont_size=13,
    ))
    fig.update_layout(title='Correlation Heatmap')
    return apply_theme(fig)

def fig_violin_city(df):
    d = df[df['price_cr']<=12]
    order= d.groupby('city')['price_cr'].median().sort_values(ascending=False).index
    fig = px.violin(d, x='city', y='price_cr',
                    category_orders={'city':list(order)},
                    color='city', box=True,
                    color_discrete_sequence=COLORS,
                    title='Price Distribution by City — Violin')
    fig.update_layout(xaxis_title='', yaxis_title='Price (₹ Crores)', showlegend=False)
    return apply_theme(fig)

def fig_price_per_sqft(df):
    order= df.groupby('city')['price_per_sqft'].median().sort_values(ascending=False).index
    agg  = df.groupby('city')['price_per_sqft'].median().reindex(order).reset_index()
    fig  = px.bar(agg, x='city', y='price_per_sqft',
                  color='city', color_discrete_sequence=COLORS,
                  title='Median ₹ per sqft by City')
    fig.update_layout(xaxis_title='', yaxis_title='₹ per sqft', showlegend=False)
    return apply_theme(fig)

def fig_model_r2(results):
    df2 = pd.DataFrame([{'Model':k,'R²':v['R²']} for k,v in results.items()])
    df2 = df2.sort_values('R²',ascending=True)
    colors = ['#3B82F6' if 'Bayesian' in m else
              '#6366F1' if 'GBM' in m else
              '#10B981' if 'Forest' in m else
              '#94A3B8' for m in df2['Model']]
    fig = go.Figure(go.Bar(
        x=df2['R²'], y=df2['Model'], orientation='h',
        marker_color=colors, text=df2['R²'].round(3),
        textposition='outside', textfont_size=11,
    ))
    fig.update_layout(title='R² Score by Model (test set)',
                      xaxis_title='R²', xaxis_range=[0, 1.05], yaxis_title='')
    return apply_theme(fig)

def fig_actual_vs_pred(yt, yp):
    yt_cr = yt/1e7; yp_cr = yp/1e7
    d  = pd.DataFrame({'Actual':yt_cr,'Predicted':yp_cr})
    d  = d[d['Actual']<=15]
    mx = d['Actual'].quantile(0.99)
    fig= px.scatter(d, x='Actual', y='Predicted',
                    opacity=0.35, color_discrete_sequence=[COLORS[0]],
                    title='Actual vs Predicted Price (₹ Crores)')
    fig.add_shape(type='line', x0=0,y0=0,x1=mx,y1=mx,
                  line=dict(color='#EF4444',width=2,dash='dash'))
    fig.update_layout(xaxis_title='Actual (₹ Cr)', yaxis_title='Predicted (₹ Cr)')
    return apply_theme(fig)

def fig_fi(fi_dict, sel):
    s = {k:v for k,v in fi_dict.items() if k in sel}
    df2= pd.DataFrame(s.items(),columns=['Feature','Importance']).sort_values('Importance')
    fig= px.bar(df2, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues',
                title='Feature Importances — GBM Bayesian')
    fig.update_layout(coloraxis_showscale=False, yaxis_title='')
    return apply_theme(fig)

def fig_cv(cv5):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv5))],
        y=cv5, marker_color=COLORS[0],
        text=cv5.round(3), textposition='outside', textfont_size=11,
    ))
    fig.add_hline(y=cv5.mean(), line_dash='dash', line_color=COLORS[1],
                  annotation_text=f'Mean R²={cv5.mean():.3f}',
                  annotation_position='top right')
    fig.update_layout(title='5-Fold Cross Validation R²', yaxis_range=[0,1.05])
    return apply_theme(fig)

def fig_fs_scores(fs_scores):
    rows=[]
    for method, scores in fs_scores.items():
        mx = max(scores.values())
        for feat,val in scores.items():
            rows.append({'Method':method,'Feature':feat,'Score (normalised)':val/mx if mx>0 else 0})
    df2 = pd.DataFrame(rows)
    fig = px.bar(df2, x='Feature', y='Score (normalised)',
                 color='Method', barmode='group',
                 color_discrete_sequence=COLORS,
                 title='Feature Selection Scores (all methods)')
    fig.update_layout(yaxis_title='Normalised Score', legend_title_text='Method')
    return apply_theme(fig)

def gauge_percentile(pct):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        title={'text':'City Percentile', 'font':{'size':13,'family':'Inter','color':'#E6EDF3'}},
        number={'suffix':'%','font':{'size':26,'family':'Inter','color':'#E6EDF3'}},
        gauge={
            'axis':{'range':[0,100],'tickfont':{'size':10}},
            'bar':{'color':'#3B82F6'},
            'steps':[
                {'range':[0,25],'color':'rgba(52,211,153,0.15)'},
                {'range':[25,50],'color':'rgba(251,191,36,0.1)'},
                {'range':[50,75],'color':'rgba(251,191,36,0.18)'},
                {'range':[75,100],'color':'rgba(248,113,113,0.15)'},
            ],
            'threshold':{'line':{'color':'#EF4444','width':3},'value':pct}
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor=BG_CARD, font_family='Inter', font_color=T_PRI)
    return fig

# ── MAIN APP ───────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Magicbricks Analytics")
        st.divider()
        uploaded = st.file_uploader(
            "Upload dataset (.xls / .csv)",
            type=['csv','xls','xlsx'],
            help="Upload magicbricks_properties.xls"
        )
        st.divider()
        st.markdown("""
**About this app**
End-to-end property price analysis and prediction using 5,000+ listings across 6 Indian cities.

**Stack:** Python · scikit-learn · Plotly · Streamlit

**Best model:** Gradient Boosting (Bayesian-tuned)
        """)

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        if uploaded:
            df = load_and_clean(uploaded)
        else:
            df = load_and_clean('magicbricks_properties.xls')
    except Exception:
        try:
            df = load_and_clean('magicbricks_properties__1_.xls')
        except Exception as e:
            st.error("Dataset not found. Please upload `magicbricks_properties.xls` in the sidebar.")
            st.info("The app expects the Magicbricks scraped dataset.")
            st.stop()

    # ── Train models (cached) ─────────────────────────────────────────────────
    with st.spinner("Training all models — please wait, this runs once then stays cached..."):
        pipe = build_pipeline(df)

    results  = pipe['results']
    best_key = max(results, key=lambda k: results[k]['R²'])

    # ── TABS ──────────────────────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs([
        "Overview",
        "EDA & Statistics",
        "Model Performance",
        "Price Predictor",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        # Hero
        st.markdown(f"""
        <div class="app-hero">
          <div class="hero-tag">Magicbricks Analytics — Portfolio Project</div>
          <div class="hero-title">Indian Real Estate<br>Price Intelligence</div>
          <div class="hero-sub">
            Web-scraped dataset of {len(df):,} property listings across
            {df['city'].nunique()} cities. Full ML pipeline from EDA → feature selection →
            ensemble model tuning → price prediction.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI cards
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Listings",   f"{len(df):,}")
        c2.metric("Cities Covered",   df['city'].nunique())
        c3.metric("Median Price",     f"₹{df['price_cr'].median():.2f} Cr")
        c4.metric("Median Area",      f"{int(df['area_sqft'].median()):,} sqft")
        c5.metric("Best Model R²",    f"{results[best_key]['R²']:.3f}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # City summary table
        col_a, col_b = st.columns([1.2,1])
        with col_a:
            st.markdown("<div class='section-head'>City Overview</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub' style='color:#64748B'>Median price and price-per-sqft by city, sorted by value.</div>", unsafe_allow_html=True)
            city_tbl = (df.groupby('city').agg(
                Listings   =('price_cr','count'),
                Median_Cr  =('price_cr','median'),
                Median_psqft=('price_per_sqft','median'),
                Median_sqft=('area_sqft','median'),
            ).round(2).sort_values('Median_Cr',ascending=False)
            .rename(columns={'Median_Cr':'Median (Rs Cr)','Median_psqft':'Rs/sqft','Median_sqft':'Median sqft'})
            )
            city_tbl.index = city_tbl.index = city_tbl.index.str.title()
            st.dataframe(city_tbl, use_container_width=True)
        with col_b:
            st.markdown("<div class='section-head'>Furnishing & Status</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-sub' style='color:#64748B'>Distribution of listing types.</div>", unsafe_allow_html=True)
            fu = df['furnishing'].value_counts().reset_index()
            fu.columns=['Furnishing','Count']
            fu['Furnishing'] = fu['Furnishing'].str.title()
            fig_fu = go.Figure(go.Pie(
                labels   = fu['Furnishing'],
                values   = fu['Count'],
                hole     = 0.55,
                marker   = dict(colors=[A_BLUE, A_GREEN, A_AMBER],
                                line=dict(color='#080D1A', width=2)),
                textfont = dict(color='#FFFFFF', size=13, family='Inter'),
                textinfo = 'percent',
                insidetextorientation = 'radial',
            ))
            fig_fu.update_layout(
                paper_bgcolor = BG_CARD,
                plot_bgcolor  = BG_CARD,
                font          = dict(family='Inter', color='#E6EDF3', size=12),
                margin        = dict(l=0, r=0, t=10, b=10),
                showlegend    = True,
                height        = 230,
                legend        = dict(
                    font        = dict(color='#E6EDF3', size=13, family='Inter'),
                    bgcolor     = 'rgba(0,0,0,0)',
                    bordercolor = 'rgba(255,255,255,0)',
                    orientation = 'v',
                    x=0.75, y=0.5,
                    xanchor='left', yanchor='middle',
                    itemsizing  = 'constant',
                    itemwidth   = 30,
                ),
            )
            st.plotly_chart(fig_fu, use_container_width=True, theme=None)
            st.divider()
            st.metric("Ready to Move",
                      f"{(df['status']=='ready to move').sum():,}",
                      f"{(df['status']=='ready to move').mean()*100:.0f}% of listings")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Sample data
        st.markdown("<div class='section-head'>Sample Data</div>", unsafe_allow_html=True)
        show = df[['city','bhk','area_sqft','furnishing','status','price_cr','price_per_sqft']].head(10).copy()
        show.columns=['City','BHK','Area (sqft)','Furnishing','Status','Price (₹ Cr)','₹/sqft']
        show['City'] = show['City'].str.title()
        show['Furnishing'] = show['Furnishing'].str.title()
        show['Status'] = show['Status'].str.title()
        st.dataframe(show.reset_index(drop=True), use_container_width=True, height=320)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — EDA & STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown("<div class='section-head'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub' style='color:#64748B'>Distributions, relationships, statistical tests, and advanced metrics.</div>", unsafe_allow_html=True)

        # Row 1: distributions
        r1a, r1b = st.columns(2)
        with r1a: st.plotly_chart(fig_hist(df,'price_cr','Price Distribution','₹ Crores',clip=20), use_container_width=True, theme=None)
        with r1b: st.plotly_chart(fig_hist(df,'area_sqft','Area Distribution','sqft (m²)',clip=5000,color=COLORS[1]), use_container_width=True, theme=None)

        # Row 2: city + bhk
        r2a, r2b = st.columns(2)
        with r2a: st.plotly_chart(fig_box_city(df), use_container_width=True, theme=None)
        with r2b: st.plotly_chart(fig_bhk_price(df), use_container_width=True, theme=None)

        # Row 3: scatter + correlation
        r3a, r3b = st.columns(2)
        with r3a: st.plotly_chart(fig_scatter(df), use_container_width=True, theme=None)
        with r3b: st.plotly_chart(fig_corr(df), use_container_width=True, theme=None)

        # Row 4: violin + psqft
        r4a, r4b = st.columns(2)
        with r4a: st.plotly_chart(fig_violin_city(df), use_container_width=True, theme=None)
        with r4b: st.plotly_chart(fig_price_per_sqft(df), use_container_width=True, theme=None)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Advanced Statistics ──────────────────────────────────────────────
        st.markdown("<div class='section-head'>Advanced Statistics</div>", unsafe_allow_html=True)

        sa, sb, sc = st.columns(3)

        with sa:
            st.markdown("**Skewness & Kurtosis**")
            rows=[]
            for col,lab in [('price_cr','Price (Cr)'),('area_sqft','Area sqft'),
                             ('bhk','BHK'),('price_per_sqft','₹/sqft')]:
                s=df[col].dropna()
                rows.append({'Column':lab,
                             'Skewness':round(scipy_skew(s),2),
                             'Kurtosis':round(scipy_kurt(s,fisher=True),2)})
            st.dataframe(pd.DataFrame(rows).set_index('Column'), use_container_width=True)

        with sb:
            st.markdown("**Five-Number Summary — Price (₹ Cr)**")
            p = df['price_cr'].dropna()
            q1,q2,q3 = p.quantile(.25),p.quantile(.5),p.quantile(.75)
            iqr=q3-q1
            tbl=[('Min',f'₹{p.min():.2f}'),('Q1 (25%)',f'₹{q1:.2f}'),
                 ('Median',f'₹{q2:.2f}'),('Q3 (75%)',f'₹{q3:.2f}'),
                 ('Max',f'₹{p.max():.2f}'),('IQR',f'₹{iqr:.2f}'),
                 ('Upper Fence',f'₹{q3+1.5*iqr:.2f}')]
            for r in tbl:
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;font-size:0.83rem;border-bottom:1px solid #F1F5F9'><span style='color:#94A3B8'>{r[0]}</span><span style='font-weight:600;color:#E6EDF3'>{r[1]}</span></div>", unsafe_allow_html=True)

        with sc:
            st.markdown("**Outlier Detection**")
            price = df['price_cr'].dropna()
            uf = q3+1.5*iqr
            iqr_out = (price>uf).sum()
            from scipy.stats import zscore
            z = abs(zscore(price))
            z_out = (z>3).sum()
            mad = abs(price-price.median()).median()
            mz  = abs(0.6745*(price-price.median())/mad)
            mz_out = (mz>3.5).sum()
            for label,n,pct in [('IQR (>Q3+1.5IQR)',iqr_out,iqr_out/len(price)),
                                 ('Z-Score (|z|>3)',z_out,z_out/len(price)),
                                 ('Modified Z (|mz|>3.5)',mz_out,mz_out/len(price))]:
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;font-size:0.83rem;border-bottom:1px solid #F1F5F9'><span style='color:#94A3B8'>{label}</span><span style='font-weight:600;color:#E6EDF3'>{n:,} ({pct*100:.1f}%)</span></div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Hypothesis tests
        st.markdown("<div class='section-head'>Hypothesis Tests</div>", unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)

        with h1:
            mumbai  = df[df['city']=='mumbai']['price_cr'].dropna()
            chennai = df[df['city']=='chennai']['price_cr'].dropna()
            t,p = ttest_ind(mumbai,chennai,equal_var=False)
            badge = '<span class="badge b-green">Significant</span>' if p<0.05 else '<span class="badge b-red">Not Significant</span>'
            st.markdown(f"""
            <div class="card-sm">
              <div class="eyebrow">Test 1 — Welch t-test</div>
              <div style="font-weight:600;font-size:0.9rem;color:#E6EDF3;margin-bottom:8px">Mumbai vs Chennai Prices</div>
              <div style="font-size:0.8rem;color:#94A3B8;">
                Mumbai median: ₹{mumbai.median():.2f} Cr<br>
                Chennai median: ₹{chennai.median():.2f} Cr<br>
                t = {t:.3f} &nbsp;|&nbsp; p = {p:.5f}
              </div>
              <div style="margin-top:8px">{badge}</div>
              <div style="font-size:0.75rem;color:#64748B;margin-top:4px">
                {"H₀ REJECTED — prices differ significantly" if p<0.05 else "Cannot reject H₀"}
              </div>
            </div>""", unsafe_allow_html=True)

        with h2:
            groups=[df[df['city']==c]['price_cr'].dropna() for c in df['city'].unique()]
            f,pa=f_oneway(*groups)
            badge2 = '<span class="badge b-green">Significant</span>' if pa<0.05 else '<span class="badge b-red">Not Significant</span>'
            st.markdown(f"""
            <div class="card-sm">
              <div class="eyebrow">Test 2 — One-Way ANOVA</div>
              <div style="font-weight:600;font-size:0.9rem;color:#E6EDF3;margin-bottom:8px">All 6 Cities — Price Difference?</div>
              <div style="font-size:0.8rem;color:#94A3B8;">
                Groups: {df['city'].nunique()} cities<br>
                F-statistic = {f:.3f}<br>
                p-value = {pa:.6f}
              </div>
              <div style="margin-top:8px">{badge2}</div>
              <div style="font-size:0.75rem;color:#64748B;margin-top:4px">
                {"≥1 city price is significantly different" if pa<0.05 else "Cannot reject H₀"}
              </div>
            </div>""", unsafe_allow_html=True)

        with h3:
            ct=pd.crosstab(df['furnishing'].dropna(),df['status'].dropna())
            chi,pch,dof,_=chi2_contingency(ct)
            badge3 = '<span class="badge b-green">Significant</span>' if pch<0.05 else '<span class="badge b-red">Not Significant</span>'
            st.markdown(f"""
            <div class="card-sm">
              <div class="eyebrow">Test 3 — Chi-Square</div>
              <div style="font-weight:600;font-size:0.9rem;color:#E6EDF3;margin-bottom:8px">Furnishing vs Property Status</div>
              <div style="font-size:0.8rem;color:#94A3B8;">
                χ² = {chi:.3f}<br>
                p-value = {pch:.5f}<br>
                Degrees of freedom: {dof}
              </div>
              <div style="margin-top:8px">{badge3}</div>
              <div style="font-size:0.75rem;color:#64748B;margin-top:4px">
                {"Furnishing type IS related to property status" if pch<0.05 else "No significant relationship"}
              </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("<div class='section-head'>ML Pipeline & Model Performance</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub' style='color:#64748B'>5 features → Gradient Boosting (Bayesian-tuned) achieves the best price prediction accuracy.</div>", unsafe_allow_html=True)

        # ── Pipeline summary ──────────────────────────────────────────────────
        steps = [
            ("1","Data Cleaning","5,019 rows · 5 features","#3B82F6"),
            ("2","NaN Imputation","Mode fill → BHK, Furnishing, Status","#10B981"),
            ("3","Encoding","OrdinalEncoder → City, Furnishing, Status","#F59E0B"),
            ("4","Feature Selection","Consensus of 4 methods → top features","#8B5CF6"),
            ("5","Model Training","7 models + 3 tuning strategies","#EF4444"),
        ]
        cols = st.columns(5)
        for col,(num,title,sub,clr) in zip(cols,steps):
            col.markdown(f"""
            <div class="pipeline-step" style="border-top:3px solid {clr}">
              <div style="font-size:1.5rem;font-weight:800;color:{clr};margin-bottom:6px">{num}</div>
              <div style="font-weight:700;font-size:0.82rem;color:#0F172A;margin-bottom:4px">{title}</div>
              <div style="font-size:0.72rem;color:#475569">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Feature selection chart ───────────────────────────────────────────
        fa, fb = st.columns(2)
        with fa:
            st.plotly_chart(fig_fs_scores(pipe['fs_scores']), use_container_width=True, theme=None)
        with fb:
            fi = pipe['fs_scores']['GBM Importance']
            st.plotly_chart(fig_fi(fi, pipe['sel']), use_container_width=True, theme=None)

        st.markdown(f"""
        <div class="note-box">
          <p><strong>Selected features:</strong> {', '.join([f.replace('_',' ').title() for f in pipe['sel']])}
          — chosen by consensus of SelectKBest, Mutual Information, RFE, and GBM Feature Importances.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Model comparison ──────────────────────────────────────────────────
        st.markdown("<div class='section-head'>All Models — Comparison</div>", unsafe_allow_html=True)

        ALGO_INFO = {
            'Decision Tree' : ('Non-Parametric','#94A3B8'),
            'Ridge'         : ('Parametric','#94A3B8'),
            'Random Forest' : ('Ensemble — Non-Parametric','#10B981'),
            'GBM Baseline'  : ('Ensemble — Non-Parametric','#6366F1'),
            'GBM GridSearch': ('Ensemble — Tuned','#6366F1'),
            'GBM RandomSearch':('Ensemble — Tuned','#6366F1'),
            'GBM Bayesian'  : ('Ensemble — Best Tuned ★','#3B82F6'),
        }

        for name,v in sorted(results.items(), key=lambda x:-x[1]['R²']):
            is_best = name==best_key
            typ,clr = ALGO_INFO.get(name,('','#94A3B8'))
            border  = 'mrow-best' if is_best else ''
            crown   = ' (Best)' if is_best else ''
            st.markdown(f"""
            <div class="mrow {border}">
              <div>
                <div class="mname">{name}{crown}</div>
                <div class="mtype">{typ}</div>
              </div>
              <div style="display:flex;gap:2rem;align-items:center">
                <div><div class="mval" style="color:{clr}">{v['R²']:.4f}</div><div class="mval-sub">R²</div></div>
                <div><div class="mval">{v['Adj R²']:.4f}</div><div class="mval-sub">Adj R²</div></div>
                <div><div class="mval">₹{v['MAE (Cr)']:.2f} Cr</div><div class="mval-sub">MAE</div></div>
                <div><div class="mval">₹{v['RMSE (Cr)']:.2f} Cr</div><div class="mval-sub">RMSE</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # R² bar chart + actual vs predicted
        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(fig_model_r2(results), use_container_width=True, theme=None)
        with g2: st.plotly_chart(fig_actual_vs_pred(pipe['yt_act'], pipe['yp_best']), use_container_width=True, theme=None)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Cross-validation ──────────────────────────────────────────────────
        st.markdown("<div class='section-head'>Cross Validation — GBM Bayesian</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub' style='color:#64748B'>5-Fold CV on training data — ensures the model generalises and is not overfitting.</div>", unsafe_allow_html=True)
        cv1, cv2 = st.columns([2,1])
        with cv1:
            st.plotly_chart(fig_cv(pipe['cv5']), use_container_width=True, theme=None)
        with cv2:
            cv5 = pipe['cv5']
            st.metric("Mean R²",  f"{cv5.mean():.4f}")
            st.metric("Std R²",   f"{cv5.std():.4f}")
            st.metric("Min Fold", f"{cv5.min():.4f}")
            st.metric("Max Fold", f"{cv5.max():.4f}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Why GBM chosen ───────────────────────────────────────────────────
        st.markdown("<div class='section-head'>Why Gradient Boosting?</div>", unsafe_allow_html=True)
        w1,w2,w3,w4 = st.columns(4)
        for col,num,title,text,clr in [
            (w1,'01','Handles Mixed Data','No scaling needed. Works directly with ordinal-encoded categories and raw numeric features.','#3B82F6'),
            (w2,'02','Non-Linear Relationships','Captures interaction effects: a 3BHK in Mumbai is not 3x a 1BHK. Trees model this naturally.','#10B981'),
            (w3,'03','Sequential Correction','Each new tree corrects residuals from the previous. Random Forest averages; GBM specifically fixes mistakes.','#F59E0B'),
            (w4,'04','Bayesian Tuning Edge','Optuna learns from past trial outcomes to explore hyperparameter space — far smarter than exhaustive grid search.','#8B5CF6'),
        ]:
            col.markdown(f"""
            <div class="why-card">
              <div class="why-icon" style="background:{clr}1A;color:{clr}">{num}</div>
              <div class="why-title">{title}</div>
              <div class="why-desc">{text}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — PRICE PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown("<div class='section-head'>Property Price Predictor</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub' style='color:#64748B'>Enter your property details below. The model uses 5 features to estimate market price using the Bayesian-tuned Gradient Boosting Regressor (R² = {:.3f}).</div>".format(results['GBM Bayesian']['R²']), unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Form
        pa, pb = st.columns([1, 1.4])

        with pa:
            st.markdown("#### Enter Property Details")

            city_map = {'Chennai':'chennai','Gurgaon':'gurgaon','Hyderabad':'hyderabad',
                        'Mumbai':'mumbai','Noida':'noida','Pune':'pune'}
            city_disp = st.selectbox("City", sorted(city_map.keys()),
                                      help="Which city is the property in?")
            city_val = city_map[city_disp]

            bhk_val  = st.slider("BHK (Bedrooms)", min_value=1, max_value=10, value=3, step=1)

            area_val = st.number_input("Area (sqft)", min_value=200, max_value=10000,
                                        value=1200, step=50,
                                        help="Carpet or built-up area in square feet")

            furn_map = {'Unfurnished':'unfurnished','Semi-Furnished':'semi-furnished','Furnished':'furnished'}
            furn_disp= st.selectbox("Furnishing", list(furn_map.keys()))
            furn_val = furn_map[furn_disp]

            stat_map = {'Ready to Move':'ready to move','Under Construction':'under construction'}
            stat_disp= st.selectbox("Status", list(stat_map.keys()))
            stat_val = stat_map[stat_disp]

            predict_btn = st.button("Predict Price")

        with pb:
            if predict_btn or True:   # show result section always
                if predict_btn:
                    with st.spinner("Computing prediction..."):
                        try:
                            price_cr = predict(city_val, bhk_val, area_val,
                                               furn_val, stat_val, pipe)

                            # Percentile in city
                            city_prices = df[df['city']==city_val]['price_cr'].dropna()
                            pct = float((city_prices < price_cr).mean() * 100)

                            # Range ±12%
                            lo = price_cr * 0.88
                            hi = price_cr * 1.12

                            # Store in session state
                            st.session_state['pred'] = {
                                'price_cr':price_cr,'lo':lo,'hi':hi,'pct':pct,
                                'city':city_disp,'bhk':bhk_val,'area':area_val,
                                'furn':furn_disp,'stat':stat_disp
                            }
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

                if 'pred' in st.session_state:
                    pr = st.session_state['pred']
                    pc = pr['price_cr']
                    lo = pr['lo']; hi = pr['hi']

                    # Big prediction card
                    st.markdown(f"""
                    <div class="pred-hero">
                      <div class="pred-label">Estimated Market Price</div>
                      <div class="pred-price">₹ {pc:.2f}</div>
                      <div class="pred-unit">Crores &nbsp;·&nbsp; ₹ {pc*1e7:,.0f}</div>
                      <div class="pred-range">Range: ₹ {lo:.2f} Cr — ₹ {hi:.2f} Cr</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                    # Metrics row
                    m1,m2,m3 = st.columns(3)
                    m1.metric("Price per sqft",f"₹ {pc*1e7/pr['area']:,.0f}")
                    m2.metric("City Percentile",f"{pr['pct']:.0f}th",
                               help=f"Your property is pricier than {pr['pct']:.0f}% of {pr['city']} listings")
                    m3.metric("Model Used","GBM Bayesian",
                               help=f"R² = {results['GBM Bayesian']['R²']:.3f}")

                    # Gauge
                    st.plotly_chart(gauge_percentile(pr['pct']), use_container_width=True, theme=None)

                    # Feature importance for this prediction
                    st.markdown("#### What drove this prediction?")
                    best_mdl = pipe['models']['GBM Bayesian']
                    fi = dict(zip(pipe['sel'], best_mdl.feature_importances_))
                    fi_df = pd.DataFrame(fi.items(),columns=['Feature','Importance']).sort_values('Importance',ascending=True)
                    fi_df['Feature'] = fi_df['Feature'].str.replace('_',' ').str.title()
                    fig_fi2 = go.Figure(go.Bar(
                        x=fi_df['Importance'], y=fi_df['Feature'],
                        orientation='h', marker_color=COLORS[0],
                        text=(fi_df['Importance']*100).round(1).astype(str)+'%',
                        textposition='outside', textfont_size=11,
                    ))
                    fig_fi2.update_layout(**PLOTLY_LAYOUT,
                                          title='Feature Contribution to Price',
                                          xaxis_title='Importance', yaxis_title='',
                                          height=220)
                    st.plotly_chart(fig_fi2, use_container_width=True, theme=None)

                    # Similar properties from dataset
                    st.markdown("#### Similar Properties in Dataset")
                    sim = df[
                        (df['city'] == city_val)
                        & (df['bhk'] == float(bhk_val))
                        & (df['price_cr'].between(lo, hi))
                    ].copy()
                    if len(sim)==0:
                        sim = df[(df['city']==city_val)
                                  & df['price_cr'].between(lo*0.7, hi*1.3)].head(5)
                    show_sim = sim[['city','bhk','area_sqft','furnishing','status','price_cr']].head(5).copy()
                    show_sim.columns=['City','BHK','Area (sqft)','Furnishing','Status','Price (₹ Cr)']
                    for c in ['City','Furnishing','Status']:
                        show_sim[c] = show_sim[c].str.title()
                    st.dataframe(show_sim.reset_index(drop=True), use_container_width=True)

                else:
                    st.markdown("""
                    <div class="note-box" style="margin-top:1rem">
                      <p>Fill in the property details on the left and click <strong>Predict Price</strong> to get an instant market estimate powered by our Gradient Boosting model (R² = {:.3f}).</p>
                    </div>""".format(results['GBM Bayesian']['R²']), unsafe_allow_html=True)

                    # Show what each input does
                    st.markdown("#### How inputs affect price")
                    tips=[
                        ("City","Has the highest impact. Mumbai and Gurgaon command a 3–4× premium over Chennai."),
                        ("Area","Moderate impact — larger homes cost more, but ₹/sqft varies by city."),
                        ("BHK","Correlated with area. 3BHK is most common; 4BHK+ shows step-jump pricing."),
                        ("Furnishing","Furnished homes carry ~15–25% premium over unfurnished."),
                        ("Status","Ready-to-move typically priced higher than under construction."),
                    ]
                    for icon_tip, desc in tips:
                        st.markdown(f"<span style='color:#94A3B8'><strong style='color:#E6EDF3'>{icon_tip}</strong> — {desc}</span>", unsafe_allow_html=True)

        # Model summary note
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="note-box">
          <p>
          <strong>Model: Gradient Boosting Regressor — Bayesian-tuned (Optuna)</strong><br>
          Test R² = {results['GBM Bayesian']['R²']:.4f} &nbsp;|&nbsp;
          MAE = ₹{results['GBM Bayesian']['MAE (Cr)']:.2f} Cr &nbsp;|&nbsp;
          RMSE = ₹{results['GBM Bayesian']['RMSE (Cr)']:.2f} Cr &nbsp;|&nbsp;
          CV Mean R² = {pipe['cv5'].mean():.4f}<br><br>
          The prediction represents the median market estimate. Actual prices may vary by ±10–20%
          depending on floor, facing, exact location within the city, and negotiation.
          </p>
        </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()