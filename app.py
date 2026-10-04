import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="CreditSense | Loan Risk Predictor", page_icon="💳", layout="wide")

# ---------- Styling (pure CSS animations: GPU-friendly, no extra Python work) ----------
st.markdown("""
<style>
@keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes fadeUp { from{opacity:0; transform:translateY(24px)} to{opacity:1; transform:translateY(0)} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
@keyframes pulseGlow { 0%,100%{box-shadow:0 0 0 0 rgba(255,255,255,.45)} 70%{box-shadow:0 0 0 22px rgba(255,255,255,0)} }
@keyframes pop { 0%{transform:scale(.6); opacity:0} 70%{transform:scale(1.06)} 100%{transform:scale(1); opacity:1} }
@keyframes blob { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(40px,-30px) scale(1.15)} 66%{transform:translate(-30px,30px) scale(.9)} }
@keyframes grow { from{width:0} }

.stApp { background: linear-gradient(-45deg,#0f0c29,#302b63,#24243e,#41295a,#1f4068);
  background-size:400% 400%; animation:bgShift 18s ease infinite; }
.stApp::before, .stApp::after { content:""; position:fixed; z-index:0; border-radius:50%; pointer-events:none;
  filter:blur(70px); opacity:.35; will-change:transform; }
.stApp::before { width:380px; height:380px; top:-80px; left:-80px; background:#ff6a88; animation:blob 14s ease-in-out infinite; }
.stApp::after { width:420px; height:420px; bottom:-120px; right:-100px; background:#4facfe; animation:blob 18s ease-in-out infinite reverse; }

.hero { position:relative; z-index:1; padding:28px 32px; border-radius:22px; margin-bottom:18px; overflow:hidden;
  background: linear-gradient(90deg,#ff6a88,#ff99ac,#7f7fd5,#86a8e7,#91eae4,#ff6a88); background-size:300% 100%;
  animation: shimmer 8s linear infinite, fadeUp .7s ease both;
  box-shadow:0 10px 40px rgba(127,127,213,.45); }
.hero h1 { margin:0; color:#fff; font-size:2.4rem; text-shadow:0 2px 8px rgba(0,0,0,.25); }
.hero .ico { display:inline-block; animation:float 3s ease-in-out infinite; }
.hero p { margin:6px 0 0; color:#fff; opacity:.95; font-size:1.05rem; }

.card { position:relative; z-index:1; padding:18px 20px; border-radius:18px; color:#fff; margin-bottom:12px;
  background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.2);
  animation:fadeUp .6s ease both; transition:transform .25s, box-shadow .25s; }
.card:hover { transform:translateY(-4px); box-shadow:0 12px 30px rgba(0,0,0,.35); }

.metric { position:relative; z-index:1; text-align:center; padding:16px; border-radius:16px; color:#fff; font-weight:600;
  animation:fadeUp .6s ease both; transition:transform .25s, box-shadow .25s, filter .25s; cursor:default; }
.metric:hover { transform:translateY(-6px) scale(1.04); box-shadow:0 14px 28px rgba(0,0,0,.4); filter:brightness(1.08); }
.metric .v { font-size:1.9rem; display:block; }
.m1 { background:linear-gradient(135deg,#f093fb,#f5576c); animation-delay:.05s; }
.m2 { background:linear-gradient(135deg,#4facfe,#00c6fb); animation-delay:.15s; }
.m3 { background:linear-gradient(135deg,#43e97b,#38c1a8); animation-delay:.25s; }
.m4 { background:linear-gradient(135deg,#fa709a,#fee140); color:#3a2a00; animation-delay:.35s; }

.verdict { position:relative; z-index:1; padding:22px; border-radius:20px; text-align:center; color:#fff;
  font-size:1.5rem; font-weight:700; animation:pop .6s cubic-bezier(.2,.9,.3,1.2) both, pulseGlow 2.2s ease-out 0.6s infinite;
  box-shadow:0 8px 30px rgba(0,0,0,.35); }
.safe { background:linear-gradient(135deg,#11998e,#38ef7d); }
.risky { background:linear-gradient(135deg,#f85032,#e73827); }

.bar { height:14px; border-radius:99px; background:rgba(255,255,255,.15); overflow:hidden; margin-top:8px; }
.bar > div { height:100%; border-radius:99px; background:linear-gradient(90deg,#38ef7d,#fee140,#f5576c);
  background-size:200% 100%; animation:grow 1.2s cubic-bezier(.2,.8,.2,1) both, shimmer 3s linear infinite; }

.stPlotlyChart { animation:fadeUp .8s ease both; }
label, .stMarkdown, h2, h3 { color:#fff !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#1f1c4a,#3a2f7a); }
section[data-testid="stSidebar"] * { color:#fff !important; }

div.stButton > button { position:relative; overflow:hidden; width:100%; border:0; border-radius:14px; padding:.8rem;
  font-size:1.1rem; font-weight:700; color:#fff;
  background:linear-gradient(90deg,#ff6a88,#7f7fd5,#91eae4,#ff6a88); background-size:300% 100%;
  animation:shimmer 6s linear infinite; transition:transform .2s, box-shadow .2s; }
div.stButton > button:hover { transform:translateY(-3px) scale(1.02); box-shadow:0 10px 25px rgba(127,127,213,.6); }
div.stButton > button:active { transform:scale(.97); }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_risk_model.pkl")
    thr = float(joblib.load("threshold.pkl"))
    return model, thr

model, THRESHOLD = load_artifacts()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1><span class="ico">💳</span> CreditSense</h1>
  <p>AI-powered loan default risk predictor — enter applicant details and get an instant decision.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("🧾 Applicant Profile")
    age = st.slider("Age", 18, 100, 28)
    income = st.number_input("Annual income", min_value=1000, value=55000, step=1000)
    home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    emp = st.slider("Employment length (years)", 0, 50, 4)
    cred_hist = st.slider("Credit history length (years)", 0, 40, 5)
    default_file = st.radio("Previous default on file?", ["N", "Y"], horizontal=True,
                            format_func=lambda x: "No" if x == "N" else "Yes")

    st.header("🏦 Loan Details")
    intent = st.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE",
                                           "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    grade = st.select_slider("Loan grade", options=list("ABCDEFG"), value="B")
    amount = st.number_input("Loan amount", min_value=500, value=10000, step=500)
    rate = st.slider("Interest rate (%)", 5.0, 25.0, 11.0, 0.1)

    predict = st.button("🔮 Predict Risk")

pct_income = amount / income
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric m1">Loan Amount<span class="v">{amount:,.0f}</span></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric m2">Income<span class="v">{income:,.0f}</span></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric m3">Loan / Income<span class="v">{pct_income:.0%}</span></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric m4">Grade<span class="v">{grade}</span></div>', unsafe_allow_html=True)
st.write("")

def gauge(prob, thr):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"color": "white", "size": 48}},
        title={"text": "Default Probability", "font": {"color": "white", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ffffff", "thickness": 0.25},
            "steps": [
                {"range": [0, thr * 50], "color": "#38ef7d"},
                {"range": [thr * 50, thr * 100], "color": "#fee140"},
                {"range": [thr * 100, 100], "color": "#f5576c"},
            ],
            "threshold": {"line": {"color": "white", "width": 5}, "thickness": 0.85, "value": thr * 100},
        }))
    fig.update_layout(transition={"duration": 900, "easing": "cubic-in-out"}, height=340, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=10, l=30, r=30))
    return fig

if predict:
    row = pd.DataFrame([{
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
        "loan_amnt": amount, "loan_int_rate": rate, "loan_percent_income": pct_income,
        "cb_person_default_on_file": default_file, "cb_person_cred_hist_length": cred_hist,
    }])
    prob = float(model.predict_proba(row)[0, 1])
    is_risky = prob >= THRESHOLD

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(gauge(prob, THRESHOLD), width="stretch")
    with right:
        if is_risky:
            st.markdown('<div class="verdict risky">⚠️ HIGH RISK<br><small>Likely to default — review / reject</small></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict safe">✅ LOW RISK<br><small>Likely to repay — eligible for approval</small></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="card">
        <b>Default probability:</b> {prob:.1%}<br>
        <b>Decision threshold:</b> {THRESHOLD:.1%}<br>
        <b>Margin:</b> {abs(prob-THRESHOLD):.1%} {'above' if is_risky else 'below'} threshold
        <div class="bar"><div style="width:{prob*100:.1f}%"></div></div>
        </div>""", unsafe_allow_html=True)
    if not is_risky:
        st.balloons()
    with st.expander("📋 Input summary"):
        st.dataframe(row.astype(str).T.rename(columns={0: "value"}), width="stretch")
else:
    st.markdown('<div class="card">👈 Fill in the applicant and loan details in the sidebar, then click <b>Predict Risk</b>.</div>',
                unsafe_allow_html=True)

st.caption("Model: calibrated XGBoost pipeline · Decision rule: probability ≥ saved threshold → high risk")
