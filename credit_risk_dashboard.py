"""
╔══════════════════════════════════════════════════════════════╗
║      CREDIT RISK MANAGEMENT DASHBOARD — Streamlit App        ║
║      Senior Data Scientist & Financial Analyst Build         ║
╚══════════════════════════════════════════════════════════════╝

Run with:  streamlit run credit_risk_dashboard.py
Requires:  pip install streamlit pandas numpy plotly scipy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-amber: #f59e0b;
    --risk-low:     #10b981;
    --risk-med:     #f59e0b;
    --risk-high:    #ef4444;
    --risk-crit:    #dc2626;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e293b;
  }

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent-cyan) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .kpi-card.blue::before  { background: var(--accent-blue); }
  .kpi-card.cyan::before  { background: var(--accent-cyan); }
  .kpi-card.amber::before { background: var(--accent-amber); }
  .kpi-card.red::before   { background: var(--risk-high); }
  .kpi-card.green::before { background: var(--risk-low); }

  .kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
  }
  .kpi-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
  }
  .kpi-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin-top: 0.35rem; }
  .kpi-delta.up   { color: var(--risk-high); }
  .kpi-delta.down { color: var(--risk-low); }

  .badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:4px;
            font-size:0.7rem; font-family:'IBM Plex Mono',monospace; font-weight:600;
            letter-spacing:0.08em; text-transform:uppercase; }
  .badge-low      { background:#064e3b; color:#6ee7b7; border:1px solid #065f46; }
  .badge-medium   { background:#451a03; color:#fcd34d; border:1px solid #78350f; }
  .badge-high     { background:#450a0a; color:#fca5a5; border:1px solid #7f1d1d; }
  .badge-critical { background:#3b0a0a; color:#f87171; border:1px solid #991b1b; }

  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  .stTabs [data-baseweb="tab-list"] { background:transparent; border-bottom:1px solid var(--border); gap:0; }
  .stTabs [data-baseweb="tab"] {
    background:transparent; color:var(--text-muted);
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
    letter-spacing:0.1em; text-transform:uppercase;
    padding:0.6rem 1.2rem; border:none;
  }
  .stTabs [aria-selected="true"] {
    background:var(--bg-card) !important;
    color:var(--accent-cyan) !important;
    border-bottom:2px solid var(--accent-cyan) !important;
  }
  .block-container { padding: 1.5rem 2rem; }
  h1 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_portfolio(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    loan_purposes = ["Mortgage","Auto","Personal","Business","Education","Credit Card"]
    loan_grades   = ["A","B","C","D","E","F"]
    sectors       = ["Technology","Healthcare","Manufacturing","Retail","Finance","Real Estate","Energy"]
    regions       = ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret","Thika","Nyeri"]
    emp_lengths   = ["<1 yr","1-2 yrs","3-5 yrs","6-10 yrs","10+ yrs"]

    grade_pd   = {"A":0.01,"B":0.03,"C":0.07,"D":0.13,"E":0.20,"F":0.30}
    grade_lgd  = {"A":0.25,"B":0.30,"C":0.40,"D":0.50,"E":0.60,"F":0.70}

    grades   = rng.choice(loan_grades, n, p=[0.20,0.25,0.22,0.15,0.10,0.08])
    purposes = rng.choice(loan_purposes, n)
    sectors_ = rng.choice(sectors, n)
    regions_ = rng.choice(regions, n)

    loan_amounts = rng.lognormal(11.5, 0.8, n).clip(5_000, 2_000_000).round(-2)
    credit_scores = rng.normal(650, 80, n).clip(300, 850).astype(int)
    annual_income = rng.lognormal(11.3, 0.6, n).clip(20_000, 1_200_000).round(-3)
    dti  = (loan_amounts / annual_income * rng.uniform(0.8, 1.4, n)).clip(0.05, 1.5).round(3)
    lti  = (loan_amounts / annual_income).round(3)

    pd_vals  = np.array([grade_pd[g]  for g in grades]) * rng.uniform(0.6, 1.4, n)
    lgd_vals = np.array([grade_lgd[g] for g in grades]) * rng.uniform(0.8, 1.2, n)
    ead_vals = loan_amounts * rng.uniform(0.7, 1.0, n)
    el_vals  = pd_vals * lgd_vals * ead_vals

    is_default = rng.binomial(1, pd_vals.clip(0, 0.9))

    dates_start = pd.date_range("2020-01-01","2024-06-01", periods=n)
    dates_idx   = rng.integers(0, len(dates_start), n)
    dates_sel   = dates_start[dates_idx]

    df = pd.DataFrame({
        "loan_id":        [f"LN{i:06d}" for i in range(n)],
        "grade":          grades,
        "purpose":        purposes,
        "sector":         sectors_,
        "region":         regions_,
        "loan_amount":    loan_amounts,
        "ead":            ead_vals.round(2),
        "pd":             pd_vals.clip(0, 0.95).round(4),
        "lgd":            lgd_vals.clip(0.1, 0.9).round(4),
        "el":             el_vals.round(2),
        "credit_score":   credit_scores,
        "annual_income":  annual_income,
        "dti":            dti,
        "lti":            lti,
        "emp_length":     rng.choice(emp_lengths, n),
        "is_default":     is_default,
        "origination_date": pd.to_datetime(dates_sel),
        "status":         np.where(is_default == 1,
                                   rng.choice(["Late 60","Late 90","Default"], n),
                                   rng.choice(["Current","Paid Off","Late 30"], n,
                                              p=[0.72, 0.20, 0.08])),
    })
    df["year_month"] = df["origination_date"].dt.to_period("M").astype(str)
    df["risk_band"] = pd.cut(df["pd"], bins=[0,0.05,0.10,0.20,1.0],
                              labels=["Low","Medium","High","Critical"])
    return df


@st.cache_data
def generate_time_series(seed=42):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01", "2024-07", freq="MS")
    base_dr = 0.045
    trend   = np.linspace(0, 0.012, len(months))
    noise   = rng.normal(0, 0.003, len(months))
    default_rates = (base_dr + trend + noise).clip(0.01, 0.15)
    return pd.DataFrame({
        "month":           months,
        "default_rate":    default_rates,
        "new_originations":rng.integers(40, 120, len(months)),
        "total_exposure":  rng.lognormal(19, 0.15, len(months)),
        "provisions":      rng.lognormal(17, 0.2,  len(months)),
    })


df = generate_portfolio()
ts = generate_time_series()

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = {"A":"#10b981","B":"#3b82f6","C":"#8b5cf6","D":"#f59e0b","E":"#ef4444","F":"#dc2626"}
RISK_COLORS = {"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444","Critical":"#dc2626"}


def kpi_card(label, value, delta=None, delta_dir="up", color="blue"):
    delta_html = ""
    if delta:
        cls  = "up" if delta_dir == "up" else "down"
        icon = "▲" if delta_dir == "up" else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{icon} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk\nManagement")
    st.markdown("---")
    st.markdown("## Filters")

    date_min = df["origination_date"].min().date()
    date_max = df["origination_date"].max().date()
    date_range = st.date_input("📅 Origination Date Range",
                               value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    sel_grades  = st.multiselect("📊 Loan Grade",   sorted(df["grade"].unique()),   default=sorted(df["grade"].unique()))
    sel_purpose = st.multiselect("🎯 Loan Purpose", sorted(df["purpose"].unique()), default=sorted(df["purpose"].unique()))
    sel_sector  = st.multiselect("🏭 Sector",       sorted(df["sector"].unique()),  default=sorted(df["sector"].unique()))
    sel_region  = st.multiselect("🗺️ Region",       sorted(df["region"].unique()),  default=sorted(df["region"].unique()))

    st.markdown("---")
    pd_threshold = st.slider("⚠️ PD Alert Threshold", 0.0, 1.0, 0.15, 0.01, format="%.2f")
    st.markdown("---")
    st.caption("Data: Synthetic Portfolio · 2,000 loans")


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = df["origination_date"].min(), df["origination_date"].max()

fdf = df[
    df["origination_date"].between(d0, d1) &
    df["grade"].isin(sel_grades) &
    df["purpose"].isin(sel_purpose) &
    df["sector"].isin(sel_sector) &
    df["region"].isin(sel_region)
].copy()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# Credit Risk Management Dashboard")
    st.caption(f"Portfolio snapshot · {len(fdf):,} loans · filtered view")
with col_h2:
    alerts = (fdf["pd"] > pd_threshold).sum()
    st.markdown(f"""
    <div class="kpi-card red" style="text-align:center;">
      <div class="kpi-label">High-Risk Alerts</div>
      <div class="kpi-value" style="color:#ef4444;">{alerts:,}</div>
      <div class="kpi-delta up">PD &gt; {pd_threshold:.0%}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PRIMARY KPIs
# ─────────────────────────────────────────────
# ── KPI FORMULAS (Pandas equivalents of DAX) ──
# Weighted Average PD  = SUMPRODUCT(EAD, PD) / SUM(EAD)
# Portfolio EAD        = SUM(EAD)
# Expected Loss        = SUM(PD * LGD * EAD)
# CDR                  = COUNT(Defaults) / COUNT(Total)
# Weighted Avg LGD     = SUMPRODUCT(EAD, LGD) / SUM(EAD)
# ──────────────────────────────────────────────

st.markdown('<div class="section-header">Primary KPIs — Portfolio Summary</div>', unsafe_allow_html=True)

total_ead = fdf["ead"].sum()
wa_pd     = (fdf["pd"]  * fdf["ead"]).sum() / total_ead if total_ead > 0 else 0
wa_lgd    = (fdf["lgd"] * fdf["ead"]).sum() / total_ead if total_ead > 0 else 0
exp_loss  = fdf["el"].sum()
cdr       = fdf["is_default"].mean()
avg_cs    = fdf["credit_score"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: kpi_card("Weighted Avg PD",     f"{wa_pd:.2%}",          "+0.31% MoM", "up",   "red")
with k2: kpi_card("Weighted Avg LGD",    f"{wa_lgd:.2%}",         "+0.8% QoQ",  "up",   "amber")
with k3: kpi_card("Total EAD",           f"${total_ead/1e6:.1f}M","-2.1% MoM",  "down", "blue")
with k4: kpi_card("Expected Loss",       f"${exp_loss/1e6:.1f}M", "+4.2% MoM",  "up",   "red")
with k5: kpi_card("Credit Default Rate", f"{cdr:.2%}",            "+0.18% MoM", "up",   "amber")
with k6: kpi_card("Avg Credit Score",    f"{avg_cs:.0f}",         "+3 pts MoM", "down", "green")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📈  Trend Analysis",
    "📊  Risk Distribution",
    "🗺️  Heatmaps",
    "🍩  Portfolio Mix",
    "⚠️  Risk Matrix",
    "🔍  Borrower Drill-Down",
])


# ══ TAB 1 — TREND ANALYSIS ═══════════════════
with tabs[0]:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-header">Default Rate Trend (Monthly)</div>', unsafe_allow_html=True)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts["month"], y=ts["default_rate"], mode="lines", name="Default Rate",
            line=dict(color="#ef4444", width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts["month"], y=ts["default_rate"].rolling(3, center=True).mean(),
            mode="lines", name="3M Avg",
            line=dict(color="#f59e0b", width=1.5, dash="dot")
        ))
        fig_ts.add_hline(y=pd_threshold, line_color="#ef4444", line_dash="dash",
                         line_width=1, annotation_text=f"Threshold {pd_threshold:.0%}",
                         annotation_font_color="#ef4444")
        fig_ts.update_layout(**DARK, height=320, yaxis_tickformat=".1%",
                             legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Originations vs Provisions</div>', unsafe_allow_html=True)
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bar.add_trace(go.Bar(
            x=ts["month"], y=ts["new_originations"],
            name="Originations", marker_color="#3b82f6", opacity=0.8
        ), secondary_y=False)
        fig_bar.add_trace(go.Scatter(
            x=ts["month"], y=ts["provisions"], name="Provisions ($)",
            mode="lines+markers", line=dict(color="#f59e0b", width=2),
            marker=dict(size=4)
        ), secondary_y=True)
        fig_bar.update_layout(**DARK, height=320, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">Default Rate by Grade Over Time</div>', unsafe_allow_html=True)
    grade_monthly = (fdf.groupby(["year_month","grade"])["is_default"]
                       .mean().reset_index().rename(columns={"is_default":"default_rate"}))
    fig_gt = px.line(grade_monthly, x="year_month", y="default_rate",
                     color="grade", color_discrete_map=COLORS,
                     markers=False, height=300)
    fig_gt.update_layout(**DARK, yaxis_tickformat=".1%", xaxis_tickangle=-45,
                         legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_gt, use_container_width=True)


# ══ TAB 2 — RISK DISTRIBUTION ═══════════════
with tabs[1]:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Stacked Loan Count — Grade × Risk Band</div>', unsafe_allow_html=True)
        dist = fdf.groupby(["grade","risk_band"]).size().reset_index(name="count")
        fig_stack = px.bar(dist, x="grade", y="count", color="risk_band",
                           color_discrete_map=RISK_COLORS, barmode="stack", height=340)
        fig_stack.update_layout(**DARK, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_stack, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Expected Loss vs EAD by Grade</div>', unsafe_allow_html=True)
        el_grade = (fdf.groupby("grade")
                      .agg(total_el=("el","sum"), total_ead=("ead","sum")).reset_index())
        fig_el = go.Figure()
        fig_el.add_trace(go.Bar(x=el_grade["grade"], y=el_grade["total_ead"],
                                name="EAD", marker_color="#1e3a5f", opacity=0.9))
        fig_el.add_trace(go.Bar(x=el_grade["grade"], y=el_grade["total_el"],
                                name="Expected Loss", marker_color="#ef4444", opacity=0.85))
        fig_el.update_layout(**DARK, barmode="overlay", height=340,
                             yaxis_tickformat="$,.0f",
                             legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_el, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">DTI Distribution by Grade</div>', unsafe_allow_html=True)
        fig_dti = px.box(fdf, x="grade", y="dti", color="grade",
                         color_discrete_map=COLORS, height=300, points=False)
        fig_dti.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig_dti, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Credit Score vs PD</div>', unsafe_allow_html=True)
        sample = fdf.sample(min(500, len(fdf)), random_state=1)
        fig_sc = px.scatter(sample, x="credit_score", y="pd",
                            color="risk_band", color_discrete_map=RISK_COLORS,
                            opacity=0.6, height=300, trendline="lowess")
        fig_sc.update_layout(**DARK, yaxis_tickformat=".1%",
                             legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_sc, use_container_width=True)


# ══ TAB 3 — HEATMAPS ════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Average PD Heatmap — Sector × Region</div>', unsafe_allow_html=True)
    heat_data = (fdf.groupby(["sector","region"])["pd"].mean().reset_index()
                   .pivot(index="sector", columns="region", values="pd"))
    fig_heat = px.imshow(heat_data,
                         color_continuous_scale=["#064e3b","#fbbf24","#ef4444"],
                         aspect="auto", height=380, text_auto=".1%", zmin=0, zmax=0.25)
    fig_heat.update_layout(**DARK, coloraxis_colorbar=dict(tickformat=".0%"))
    fig_heat.update_traces(textfont_size=10)
    st.plotly_chart(fig_heat, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Expected Loss — Sector × Grade</div>', unsafe_allow_html=True)
        heat2 = (fdf.groupby(["sector","grade"])["el"].sum().reset_index()
                   .pivot(index="sector", columns="grade", values="el"))
        fig_h2 = px.imshow(heat2,
                           color_continuous_scale=["#0f172a","#3b82f6","#ef4444"],
                           aspect="auto", height=320, text_auto="$,.0f")
        fig_h2.update_layout(**DARK)
        fig_h2.update_traces(textfont_size=9)
        st.plotly_chart(fig_h2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Default Rate — Purpose × Region</div>', unsafe_allow_html=True)
        heat3 = (fdf.groupby(["purpose","region"])["is_default"].mean().reset_index()
                   .pivot(index="purpose", columns="region", values="is_default"))
        fig_h3 = px.imshow(heat3,
                           color_continuous_scale=["#064e3b","#f59e0b","#dc2626"],
                           aspect="auto", height=320, text_auto=".1%")
        fig_h3.update_layout(**DARK, coloraxis_colorbar=dict(tickformat=".0%"))
        fig_h3.update_traces(textfont_size=9)
        st.plotly_chart(fig_h3, use_container_width=True)


# ══ TAB 4 — PORTFOLIO MIX ═══════════════════
with tabs[3]:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">EAD by Loan Grade</div>', unsafe_allow_html=True)
        grade_ead = fdf.groupby("grade")["ead"].sum().reset_index()
        fig_d1 = px.pie(grade_ead, names="grade", values="ead", hole=0.55,
                        color="grade", color_discrete_map=COLORS, height=300)
        fig_d1.update_traces(textposition="outside", textinfo="label+percent", textfont_size=10)
        fig_d1.update_layout(**DARK, showlegend=False,
                             annotations=[dict(text="By Grade", x=0.5, y=0.5,
                                              font_size=11, showarrow=False, font_color="#64748b")])
        st.plotly_chart(fig_d1, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">EAD by Loan Purpose</div>', unsafe_allow_html=True)
        purpose_ead = fdf.groupby("purpose")["ead"].sum().reset_index()
        fig_d2 = px.pie(purpose_ead, names="purpose", values="ead", hole=0.55,
                        color_discrete_sequence=px.colors.qualitative.Set3, height=300)
        fig_d2.update_traces(textposition="outside", textinfo="label+percent", textfont_size=9)
        fig_d2.update_layout(**DARK, showlegend=False,
                             annotations=[dict(text="By Purpose", x=0.5, y=0.5,
                                              font_size=11, showarrow=False, font_color="#64748b")])
        st.plotly_chart(fig_d2, use_container_width=True)

    with c3:
        st.markdown('<div class="section-header">EAD by Risk Band</div>', unsafe_allow_html=True)
        risk_ead = fdf.groupby("risk_band")["ead"].sum().reset_index()
        fig_d3 = px.pie(risk_ead, names="risk_band", values="ead", hole=0.55,
                        color="risk_band", color_discrete_map=RISK_COLORS, height=300)
        fig_d3.update_traces(textposition="outside", textinfo="label+percent", textfont_size=10)
        fig_d3.update_layout(**DARK, showlegend=False,
                             annotations=[dict(text="By Risk", x=0.5, y=0.5,
                                              font_size=11, showarrow=False, font_color="#64748b")])
        st.plotly_chart(fig_d3, use_container_width=True)

    st.markdown('<div class="section-header">Top Sector Exposure Concentration</div>', unsafe_allow_html=True)
    sector_conc = (fdf.groupby("sector")
                     .agg(total_ead=("ead","sum"), avg_pd=("pd","mean"))
                     .sort_values("total_ead", ascending=True).reset_index())
    sector_conc["pct_portfolio"] = sector_conc["total_ead"] / total_ead
    fig_conc = go.Figure()
    fig_conc.add_trace(go.Bar(
        y=sector_conc["sector"], x=sector_conc["total_ead"],
        orientation="h",
        marker=dict(color=sector_conc["avg_pd"],
                    colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                    showscale=True,
                    colorbar=dict(title="Avg PD", tickformat=".0%", len=0.6)),
        text=[f"${v/1e6:.1f}M · {p:.1%}" for v, p in
              zip(sector_conc["total_ead"], sector_conc["pct_portfolio"])],
        textposition="outside"
    ))
    fig_conc.update_layout(**DARK, height=280, xaxis_tickformat="$,.0f", showlegend=False)
    st.plotly_chart(fig_conc, use_container_width=True)


# ══ TAB 5 — RISK MATRIX ═════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Dynamic Risk Matrix — PD vs LGD (Bubble=EAD, Color=Expected Loss)</div>', unsafe_allow_html=True)

    matrix_data = (fdf.groupby(["grade","sector"])
                     .agg(avg_pd=("pd","mean"), avg_lgd=("lgd","mean"),
                          total_ead=("ead","sum"), total_el=("el","sum"),
                          loan_count=("loan_id","count"))
                     .reset_index())

    fig_matrix = px.scatter(
        matrix_data, x="avg_pd", y="avg_lgd",
        size="total_ead", color="total_el",
        color_continuous_scale=["#10b981","#f59e0b","#dc2626"],
        hover_data={"grade":True,"sector":True,
                    "avg_pd":":.2%","avg_lgd":":.2%",
                    "total_ead":"$:,.0f","loan_count":True},
        text="grade", size_max=60, height=480,
        labels={"avg_pd":"Probability of Default","avg_lgd":"Loss Given Default","total_el":"Expected Loss ($)"}
    )
    fig_matrix.add_hline(y=0.45, line_color="#374151", line_dash="dot", line_width=1)
    fig_matrix.add_vline(x=0.12, line_color="#374151", line_dash="dot", line_width=1)
    fig_matrix.update_traces(textposition="top center", textfont=dict(size=9, color="white"))
    fig_matrix.update_layout(**DARK, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                             coloraxis_colorbar=dict(title="EL ($)", tickformat="$,.0f"))
    for txt, x, y in [("◀ LOW RISK",0.03,0.15),("HIGH PD ▶",0.22,0.15),
                       ("HIGH LGD ▲",0.03,0.75),("⚠ CRITICAL",0.22,0.75)]:
        fig_matrix.add_annotation(x=x, y=y, text=txt, showarrow=False,
                                  font=dict(size=9, color="#475569"), xanchor="center")
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown('<div class="section-header">Risk Band Summary — Color-Coded Status Tags</div>', unsafe_allow_html=True)

    def status_badge(rb):
        m = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high","Critical":"badge-critical"}
        return f'<span class="badge {m.get(rb,"")}">{rb}</span>'

    summary = (fdf.groupby("risk_band")
                 .agg(loans=("loan_id","count"), total_ead=("ead","sum"),
                      avg_pd=("pd","mean"), avg_lgd=("lgd","mean"),
                      total_el=("el","sum"), defaults=("is_default","sum"))
                 .reset_index())
    summary["default_rate"] = summary["defaults"] / summary["loans"]

    for _, row in summary.iterrows():
        cols = st.columns([1.5,1.2,1.5,1.2,1.2,1.5,1.5])
        cols[0].markdown(status_badge(str(row["risk_band"])), unsafe_allow_html=True)
        cols[1].metric("Loans",       f"{row['loans']:,}")
        cols[2].metric("EAD",         f"${row['total_ead']/1e6:.1f}M")
        cols[3].metric("Avg PD",      f"{row['avg_pd']:.2%}")
        cols[4].metric("Avg LGD",     f"{row['avg_lgd']:.2%}")
        cols[5].metric("Exp. Loss",   f"${row['total_el']/1e6:.1f}M")
        cols[6].metric("Default Rate",f"{row['default_rate']:.2%}")


# ══ TAB 6 — BORROWER DRILL-DOWN ══════════════
with tabs[5]:
    st.markdown('<div class="section-header">Individual Borrower Analysis — Drill-Down</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: search_id     = st.text_input("🔍 Search Loan ID", placeholder="e.g. LN000042")
    with col_s2: filter_grade  = st.selectbox("Grade",    ["All"] + sorted(df["grade"].unique().tolist()))
    with col_s3: filter_risk   = st.selectbox("Risk Band",["All"] + ["Low","Medium","High","Critical"])
    with col_s4: filter_status = st.selectbox("Status",   ["All"] + sorted(df["status"].unique().tolist()))

    drilldf = fdf.copy()
    if search_id:          drilldf = drilldf[drilldf["loan_id"].str.contains(search_id, case=False)]
    if filter_grade  != "All": drilldf = drilldf[drilldf["grade"]     == filter_grade]
    if filter_risk   != "All": drilldf = drilldf[drilldf["risk_band"].astype(str) == filter_risk]
    if filter_status != "All": drilldf = drilldf[drilldf["status"]    == filter_status]

    drilldf_display = drilldf[[
        "loan_id","grade","purpose","sector","region",
        "loan_amount","ead","pd","lgd","el",
        "credit_score","dti","lti","emp_length","status","risk_band"
    ]].copy()

    drilldf_display["pd"]          = drilldf_display["pd"].map(lambda x: f"{x:.2%}")
    drilldf_display["lgd"]         = drilldf_display["lgd"].map(lambda x: f"{x:.2%}")
    drilldf_display["dti"]         = drilldf_display["dti"].map(lambda x: f"{x:.2f}")
    drilldf_display["lti"]         = drilldf_display["lti"].map(lambda x: f"{x:.2f}")
    drilldf_display["loan_amount"] = drilldf_display["loan_amount"].map(lambda x: f"${x:,.0f}")
    drilldf_display["ead"]         = drilldf_display["ead"].map(lambda x: f"${x:,.0f}")
    drilldf_display["el"]          = drilldf_display["el"].map(lambda x: f"${x:,.0f}")

    st.dataframe(drilldf_display.head(200), use_container_width=True, height=320,
                 column_config={
                     "loan_id":      st.column_config.TextColumn("Loan ID"),
                     "credit_score": st.column_config.ProgressColumn(
                                        "Credit Score", min_value=300, max_value=850, format="%d"),
                     "status":       st.column_config.TextColumn("Status"),
                     "risk_band":    st.column_config.TextColumn("Risk Band"),
                 })
    st.caption(f"Showing {min(200, len(drilldf)):,} of {len(drilldf):,} matching loans")

    st.markdown("---")
    st.markdown('<div class="section-header">Deep Dive — Single Loan Profile</div>', unsafe_allow_html=True)

    if len(drilldf) > 0:
        selected_id = st.selectbox("Select Loan for Full Profile", drilldf["loan_id"].head(50).tolist())
        row = drilldf[drilldf["loan_id"] == selected_id].iloc[0]

        p1, p2, p3, p4 = st.columns(4)
        rb = str(row["risk_band"])
        color_map = {"Low":"green","Medium":"amber","High":"red","Critical":"red"}
        with p1: kpi_card("Probability of Default", f"{row['pd']:.2%}",    color=color_map.get(rb,"blue"))
        with p2: kpi_card("Loss Given Default",      f"{row['lgd']:.2%}",   color="amber")
        with p3: kpi_card("Exposure at Default",     f"${row['ead']:,.0f}", color="blue")
        with p4: kpi_card("Expected Loss",           f"${row['el']:,.0f}",  color=color_map.get(rb,"blue"))

        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)

        with d1:
            st.markdown("**Borrower Profile**")
            for k, v in {"Loan ID":row["loan_id"],"Grade":row["grade"],"Purpose":row["purpose"],
                         "Sector":row["sector"],"Region":row["region"],
                         "Employment":row["emp_length"],"Status":row["status"],
                         "Risk Band":str(row["risk_band"])}.items():
                st.markdown(f"`{k}:` **{v}**")

        with d2:
            st.markdown("**Financial Metrics**")
            for k, v in {"Loan Amount":f"${row['loan_amount']:,.0f}",
                         "Annual Income":f"${row['annual_income']:,.0f}",
                         "DTI Ratio":f"{row['dti']:.2f}","LTI Ratio":f"{row['lti']:.2f}",
                         "Credit Score":f"{row['credit_score']}",
                         "Origination":str(row["origination_date"].date())}.items():
                st.markdown(f"`{k}:` **{v}**")

        with d3:
            st.markdown("**Risk Gauge**")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(row["pd"]) * 100,
                title=dict(text="PD (%)", font=dict(size=12, color="#94a3b8")),
                number=dict(suffix="%", font=dict(size=22, color="#f1f5f9")),
                gauge=dict(
                    axis=dict(range=[0,40], tickcolor="#374151", tickfont=dict(color="#64748b")),
                    bar=dict(color="#ef4444"),
                    bgcolor="#111827", bordercolor="#1e293b",
                    steps=[dict(range=[0,5],color="#064e3b"),dict(range=[5,10],color="#065f46"),
                           dict(range=[10,20],color="#78350f"),dict(range=[20,40],color="#7f1d1d")],
                    threshold=dict(line=dict(color="#fbbf24",width=2),
                                   thickness=0.75, value=pd_threshold*100),
                )
            ))
            fig_gauge.update_layout(**DARK, height=220, margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("No loans match current filters.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; opacity:0.4;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; letter-spacing:0.1em;">
    CREDIT RISK MANAGEMENT SYSTEM · PORTFOLIO ANALYTICS ENGINE v2.0
  </span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;">
    Data: Synthetic · For Demonstration Purposes Only
  </span>
</div>
""", unsafe_allow_html=True)
