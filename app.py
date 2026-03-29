import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import textstat
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
from nltk.corpus import stopwords
from io import BytesIO

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatGPT Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       rgba(255,255,255,0.04);
    --bg-card-hover: rgba(255,255,255,0.07);
    --border:        rgba(255,255,255,0.08);
    --accent-blue:   #4f8ef7;
    --accent-purple: #a855f7;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-orange: #f97316;
    --accent-pink:   #ec4899;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --glow-blue:     0 0 40px rgba(79,142,247,0.15);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp { background: var(--bg-primary) !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.8;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: var(--glow-blue);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.4rem 0;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.metric-delta {
    font-size: 0.82rem;
    margin-top: 0.3rem;
    color: var(--text-secondary);
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.section-badge {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    color: white;
}

/* ── Chart wrapper ── */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}
.chart-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(168,85,247,0.06));
    border: 1px solid rgba(79,142,247,0.2);
    border-left: 3px solid var(--accent-blue);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.insight-box strong { color: var(--accent-blue); }

/* ── Big hero header ── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: var(--accent-blue);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #4f8ef7 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 620px;
    margin: 0 auto 2rem;
    line-height: 1.65;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: center;
}
.hero-stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.hero-stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 1rem 0 0.5rem;
    text-align: center;
}
.sidebar-brand-icon {
    font-size: 2.5rem;
    line-height: 1;
}
.sidebar-brand-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-top: 0.4rem;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Nav group labels ── */
.nav-group-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 1rem 0 0.4rem;
}

/* ── Data table ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* ── Spinner / loading ── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Plotly charts background fix ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-blue) !important;
    background: rgba(79,142,247,0.08) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── Selectbox / slider ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0 !important; }

/* ── Info / success boxes ── */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Rank badge ── */
.rank-badge {
    display: inline-block;
    width: 24px; height: 24px;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 24px;
    text-align: center;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS & CONSTANTS
# ─────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=12),
    margin=dict(t=50, b=40, l=40, r=20),
    title_font=dict(size=15, color="#f1f5f9", family="Space Grotesk"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
)

ACCENT_PALETTE = ["#4f8ef7", "#a855f7", "#06b6d4", "#10b981", "#f97316", "#ec4899", "#f59e0b", "#ef4444"]
CATEGORY_COLORS = {
    "Creative Task":    "#4f8ef7",
    "Listing Task":     "#a855f7",
    "Explanation":      "#06b6d4",
    "Editing/Rewriting":"#10b981",
    "Classification":   "#f97316",
    "Advice":           "#ec4899",
    "Problem Solving":  "#f59e0b",
    "Question":         "#ef4444",
    "Other":            "#64748b",
}
READABILITY_COLORS = {
    "Very easy":          "#10b981",
    "Easy":               "#4f8ef7",
    "Medium":             "#a855f7",
    "Difficult":          "#f97316",
    "Very Difficult":     "#ef4444",
    "Extremely Difficult":"#7f1d1d",
}


def section_header(icon, title, badge_text, badge_color):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.4rem">{icon}</span>
        <h2>{title}</h2>
        <span class="section-badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44">
            {badge_text}
        </span>
    </div>""", unsafe_allow_html=True)


def chart_card(title, content_fn, insight=None):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
    content_fn()
    if insight:
        st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label, value, delta="", color="#4f8ef7", icon=""):
    st.markdown(f"""
    <div class="metric-card" style="--accent:{color}">
        <div style="font-size:1.6rem;margin-bottom:0.2rem">{icon}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_parquet(
        "hf://datasets/vicgalle/alpaca-gpt4/data/train-00000-of-00001-6ef3991c06080e14.parquet"
    )
    return df


@st.cache_data(show_spinner=False)
def build_features(_df):
    df = _df.copy()

    # ── strip whitespace
    df["instruction"] = df["instruction"].str.strip()

    # ── classify prompts
    def classify_prompt(text):
        t = text.lower()
        if t.startswith(("can you","could you","would you","do you","did you","is it","are there","should i")):
            return "Question"
        elif t.startswith(("write","create","generate","compose","draft")):
            return "Creative Task"
        elif t.startswith(("explain","describe","define","clarify","elaborate")):
            return "Explanation"
        elif t.startswith(("calculate","solve","compute","find the value","evaluate")):
            return "Problem Solving"
        elif t.startswith(("give","list","provide","name","mention","outline","state")):
            return "Listing Task"
        elif t.startswith(("suggest","recommend","advice","tips for","ways to")):
            return "Advice"
        elif t.startswith(("rewrite","rephrase","improve","edit","correct","fix")):
            return "Editing/Rewriting"
        elif t.startswith(("classify","categorize","group the following","label the following")):
            return "Classification"
        else:
            return "Other"

    df["prompt_class"] = df["instruction"].apply(classify_prompt)

    # ── flesch score
    df["flesch_score"] = df["output"].apply(textstat.flesch_reading_ease)

    # ── readability level
    def readability_level(s):
        if s >= 90:   return "Very easy"
        elif s >= 60: return "Easy"
        elif s >= 30: return "Medium"
        elif s >= 10: return "Difficult"
        elif s >= 0:  return "Very Difficult"
        else:         return "Extremely Difficult"

    df["readability_level"] = df["flesch_score"].apply(readability_level)

    # ── instruction word count
    df["instruction_word_count"] = df["instruction"].apply(lambda x: len(x.split()))

    # ── sentence count + output word count
    def count_sentences(text):
        parts = re.split(r"[.!?]", str(text))
        return len([s for s in parts if s.strip()])

    df["sentence_count"]    = df["output"].apply(count_sentences)
    df["output_word_count"] = df["output"].apply(lambda x: len(x.split()))

    # ── words per sentence (safe)
    df["word_per_sentence"] = np.where(
        df["sentence_count"] == 0, 0,
        df["output_word_count"] / df["sentence_count"]
    )

    # ── has input
    df["has_input"] = df["input"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
    )

    return df


@st.cache_data(show_spinner=False)
def build_wordcloud_image(_df):
    nltk.download("stopwords", quiet=True)
    sw = stopwords.words("english")
    custom_sw = [
        "write","generate","create","give","list","describe","explain",
        "provide","make","find","identify","suggest","classify","rewrite",
        "summarize","compare","construct","edit","following","given","using",
        "based","text","sentence","sentences","paragraph","article","statement",
        "example","words","word","name","phrase","output","input","use","used",
        "way","come","new","different","various","type","kind",
        "one","two","three","four","five","first","second","third",
        "1","2","3","4","5","10","number","numbers"
    ]
    sw.extend(custom_sw)

    def clean_text(t):
        t = str(t).lower()
        return re.sub(r"[^a-z\s]", "", t)

    all_text = " ".join(_df["instruction"].apply(clean_text))

    wc = WordCloud(
        width=1400, height=600,
        stopwords=set(sw),
        background_color=None,
        mode="RGBA",
        colormap="cool",
        max_words=180,
        prefer_horizontal=0.85,
        relative_scaling=0.55,
        min_font_size=10,
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="none", transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🤖</div>
        <div class="sidebar-brand-title">ChatGPT Analytics</div>
        <div class="sidebar-brand-sub">52K Prompts · Deep Dive</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="nav-group-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        options=[
            "🏠  Overview",
            "☁️  Word Cloud",
            "🏷️  Prompt Types",
            "📖  Readability",
            "📏  Length vs Clarity",
            "🗣️  Verbosity",
            "📎  Context Effect",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="nav-group-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#475569;line-height:1.7;padding:0.25rem 0">
        📦 <span style="color:#94a3b8">Alpaca-GPT4</span><br>
        🤗 HuggingFace Hub<br>
        📊 52,002 rows · 3 cols<br>
        📅 GPT-4 Responses
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;text-align:center;padding-top:0.5rem">
        Built by <a href="https://github.com/doyancha" style="color:#4f8ef7;text-decoration:none">@doyancha</a>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOAD WITH SPINNER
# ─────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading dataset from HuggingFace…"):
    raw_df = load_data()

with st.spinner("⚙️ Computing features…"):
    df = build_features(raw_df)


# ─────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────

if "Overview" in page:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class="hero-title">ChatGPT Prompt Analytics</div>
        <div class="hero-subtitle">
            A comprehensive behavioural study of <strong>52,002 real ChatGPT prompts</strong> —
            uncovering how people write, what they ask, and how the model responds.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value" style="color:#4f8ef7">52,002</div>
                <div class="hero-stat-label">Prompt–Response Pairs</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value" style="color:#a855f7">8</div>
                <div class="hero-stat-label">Prompt Categories</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value" style="color:#06b6d4">6</div>
                <div class="hero-stat-label">Analysis Lenses</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value" style="color:#10b981">GPT-4</div>
                <div class="hero-stat-label">Response Source</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    avg_flesch = df["flesch_score"].mean()
    avg_words  = df["output_word_count"].mean()
    avg_wps    = df["word_per_sentence"].median()
    ctx_pct    = (df["has_input"] == 1).mean() * 100
    top_type   = df[df["prompt_class"] != "Other"]["prompt_class"].value_counts().index[0]

    with c1: metric_card("Avg Readability", f"{avg_flesch:.1f}", "Flesch score", "#4f8ef7", "📖")
    with c2: metric_card("Avg Response Length", f"{avg_words:.0f}", "words per answer", "#a855f7", "📝")
    with c3: metric_card("Median Sentence", f"{avg_wps:.1f}", "words/sentence", "#06b6d4", "🗣️")
    with c4: metric_card("Context Provided", f"{ctx_pct:.1f}%", "of all prompts", "#10b981", "📎")
    with c5: metric_card("Top Prompt Type", top_type.split()[0], "most used category", "#f97316", "🏆")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column quick charts
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("🏷️", "Prompt Type Snapshot", "Classification", "#4f8ef7")
        counts = df[df["prompt_class"] != "Other"]["prompt_class"].value_counts().reset_index()
        counts.columns = ["Type", "Count"]
        colors = [CATEGORY_COLORS.get(t, "#64748b") for t in counts["Type"]]

        fig = go.Figure(go.Bar(
            x=counts["Count"], y=counts["Type"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=counts["Count"], textposition="inside",
            textfont=dict(color="white", size=11, family="Space Grotesk"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title="",
                          xaxis_title="Number of Prompts", yaxis_title="",
                          yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        section_header("📖", "Readability Distribution", "Flesch Score", "#a855f7")
        rl = df["readability_level"].value_counts().reset_index()
        rl.columns = ["Level", "Count"]
        colors_rl = [READABILITY_COLORS.get(l, "#64748b") for l in rl["Level"]]

        fig2 = go.Figure(go.Pie(
            labels=rl["Level"], values=rl["Count"],
            hole=0.55,
            marker=dict(colors=colors_rl, line=dict(color="#0a0e1a", width=2)),
            textinfo="percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>responses</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#f1f5f9", family="Space Grotesk")
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True,
                           legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1,
                                       font=dict(color="#94a3b8", size=10)))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Key findings callout
    st.markdown("---")
    section_header("💡", "Key Findings", "Summary", "#10b981")
    findings = [
        ("🥇", "#4f8ef7", "Creative Tasks Dominate",
         f"<b>Creative Task</b> is the most common prompt type with <b>{counts.iloc[0]['Count']:,}</b> prompts — "
         f"nearly {counts.iloc[0]['Count']/counts.iloc[1]['Count']:.1f}× more than the second category."),
        ("📖", "#a855f7", "Medium Readability is the Norm",
         f"<b>{(df['readability_level']=='Medium').mean()*100:.1f}%</b> of ChatGPT responses sit at a "
         "\"Medium\" reading level — like a well-written magazine article."),
        ("📏", "#06b6d4", "Prompt Length ≈ Irrelevant to Clarity",
         "The correlation between prompt length and readability is near-zero (~+0.10). "
         "Word count alone does <b>not</b> predict how readable the response will be."),
        ("📎", "#10b981", "Context = Shorter + Clearer Answers",
         "Prompts with extra context get responses that are <b>52% shorter</b> and "
         "<b>7 points more readable</b>. Specificity beats length every time."),
    ]
    cols = st.columns(2)
    for i, (rank, color, title, body) in enumerate(findings):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color};text-align:left;margin-bottom:1rem">
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                    <span style="font-size:1.2rem">{rank}</span>
                    <span style="font-size:0.88rem;font-weight:600;color:{color}">{title}</span>
                </div>
                <div style="font-size:0.82rem;color:#94a3b8;line-height:1.65">{body}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: WORD CLOUD
# ─────────────────────────────────────────────────────────────────

elif "Word Cloud" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("☁️", "Word Cloud Analysis", "NLP", "#06b6d4")
    st.markdown("""
    <div class="insight-box">
        The word cloud below visualises the <strong>most frequent meaningful words</strong> across all 52,002 ChatGPT prompts,
        after filtering out stop words and common instruction verbs. Bigger = more frequent.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("🎨 Generating word cloud…"):
        wc_buf = build_wordcloud_image(df)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.image(wc_buf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top words table
    section_header("📊", "Top Words Breakdown", "Frequency", "#4f8ef7")
    import re as _re
    from collections import Counter
    nltk.download("stopwords", quiet=True)
    sw_set = set(stopwords.words("english")) | {
        "write","generate","create","give","list","describe","explain","provide","make",
        "find","identify","suggest","classify","rewrite","summarize","compare","construct",
        "edit","following","given","using","based","text","sentence","sentences",
        "paragraph","article","statement","example","words","word","name","phrase",
        "output","input","use","used","way","come","new","different","various","type",
        "kind","one","two","three","four","five","first","second","third","1","2","3","4","5","10"
    }

    all_words = " ".join(df["instruction"].str.lower()).split()
    filtered  = [w for w in all_words if w.isalpha() and w not in sw_set and len(w) > 2]
    top_words = Counter(filtered).most_common(20)
    tw_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    col1, col2 = st.columns([3, 2])
    with col1:
        colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(tw_df))]
        fig = go.Figure(go.Bar(
            x=tw_df["Frequency"], y=tw_df["Word"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=tw_df["Frequency"], textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
                          title="Top 20 Most Frequent Words",
                          xaxis_title="Frequency", yaxis_title="",
                          yaxis=dict(autorange="reversed",
                                     gridcolor="rgba(0,0,0,0)",
                                     tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("""
        <div class="chart-card" style="margin-top:0">
        <div class="chart-title">Word Frequency Table</div>""", unsafe_allow_html=True)
        st.dataframe(
            tw_df.style.background_gradient(subset=["Frequency"], cmap="Blues"),
            use_container_width=True, height=420, hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: PROMPT TYPES
# ─────────────────────────────────────────────────────────────────

elif "Prompt Types" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("🏷️", "Prompt Type Analysis", "Classification", "#4f8ef7")
    st.markdown("""
    <div class="insight-box">
        Every prompt is classified into one of <strong>8 categories</strong> using rule-based keyword matching on the
        first few words of the instruction. This reveals <em>what</em> people predominantly ask ChatGPT to do.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    counts_all = df["prompt_class"].value_counts().reset_index()
    counts_all.columns = ["Prompt Type", "Count"]
    counts_no_other = counts_all[counts_all["Prompt Type"] != "Other"].copy()
    counts_no_other["Pct"] = (counts_no_other["Count"] / counts_no_other["Count"].sum() * 100).round(1)

    # Metric row
    cols = st.columns(4)
    top4 = counts_no_other.head(4)
    colors4 = ["#4f8ef7", "#a855f7", "#06b6d4", "#10b981"]
    icons4 = ["✍️", "📋", "💡", "✂️"]
    for i, (_, row) in enumerate(top4.iterrows()):
        with cols[i]:
            metric_card(row["Prompt Type"], f"{row['Count']:,}",
                        f"{row['Pct']}% of classified", colors4[i], icons4[i])

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Bar Chart", "🍩 Donut Chart"])

    with tab1:
        color_list = [CATEGORY_COLORS.get(t, "#64748b") for t in counts_no_other["Prompt Type"]]
        fig = go.Figure(go.Bar(
            x=counts_no_other["Prompt Type"],
            y=counts_no_other["Count"],
            marker=dict(color=color_list, line=dict(width=0)),
            text=counts_no_other["Count"],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=12),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title="Distribution of ChatGPT Prompt Types",
                          xaxis_title="Prompt Type", yaxis_title="Number of Prompts",
                          bargap=0.35)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            color_pie = [CATEGORY_COLORS.get(t, "#64748b") for t in counts_no_other["Prompt Type"]]
            fig2 = go.Figure(go.Pie(
                labels=counts_no_other["Prompt Type"],
                values=counts_no_other["Count"],
                hole=0.5,
                marker=dict(colors=color_pie, line=dict(color="#0a0e1a", width=2)),
                textinfo="percent+label",
                textfont=dict(size=11, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value:,} prompts<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=420,
                               title="Prompt Type Share",
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        with col_r:
            st.markdown('<div class="chart-card" style="margin-top:0"><div class="chart-title">Full Breakdown</div>', unsafe_allow_html=True)
            for _, row in counts_no_other.iterrows():
                c = CATEGORY_COLORS.get(row["Prompt Type"], "#64748b")
                pct = row["Pct"]
                st.markdown(f"""
                <div style="margin-bottom:0.75rem">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;
                                color:#94a3b8;margin-bottom:0.25rem">
                        <span>{row['Prompt Type']}</span>
                        <span style="color:{c};font-weight:600">{row['Count']:,}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px">
                        <div style="width:{pct}%;background:{c};height:6px;border-radius:4px;
                                    transition:width 0.5s ease"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Creative Task</strong> (write, create, generate…) is <em>the dominant mode</em> — nearly double the
        second-place <strong>Listing Task</strong>. Direct <strong>Questions</strong> are surprisingly rare,
        showing ChatGPT is used more as a <em>creation engine</em> than a search engine.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: READABILITY
# ─────────────────────────────────────────────────────────────────

elif "Readability" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("📖", "Readability Analysis", "Flesch Score", "#a855f7")
    st.markdown("""
    <div class="insight-box">
        The <strong>Flesch Reading Ease</strong> score rates text from 0–100+ based on sentence length and syllable count.
        Higher = easier to read. Most newspaper articles score around 60–70.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Mean Score", f"{df['flesch_score'].mean():.1f}", "overall average", "#a855f7", "📊")
    with c2: metric_card("Median Score", f"{df['flesch_score'].median():.1f}", "50th percentile", "#4f8ef7", "📍")
    with c3: metric_card("Std Deviation", f"{df['flesch_score'].std():.1f}", "score spread", "#06b6d4", "📐")
    with c4:
        pct_easy = (df["flesch_score"] >= 60).mean() * 100
        metric_card("Easy or Better", f"{pct_easy:.1f}%", "score ≥ 60", "#10b981", "✅")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Donut
        rl = df["readability_level"].value_counts().reset_index()
        rl.columns = ["Level", "Count"]
        order = ["Very easy","Easy","Medium","Difficult","Very Difficult","Extremely Difficult"]
        rl["Level"] = pd.Categorical(rl["Level"], categories=order, ordered=True)
        rl = rl.sort_values("Level")
        colors_rl = [READABILITY_COLORS[l] for l in rl["Level"]]

        fig = go.Figure(go.Pie(
            labels=rl["Level"], values=rl["Count"],
            hole=0.55,
            marker=dict(colors=colors_rl, line=dict(color="#0a0e1a", width=2)),
            textinfo="percent+label",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,} responses (%{percent})<extra></extra>",
            sort=False,
        ))
        fig.add_annotation(
            text="<b>Readability<br>Levels</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#94a3b8", family="Space Grotesk")
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title="Readability Level Distribution", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Histogram of raw flesch scores
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df["flesch_score"].clip(-100, 120),
            nbinsx=60,
            marker=dict(
                color="#a855f7",
                opacity=0.8,
                line=dict(color="#0a0e1a", width=0.5)
            ),
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
            name="Flesch Score"
        ))
        # Reference lines
        for val, label, color in [(0,"Difficult→",  "#f97316"),
                                   (60,"Easy→",      "#10b981"),
                                   (90,"Very Easy→", "#4f8ef7")]:
            fig2.add_vline(x=val, line=dict(color=color, dash="dash", width=1.5))
            fig2.add_annotation(x=val+2, y=1, yref="paper", text=label,
                                showarrow=False, font=dict(color=color, size=10))
        fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                           title="Distribution of Flesch Scores",
                           xaxis_title="Flesch Reading Ease Score",
                           yaxis_title="Number of Responses",
                           bargap=0.02)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Readability by prompt type
    section_header("🔀", "Readability by Prompt Type", "Breakdown", "#f97316")
    grp = df[df["prompt_class"] != "Other"].groupby("prompt_class")["flesch_score"].mean().reset_index()
    grp.columns = ["Prompt Type", "Avg Flesch Score"]
    grp = grp.sort_values("Avg Flesch Score", ascending=True)
    colors_grp = [CATEGORY_COLORS.get(t, "#64748b") for t in grp["Prompt Type"]]

    fig3 = go.Figure(go.Bar(
        y=grp["Prompt Type"], x=grp["Avg Flesch Score"],
        orientation="h",
        marker=dict(color=colors_grp, line=dict(width=0)),
        text=grp["Avg Flesch Score"].round(1),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Avg Score: %{x:.1f}<extra></extra>",
    ))
    fig3.add_vline(x=60, line=dict(color="#10b981", dash="dash", width=1.5))
    fig3.add_annotation(x=61, y=1, yref="paper", text="Easy threshold",
                        showarrow=False, font=dict(color="#10b981", size=10))
    fig3.update_layout(**PLOTLY_LAYOUT, height=340,
                       title="Average Readability Score by Prompt Category",
                       xaxis_title="Average Flesch Score", yaxis_title="",
                       yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#94a3b8")))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="insight-box">
        💡 ChatGPT calibrates response readability to the <strong>task type</strong>, not to prompt length.
        Creative and advisory responses tend to be more readable, while problem-solving and classification
        tasks produce denser, more technical text.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: LENGTH vs CLARITY
# ─────────────────────────────────────────────────────────────────

elif "Length vs Clarity" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("📏", "Prompt Length vs Readability", "Regression", "#06b6d4")
    st.markdown("""
    <div class="insight-box">
        Does writing a <strong>longer prompt</strong> make ChatGPT give you a <strong>clearer answer</strong>?
        This section tests that hypothesis using linear regression and Pearson correlation.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m, c_int = np.polyfit(df["instruction_word_count"], df["flesch_score"], 1)
    corr = df["instruction_word_count"].corr(df["flesch_score"])

    co1, co2, co3, co4 = st.columns(4)
    with co1: metric_card("Correlation (r)", f"{corr:.4f}", "near-zero = no link", "#06b6d4", "🔗")
    with co2: metric_card("Slope (m)", f"{m:.4f}", "per extra word", "#4f8ef7", "📉")
    with co3: metric_card("Intercept (c)", f"{c_int:.1f}", "baseline score", "#a855f7", "📌")
    with co4:
        avg_prompt_len = df["instruction_word_count"].mean()
        metric_card("Avg Prompt Length", f"{avg_prompt_len:.1f}", "words per prompt", "#10b981", "✍️")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter slider
    max_words = int(df["instruction_word_count"].quantile(0.98))
    word_range = st.slider(
        "Filter prompts by word count range",
        min_value=1, max_value=max_words,
        value=(1, max_words),
        help="Drag to zoom into a specific word-count range"
    )
    df_filtered = df[(df["instruction_word_count"] >= word_range[0]) &
                     (df["instruction_word_count"] <= word_range[1])]
    st.caption(f"Showing {len(df_filtered):,} of {len(df):,} prompts")

    tab1, tab2 = st.tabs(["🔵 Interactive Scatter", "📉 Regression Heatmap"])

    with tab1:
        sample = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)
        fig = px.scatter(
            sample,
            x="instruction_word_count",
            y="flesch_score",
            color="flesch_score",
            size="instruction_word_count",
            size_max=14,
            color_continuous_scale="Viridis",
            labels={"instruction_word_count": "Prompt Word Count", "flesch_score": "Readability Score"},
            hover_data={"instruction_word_count": True, "flesch_score": ":.1f"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=440,
                          title="Prompt Length vs Flesch Readability Score",
                          coloraxis_colorbar=dict(
                              title=dict(text="Score", font=dict(color="#94a3b8")),
                              tickfont=dict(color="#94a3b8")
                          ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        # 2D histogram / heatmap
        fig2 = go.Figure(go.Histogram2dContour(
            x=df_filtered["instruction_word_count"],
            y=df_filtered["flesch_score"].clip(-100, 120),
            colorscale="Viridis",
            contours=dict(showlabels=True,
                          labelfont=dict(size=9, color="white")),
            hovertemplate="Word count: %{x}<br>Flesch: %{y}<br>Density: %{z}<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=440,
                           title="Density Heatmap — Where Prompts Cluster",
                           xaxis_title="Prompt Word Count",
                           yaxis_title="Flesch Score")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="insight-box">
        💡 The Pearson correlation is <strong>r = {corr:.4f}</strong> — essentially zero.
        The regression slope of <strong>{m:.4f}</strong> means that adding 10 words to a prompt
        changes the readability score by only <strong>{m*10:.2f} points</strong> on average.
        This confirms that prompt length <em>does not predict</em> response clarity.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: VERBOSITY
# ─────────────────────────────────────────────────────────────────

elif "Verbosity" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("🗣️", "Verbosity Analysis", "Words per Sentence", "#f97316")
    st.markdown("""
    <div class="insight-box">
        <strong>Words per sentence</strong> is a key indicator of text density.
        Normal professional writing sits around 15–25 words/sentence. Code, lists,
        and structured outputs can inflate this metric dramatically.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Median WPS", f"{df['word_per_sentence'].median():.1f}", "words/sentence", "#f97316", "📍")
    with c2: metric_card("Mean WPS", f"{df['word_per_sentence'].mean():.1f}", "words/sentence", "#4f8ef7", "📊")
    with c3: metric_card("75th Percentile", f"{df['word_per_sentence'].quantile(0.75):.1f}", "words/sentence", "#a855f7", "📐")
    with c4:
        outliers = (df["word_per_sentence"] > 100).sum()
        metric_card("Outliers >100", f"{outliers:,}", "extreme responses", "#ef4444", "⚠️")

    st.markdown("<br>", unsafe_allow_html=True)

    # Clip for better visualisation
    clip_val = st.slider("Clip y-axis at (words/sentence)", 20, 400, 120,
                          help="Reduce to focus on the main distribution")
    df_clip = df[df["word_per_sentence"] <= clip_val]

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df_clip["word_per_sentence"],
            name="Words/Sentence",
            boxmean="sd",
            jitter=0.35,
            pointpos=-1.6,
            marker=dict(color="#f97316", size=3, opacity=0.35),
            line=dict(color="#f97316", width=2),
            fillcolor="rgba(249,115,22,0.15)",
            hovertemplate="WPS: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=f"Distribution of Words per Sentence (clipped at {clip_val})",
                          yaxis_title="Words Per Sentence",
                          xaxis_title="",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Stats table
        st.markdown('<div class="chart-card"><div class="chart-title">Descriptive Stats</div>', unsafe_allow_html=True)
        stats = df["word_per_sentence"].describe().round(2)
        stats_df = pd.DataFrame({"Statistic": stats.index, "Value": stats.values})
        st.dataframe(stats_df, use_container_width=True, height=300, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Sentence count distribution
        st.markdown('<div class="chart-card"><div class="chart-title">Sentence Count Distribution</div>', unsafe_allow_html=True)
        sc_clip = df[df["sentence_count"] <= 50]["sentence_count"]
        fig_sc = go.Figure(go.Histogram(
            x=sc_clip, nbinsx=30,
            marker=dict(color="#06b6d4", opacity=0.8, line=dict(color="#0a0e1a", width=0.5)),
        ))
        fig_sc.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'}, height=200, margin=dict(t=10,b=30,l=30,r=10),
                              xaxis_title="Sentences", yaxis_title="Count")
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # WPS by prompt type
    section_header("🔀", "Verbosity by Prompt Type", "Comparison", "#a855f7")
    grp2 = df[df["prompt_class"] != "Other"].groupby("prompt_class")["word_per_sentence"].median().reset_index()
    grp2.columns = ["Prompt Type", "Median WPS"]
    grp2 = grp2.sort_values("Median WPS", ascending=False)
    colors_g2 = [CATEGORY_COLORS.get(t, "#64748b") for t in grp2["Prompt Type"]]

    fig4 = go.Figure(go.Bar(
        x=grp2["Prompt Type"], y=grp2["Median WPS"],
        marker=dict(color=colors_g2, line=dict(width=0)),
        text=grp2["Median WPS"].round(1),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig4.update_layout(**PLOTLY_LAYOUT, height=340,
                       title="Median Words per Sentence by Prompt Type",
                       xaxis_title="Prompt Type", yaxis_title="Median WPS",
                       bargap=0.35)
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="insight-box">
        💡 Most ChatGPT responses use <strong>15–25 words per sentence</strong> — consistent with
        professional writing. Extreme outliers (100+ WPS) are caused by responses containing
        bullet points, numbered lists, or code blocks that the sentence splitter interprets as one
        giant run-on sentence.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE: CONTEXT EFFECT
# ─────────────────────────────────────────────────────────────────

elif "Context Effect" in page:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>🔬 Exploratory Data Analysis · NLP · Data Viz</div>
        <div class='hero-title'>ChatGPT Prompt Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    section_header("📎", "Context Effect Analysis", "Input vs No Input", "#10b981")
    st.markdown("""
    <div class="insight-box">
        Some prompts include an <strong>extra context field</strong> (the <code>input</code> column) — additional
        background information beyond the instruction. This section tests whether that context
        changes the <em>length</em> or <em>readability</em> of ChatGPT's responses.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Build comparison
    comparison = df.groupby("has_input")[["output_word_count", "flesch_score"]].mean().reset_index()
    comparison["context"] = comparison["has_input"].map({0: "No Extra Context", 1: "Extra Context"})

    no_ctx = comparison[comparison["has_input"] == 0].iloc[0]
    with_ctx = comparison[comparison["has_input"] == 1].iloc[0]

    # Context split
    ctx_counts = df["has_input"].value_counts()
    no_n  = ctx_counts.get(0, 0)
    yes_n = ctx_counts.get(1, 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Without Context", f"{no_n:,}", f"{no_n/len(df)*100:.1f}% of dataset", "#ef4444", "📭")
    with c2: metric_card("With Context", f"{yes_n:,}", f"{yes_n/len(df)*100:.1f}% of dataset", "#10b981", "📨")
    with c3:
        wc_diff = no_ctx["output_word_count"] - with_ctx["output_word_count"]
        metric_card("Word Count Drop", f"−{wc_diff:.0f}", "words when context given", "#f97316", "📉")
    with c4:
        rs_diff = with_ctx["flesch_score"] - no_ctx["flesch_score"]
        metric_card("Readability Gain", f"+{rs_diff:.1f}", "Flesch points with context", "#4f8ef7", "📈")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Grouped bar
        fig = go.Figure()
        for metric, color, label in [
            ("output_word_count", "#4f8ef7", "Avg Word Count"),
            ("flesch_score",      "#10b981", "Avg Flesch Score"),
        ]:
            fig.add_trace(go.Bar(
                x=comparison["context"],
                y=comparison[metric],
                name=label,
                marker=dict(color=color, line=dict(width=0), opacity=0.85),
                text=comparison[metric].round(1),
                textposition="outside",
                textfont=dict(color="#94a3b8", size=12),
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=400, barmode="group",
                          title="Effect of Extra Context on ChatGPT Responses",
                          xaxis_title="Context Provided?",
                          yaxis_title="Average Value",
                          bargap=0.25, bargroupgap=0.05)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Before/after comparison cards
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Side-by-Side Comparison</div>
        """, unsafe_allow_html=True)

        for metric_name, no_val, yes_val, unit, better in [
            ("Avg Response Length", no_ctx["output_word_count"], with_ctx["output_word_count"], "words", "lower"),
            ("Avg Readability",     no_ctx["flesch_score"],      with_ctx["flesch_score"],      "score", "higher"),
        ]:
            delta = yes_val - no_val
            arrow = "▼" if delta < 0 else "▲"
            delta_color = "#10b981" if (better == "lower" and delta < 0) or (better == "higher" and delta > 0) else "#ef4444"
            st.markdown(f"""
            <div style="margin-bottom:1.25rem">
                <div style="font-size:0.78rem;color:#475569;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:0.6rem">{metric_name}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
                    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                                border-radius:10px;padding:0.75rem;text-align:center">
                        <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:0.3rem">No Context</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#ef4444">{no_val:.1f}</div>
                        <div style="font-size:0.7rem;color:#475569">{unit}</div>
                    </div>
                    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                                border-radius:10px;padding:0.75rem;text-align:center">
                        <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:0.3rem">With Context</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#10b981">{yes_val:.1f}</div>
                        <div style="font-size:0.7rem;color:{delta_color}">{arrow} {abs(delta):.1f} {unit}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Donut of context split
        fig_ctx = go.Figure(go.Pie(
            labels=["No Context", "With Context"],
            values=[no_n, yes_n],
            hole=0.6,
            marker=dict(colors=["#ef4444", "#10b981"],
                        line=dict(color="#0a0e1a", width=2)),
            textinfo="percent+label",
            textfont=dict(size=11, color="white"),
        ))
        fig_ctx.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'}, height=220, showlegend=False,
                              margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_ctx, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Distribution comparison
    section_header("📊", "Output Length Distribution by Context", "Histogram", "#f97316")
    fig5 = go.Figure()
    for ctx_val, label, color in [(0, "No Context", "#ef4444"), (1, "With Context", "#10b981")]:
        subset = df[df["has_input"] == ctx_val]["output_word_count"]
        fig5.add_trace(go.Histogram(
            x=subset.clip(0, 600), name=label,
            nbinsx=60,
            marker=dict(color=color, opacity=0.6, line=dict(color="#0a0e1a", width=0.3)),
            hovertemplate=f"{label}<br>Words: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig5.update_layout(**PLOTLY_LAYOUT, height=360, barmode="overlay",
                       title="Output Word Count: Context vs No Context",
                       xaxis_title="Output Word Count",
                       yaxis_title="Number of Responses")
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="insight-box">
        💡 This is the most surprising finding: prompts <em>with</em> extra context produce responses that are
        <strong>{wc_diff:.0f} words shorter</strong> on average — a <strong>{wc_diff/no_ctx['output_word_count']*100:.0f}% reduction</strong> —
        and <strong>{rs_diff:.1f} Flesch points more readable</strong>.
        Specificity helps ChatGPT stay focused. Vague prompts produce long, generic answers.
        <strong>Quality beats quantity.</strong>
    </div>""", unsafe_allow_html=True)
