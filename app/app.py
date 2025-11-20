"""
app/app.py
Stakeholder-grade Sentiment Dashboard (complete file)
Assumes a CSV with columns: Year, Month, Day, Time of Tweet, text, sentiment, Platform
"""

import os, time, io, math
from collections import Counter
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------- Config ----------------
st.set_page_config(
    page_title="Stakeholder Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------- CHART FONT ENHANCEMENTS --------------
import plotly.io as pio

pio.templates["improved"] = pio.templates["plotly_dark"]
pio.templates["improved"].layout.update({
    "font": {"size": 16},
    "xaxis": {
        "title_font": {"size": 20, "color": "white", "family": "Arial Black"},
        "tickfont": {"size": 14, "color": "white"}
    },
    "yaxis": {
        "title_font": {"size": 20, "color": "white", "family": "Arial Black"},
        "tickfont": {"size": 14, "color": "white"}
    }
})

pio.templates.default = "improved"

# ---------------- Styling (big readable fonts + cards) ----------------
st.markdown(
    """
    <style>
    /* Titles */
    h1 { font-size:48px !important; font-weight:800; margin-bottom:6px; }
    h2 { font-size:28px !important; font-weight:700; margin-top:18px; }
    h3 { font-size:20px !important; font-weight:600; }

    /* Body */
    .main > div { padding: 18px 28px; }
    p, span, li { font-size:16px !important; }

    /* Cards */
    .card { padding:14px; border-radius:12px; background-color: rgba(255,255,255,0.02); }
    .kpi { font-size:40px; font-weight:700; margin:0; white-space:nowrap; }
    .kpi-label { font-size:13px; color:#c4c4c4; margin-top:6px; }

    /* small helpers */
    .muted { color:#9aa0a6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Utilities ----------------
def try_read_csv(path):
    """Robust CSV loader."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin1", engine="python")
    return df

def clean_text_simple(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"#\w+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def top_n_words(texts, n=20, stopwords=None):
    stopwords = stopwords or set()
    cnt = Counter()
    for t in texts:
        for w in str(t).split():
            w = w.strip().lower()
            if len(w) <= 2 or w in stopwords:
                continue
            cnt[w] += 1
    return cnt.most_common(n)

# VADER fallback sentiment
_vader = None
def vader_compound(text):
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except Exception:
            _vader = None
    if _vader:
        return _vader.polarity_scores(text)["compound"]
    # fallback simple heuristic:
    pos_words = {"love","good","great","nice","best","amazing","happy","like","enjoy"}
    neg_words = {"hate","bad","worst","terrible","awful","sad","angry","kill","murder"}
    tset = set(text.split())
    p = len(pos_words & tset)
    n = len(neg_words & tset)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Controls")
    default_candidates = [
        r"data/sentiment_analysis.csv",
        r"..\data\sentiment_analysis.csv",
        r"C:\Users\ASUS\Data Science\real-time-social-sentiment\data\sentiment_analysis.csv",
        "/mnt/data/data_extracted/data/sentiment_analysis.csv",
    ]
    default_path = next((p for p in default_candidates if os.path.exists(p)), default_candidates[-1])
    data_path = st.text_input("CSV path", value=default_path)
    load = st.button("Load dataset")
    st.markdown("---")
    st.subheader("Trend options")
    smoothing = st.checkbox("Apply smoothing to trend", value=True)
    rolling_window = st.slider("Rolling window size (points)", min_value=5, max_value=500, value=60, step=5)
    st.markdown("---")
    st.subheader("Drilldown & simulation")
    show_drill = st.checkbox("Show drilldown table", value=True)
    sim_delay = st.slider("Stream delay (s)", 0.1, 1.0, 0.4)

# ---------------- Load and prepare DF ----------------
@st.cache_data
def load_and_prepare(path):
    df = try_read_csv(path)

    # strip headers names
    df.columns = [str(c).strip() for c in df.columns]

    # if headerless style (numeric columns), try map to schema:
    if all(re.fullmatch(r"\d+", str(c)) for c in df.columns):
        if df.shape[1] >= 7:
            df = df.rename(columns={0:"Year",1:"Month",2:"Day",3:"Time of Tweet",4:"text",5:"sentiment",6:"Platform"})
        else:
            df.columns = [f"c{i}" for i in range(df.shape[1])]

    # ensure text column
    text_col = None
    for cand in ["text","comment","content","tweet","c3","4"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        # pick longest-median column as text
        lens = {c: df[c].astype(str).map(len).median() for c in df.columns}
        text_col = max(lens, key=lens.get)
        df = df.rename(columns={text_col:"text"})

    # create cleaned text
    df["clean_text"] = df["text"].astype(str).apply(clean_text_simple)

    # normalize sentiment
    if "sentiment" in df.columns:
        df["sentiment_norm"] = df["sentiment"].astype(str).str.strip().str.title()
        # if numeric-like mapping (0/2/4)
        unique_vals = set(df["sentiment_norm"].unique())
        numeric_vals = {v for v in unique_vals if re.fullmatch(r"\d+", v)}
        if numeric_vals:
            mapping = {"0":"Negative","1":"Negative","2":"Neutral","3":"Positive","4":"Positive"}
            df["sentiment_norm"] = df["sentiment_norm"].map(mapping).fillna(df["sentiment_norm"])
    else:
        df["vader_compound"] = df["clean_text"].apply(vader_compound)
        df["sentiment_norm"] = df["vader_compound"].apply(lambda s: "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral"))

    # platform
    if "Platform" not in df.columns and "platform" in df.columns:
        df = df.rename(columns={"platform":"Platform"})
    if "Platform" not in df.columns:
        df["Platform"] = "Unknown"

    # timestamp
    if "Time of Tweet" in df.columns:
        df["ts"] = pd.to_datetime(df["Time of Tweet"], errors="coerce")
    else:
        df["ts"] = pd.NaT

    # if all timestamps invalid, fabricate a monotonic series to allow plotting/heatmap
    if df["ts"].isna().all():
        df = df.reset_index(drop=True)
        df["ts"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(df.index, unit="m")

    # hour / bucket
    df["hour"] = df["ts"].dt.hour.fillna(-1).astype(int)
    def bucket(h):
        if h < 0: return "Unknown"
        if h < 6: return "Night"
        if h < 12: return "Morning"
        if h < 17: return "Afternoon"
        if h < 21: return "Evening"
        return "Night"
    df["time_bucket"] = df["hour"].apply(bucket)

    # ensure vader_compound exists
    if "vader_compound" not in df.columns:
        df["vader_compound"] = df["clean_text"].apply(vader_compound)

    return df

# load dataset (on button or initial)
if "df" not in st.session_state or load:
    try:
        st.session_state.df = load_and_prepare(data_path)
        st.success("Dataset loaded")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

df = st.session_state.df

# ---------------- Top controls row (filters + download) ----------------
c1, c2, c3 = st.columns([2,1,1])
with c1:
    st.markdown("**Filters**")
    platforms = sorted(df["Platform"].fillna("Unknown").unique().tolist())
    platform_sel = st.multiselect("Platform", options=platforms, default=platforms)
    sentiments = ["Positive","Neutral","Negative"]
    sent_sel = st.multiselect("Sentiment", options=sentiments, default=sentiments)
with c2:
    st.markdown("**Export**")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download filtered CSV (all)", data=csv, file_name="sentiment_export.csv", mime="text/csv")
with c3:
    st.markdown("**Quick stats**")
    st.write(f"Rows: **{len(df):,}**")
    st.write(f"Platforms: **{len(platforms)}**")

# ---------------- KPIs ----------------
total_comments = len(df)
pos_pct = (df["sentiment_norm"]=="Positive").mean()*100
neu_pct = (df["sentiment_norm"]=="Neutral").mean()*100
neg_pct = (df["sentiment_norm"]=="Negative").mean()*100
most_active = df["Platform"].value_counts().idxmax()
avg_recent = df.sort_values("ts")["vader_compound"].tail(200).mean()
if math.isnan(avg_recent): avg_recent = df["vader_compound"].mean()

k1,k2,k3,k4 = st.columns([2,1.2,1.2,1.6])
with k1:
    st.markdown(f"<div class='card'><div class='kpi'>{total_comments:,}</div><div class='kpi-label'>Total comments</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='kpi' style='color:#2ecc71'>{pos_pct:.1f}%</div><div class='kpi-label'>Positive</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='kpi' style='color:#f39c12'>{neu_pct:.1f}%</div><div class='kpi-label'>Neutral</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='kpi' style='color:#e74c3c'>{neg_pct:.1f}%</div><div class='kpi-label'>Negative</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sentiment trend (big area+line) ----------------
st.subheader("Sentiment Trend")
plot_df = df[df["Platform"].isin(platform_sel) & df["sentiment_norm"].isin(sent_sel)].sort_values("ts")
if plot_df.empty:
    st.warning("No data after filtering")
else:
    plot_df2 = plot_df.copy().reset_index(drop=True)
    if smoothing:
        plot_df2["vader_smooth"] = plot_df2["vader_compound"].rolling(window=rolling_window, min_periods=1).mean()
        ycol = "vader_smooth"
    else:
        ycol = "vader_compound"
    fig = px.area(plot_df2, x="ts", y=ycol, line_group="Platform",
                color_discrete_sequence=["#1f77b4"], labels={ycol:"Sentiment (compound)", "ts":"Time"})
    fig.update_traces(line=dict(color="#1f77b4"))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Platform insights ----------------
st.subheader("Platform Insights")
p1,p2 = st.columns([2,1])
with p1:
    grouped = plot_df.groupby(["Platform","sentiment_norm"]).size().reset_index(name="count")
    if not grouped.empty:
        fig2 = px.bar(grouped, x="Platform", y="count", color="sentiment_norm", barmode="stack",
                    title="Sentiment distribution by platform")
        st.plotly_chart(fig2, use_container_width=True)
with p2:
    vol = plot_df["Platform"].value_counts().reset_index()
    vol.columns = ["Platform","count"]
    fig3 = px.pie(vol, names="Platform", values="count", hole=0.45, title="Platform volume")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ---------------- Time-of-day heatmap ----------------
st.subheader("Time of Day vs Sentiment")
heat = plot_df.groupby(["time_bucket","sentiment_norm"]).size().reset_index(name="count")
if heat.empty:
    st.info("No time-bucket data for chosen filters")
else:
    pivot = heat.pivot_table(index="time_bucket", columns="sentiment_norm", values="count", fill_value=0)
    # normalize rows to percent
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    pivot_plot = pivot_pct.loc[sorted(pivot_pct.index, key=lambda x: ["Unknown","Night","Morning","Afternoon","Evening"].index(x) if x in ["Unknown","Night","Morning","Afternoon","Evening"] else 99)]
    fig_heat = px.imshow(pivot_plot, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ---------------- Keyword insights ----------------
st.subheader("Keyword & Theme Insights")
stopwords = set(["the","and","for","this","that","with","you","have","has","was","are","not","but","they","be","our","your","i","im","it's","its","it"])
pos_texts = plot_df[plot_df["sentiment_norm"]=="Positive"]["clean_text"].astype(str).tolist()
neg_texts = plot_df[plot_df["sentiment_norm"]=="Negative"]["clean_text"].astype(str).tolist()

pos_top = top_n_words(pos_texts, n=20, stopwords=stopwords)
neg_top = top_n_words(neg_texts, n=20, stopwords=stopwords)

cA,cB = st.columns(2)
with cA:
    st.markdown("**Top positive keywords**")
    if pos_top:
        st.bar_chart(pd.DataFrame(pos_top, columns=["word","count"]).set_index("word"))
    else:
        st.info("No positive keywords")
with cB:
    st.markdown("**Top negative keywords**")
    if neg_top:
        st.bar_chart(pd.DataFrame(neg_top, columns=["word","count"]).set_index("word"))
    else:
        st.info("No negative keywords")

st.markdown("---")

# ---------------- Drilldown Table (collapsible) ----------------
with st.expander("Comment Drilldown (click to expand)", expanded=show_drill):
    subf = plot_df.copy()
    q = st.text_input("Search text (contains)", value="")
    if q:
        subf = subf[subf["clean_text"].str.contains(q, case=False, na=False)]
    st.write(f"Showing {len(subf):,} rows (top 200)")
    st.dataframe(subf[["ts","Platform","sentiment_norm","text"]].sort_values("ts", ascending=False).head(200), height=420)

