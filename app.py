"""
app.py
SOC Sentinel — a Streamlit front end for CVE investigation, report
analysis, and risk triage. Run with: streamlit run app.py
"""

import datetime as dt

import streamlit as st

from risk_engine import compute_risk_score
from threat_intel import fetch_cve, fetch_kev_catalog, fetch_epss, is_in_kev
from rag_engine import extract_text, chunk_text, build_index, retrieve, ask_groq

st.set_page_config(
    page_title="SOC Sentinel",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {cve, score, priority, ts}
if "report_chunks" not in st.session_state:
    st.session_state.report_chunks = []
if "report_index" not in st.session_state:
    st.session_state.report_index = None
if "report_name" not in st.session_state:
    st.session_state.report_name = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --------------------------------------------------------------------------
# Theme — dark SOC console: deep navy, amber alert accent, radar/scan motifs
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0A0E1A;
    --panel: #121826;
    --panel-2: #161E30;
    --border: #232E45;
    --text: #E8ECF4;
    --muted: #8B95A8;
    --accent: #FFB020;
    --critical: #FF4757;
    --high: #FF8C42;
    --medium: #FFB020;
    --low: #2ED573;
    --info: #4FC3F7;
}

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background:
        linear-gradient(rgba(10,14,26,0.96), rgba(10,14,26,0.96)),
        repeating-linear-gradient(0deg, #101728 0px, #101728 1px, transparent 1px, transparent 32px),
        repeating-linear-gradient(90deg, #101728 0px, #101728 1px, transparent 1px, transparent 32px);
    background-color: var(--bg);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}

/* Hero */
.hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px 26px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: linear-gradient(120deg, var(--panel) 0%, var(--panel-2) 100%);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -60%; right: -10%;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(255,176,32,0.18) 40deg, transparent 90deg);
    animation: sweep 4s linear infinite;
}
@keyframes sweep { to { transform: rotate(360deg); } }
.hero-title {
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: 30px;
    letter-spacing: 1.5px;
    margin: 0;
    color: var(--text);
    z-index: 1;
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
    z-index: 1;
}
.pulse-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--low);
    box-shadow: 0 0 0 0 rgba(46,213,115,0.7);
    animation: pulse 1.8s infinite;
    display: inline-block; margin-right: 6px;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(46,213,115,0.55); }
    70% { box-shadow: 0 0 0 9px rgba(46,213,115,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,213,115,0); }
}

/* Cards */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card h4 {
    font-family: 'Chakra Petch', sans-serif;
    letter-spacing: 0.5px;
    color: var(--text);
    margin-top: 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}
.mono { font-family: 'IBM Plex Mono', monospace; }
.muted { color: var(--muted); }

.badge {
    display: inline-block;
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.6px;
    padding: 4px 10px;
    border-radius: 3px;
    text-transform: uppercase;
}

.field-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed var(--border); font-size: 14px; }
.field-row:last-child { border-bottom: none; }
.field-label { color: var(--muted); }

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: #0A0E1A;
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 3px;
}
.stButton > button:hover { background: #ffc247; color: #0A0E1A; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Chakra Petch', sans-serif;
    letter-spacing: 0.4px;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_gauge(score: int, color: str) -> str:
    """Build a horizontal 'sensor bar' risk gauge as inline SVG."""
    width = 320
    pointer_x = 8 + (score / 100) * (width - 16)
    zones = [
        (0, 40, "#2ED573"), (40, 60, "#FFB020"),
        (60, 80, "#FF8C42"), (80, 100, "#FF4757"),
    ]
    rects = ""
    for lo, hi, c in zones:
        x = 8 + (lo / 100) * (width - 16)
        w = ((hi - lo) / 100) * (width - 16)
        rects += f'<rect x="{x:.1f}" y="28" width="{w:.1f}" height="14" fill="{c}" opacity="0.85"/>'

    ticks = ""
    for t in range(0, 101, 10):
        x = 8 + (t / 100) * (width - 16)
        ticks += f'<line x1="{x:.1f}" y1="44" x2="{x:.1f}" y2="50" stroke="#3A4666" stroke-width="1"/>'

    return f"""
    <svg viewBox="0 0 {width} 70" width="100%" height="70" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {rects}
      {ticks}
      <polygon points="{pointer_x:.1f},14 {pointer_x-7:.1f},26 {pointer_x+7:.1f},26"
               fill="{color}" filter="url(#glow)"/>
      <text x="{pointer_x:.1f}" y="10" text-anchor="middle" font-family="IBM Plex Mono"
            font-size="11" fill="{color}">{score}</text>
    </svg>
    """


def severity_badge(priority: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}22; color:{color}; border:1px solid {color}66;">{priority}</span>'


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div>
            <p class="hero-title">🛰️ SOC SENTINEL</p>
            <p class="hero-sub"><span class="pulse-dot"></span>LIVE &nbsp;·&nbsp; threat intel · report triage · risk scoring</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🛡️ Console")
    mode = st.radio("Explanation mode", ["Technical", "Beginner"], horizontal=True)
    st.markdown("---")
    st.markdown(f"**Investigations this session:** {len(st.session_state.history)}")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(
                f'<div class="mono" style="font-size:12px; color:var(--muted);">'
                f'{h["ts"]} — {h["cve"]} <span style="color:{h["color"]}">{h["priority"]}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.caption("Set `GROQ_API_KEY` as an environment variable or in `.streamlit/secrets.toml`.")

tab_cve, tab_report, tab_dashboard = st.tabs(["🔎 CVE Lookup", "📄 Report Analysis", "📊 Dashboard"])

# --------------------------------------------------------------------------
# Tab 1 — CVE Lookup
# --------------------------------------------------------------------------
with tab_cve:
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        cve_id = st.text_input("CVE ID", placeholder="e.g. CVE-2024-3400", label_visibility="collapsed")
    with col_btn:
        go = st.button("Investigate", use_container_width=True)

    if go and cve_id.strip():
        with st.spinner("Querying NVD, CISA KEV, and EPSS…"):
            cve = fetch_cve(cve_id)
            kev_catalog = fetch_kev_catalog()
            epss = fetch_epss(cve_id)

        if not cve:
            st.error(f"No record found for {cve_id.strip().upper()} in NVD.")
        else:
            kev_entry = is_in_kev(cve_id, kev_catalog)
            result = compute_risk_score(
                cvss=cve.get("cvss_score"),
                epss=epss,
                is_kev=bool(kev_entry),
                patch_available=True,
            )

            st.session_state.history.append({
                "cve": cve["id"], "score": result.score, "priority": result.priority,
                "color": result.color, "ts": dt.datetime.now().strftime("%H:%M:%S"),
            })

            left, right = st.columns([2, 1])
            with left:
                st.markdown(
                    f"""<div class="card">
                        <h4>{cve['id']} {severity_badge(result.priority, result.color)}</h4>
                        <p>{cve.get('description') or '<span class="muted">No description available.</span>'}</p>
                        <div class="field-row"><span class="field-label">CVSS</span><span class="mono">{cve.get('cvss_score') if cve.get('cvss_score') is not None else '—'} ({cve.get('severity') or 'unknown'})</span></div>
                        <div class="field-row"><span class="field-label">EPSS</span><span class="mono">{f'{epss*100:.1f}%' if epss is not None else '—'}</span></div>
                        <div class="field-row"><span class="field-label">CISA KEV</span><span class="mono">{'YES — added ' + kev_entry.get('dateAdded','') if kev_entry else 'Not listed'}</span></div>
                        <div class="field-row"><span class="field-label">CWE</span><span class="mono">{', '.join(cve.get('cwe_ids') or []) or '—'}</span></div>
                        <div class="field-row"><span class="field-label">Published</span><span class="mono">{cve.get('published') or '—'}</span></div>
                        <div class="field-row"><span class="field-label">Last modified</span><span class="mono">{cve.get('last_modified') or '—'}</span></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if cve.get("affected_products"):
                    with st.expander("Affected products (CPE)"):
                        for p in cve["affected_products"]:
                            st.markdown(f'<span class="mono" style="font-size:12px;">{p}</span>', unsafe_allow_html=True)
                if cve.get("references"):
                    with st.expander("References"):
                        for r in cve["references"]:
                            st.markdown(f"- [{r}]({r})")

            with right:
                st.markdown('<div class="card"><h4>Risk Score</h4>', unsafe_allow_html=True)
                st.markdown(render_gauge(result.score, result.color), unsafe_allow_html=True)
                st.markdown(
                    "<br>".join(f'<span class="muted" style="font-size:13px;">• {r}</span>' for r in result.reasons),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    elif go:
        st.warning("Enter a CVE ID first.")

# --------------------------------------------------------------------------
# Tab 2 — Report Analysis
# --------------------------------------------------------------------------
with tab_report:
    uploaded = st.file_uploader("Upload a report (PDF)", type=["pdf"])
    if uploaded and uploaded.name != st.session_state.report_name:
        with st.spinner("Extracting, chunking, and embedding report…"):
            text = extract_text(uploaded)
            chunks = chunk_text(text)
            index = build_index(chunks) if chunks else None
        st.session_state.report_chunks = chunks
        st.session_state.report_index = index
        st.session_state.report_name = uploaded.name
        st.session_state.chat_log = []
        st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

    if st.session_state.report_chunks:
        st.markdown(f'<p class="muted mono">Active report: {st.session_state.report_name}</p>', unsafe_allow_html=True)

        for role, msg in st.session_state.chat_log:
            with st.chat_message(role):
                st.markdown(msg)

        question = st.chat_input("Ask about this report (e.g. 'Extract every CVE mentioned')")
        if question:
            st.session_state.chat_log.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and asking Groq…"):
                    context = retrieve(question, st.session_state.report_chunks, st.session_state.report_index)
                    answer = ask_groq(question, context, mode=mode.lower())
                st.markdown(answer)
            st.session_state.chat_log.append(("assistant", answer))
    else:
        st.info("Upload a PDF report to start asking questions about it.")

# --------------------------------------------------------------------------
# Tab 3 — Dashboard
# --------------------------------------------------------------------------
with tab_dashboard:
    if not st.session_state.history:
        st.info("Investigate a CVE to start populating the dashboard.")
    else:
        counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for h in st.session_state.history:
            counts[h["priority"]] += 1

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val, color) in zip(
            [c1, c2, c3, c4],
            [("Critical", counts["Critical"], "#FF4757"), ("High", counts["High"], "#FF8C42"),
             ("Medium", counts["Medium"], "#FFB020"), ("Low", counts["Low"], "#2ED573")],
        ):
            col.markdown(
                f"""<div class="card" style="text-align:center; border-top:3px solid {color};">
                    <p class="muted" style="font-size:12px; letter-spacing:1px; text-transform:uppercase;">{label}</p>
                    <p style="font-family:'Chakra Petch',sans-serif; font-size:32px; margin:0;">{val}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("#### Severity distribution")
        st.bar_chart({"Count": counts}, horizontal=False)

        st.markdown("#### Investigation log")
        st.dataframe(
            [{"CVE": h["cve"], "Score": h["score"], "Priority": h["priority"], "Time": h["ts"]}
             for h in reversed(st.session_state.history)],
            use_container_width=True,
            hide_index=True,
        )
