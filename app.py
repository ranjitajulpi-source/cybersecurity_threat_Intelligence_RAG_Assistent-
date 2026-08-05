"""
SIGNAL — Threat Intelligence Console
Project 29 — RAG + live threat-intel dashboard (Groq edition)

DEPLOY (Streamlit Community Cloud):
1. Push app.py + requirements.txt to a public GitHub repo.
2. streamlit.io -> New app -> repo -> main file: app.py
3. App settings -> Secrets:
     GROQ_API_KEY = "your-groq-key"        (required, free at console.groq.com/keys)
     ADMIN_PASSCODE = "choose-a-passcode"  (optional, default below if unset)

HONESTY NOTES on scope (so you know what's real vs simplified for a hackathon demo):
- Live feeds: NVD CVE API + CISA Known Exploited Vulnerabilities feed are REAL, free,
  no-key public APIs — data is genuinely live.
- "Real-time notifications" = checked on each page load/refresh (Streamlit has no
  server push), surfaced as a banner for KEV entries added in the last 7 days.
- "Role-based login" is a UX gate (passcode check), not production authentication —
  say so if judges ask.
- MITRE ATT&CK mapping uses a small illustrative CWE -> technique lookup table, not
  the full official CVE->ATT&CK dataset (that requires licensed/registered threat-intel
  sources) — labeled as illustrative in the UI.
- Knowledge graph links CVE <-> Vendor/Product <-> CWE, all sourced from NVD data.
  Malware/threat-actor nodes would need a paid/registered feed (e.g. OTX, MISP) —
  not wired up, noted in the UI as a future extension.
"""

import re
import html
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import fitz  # PyMuPDF
import faiss
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIGNAL — Threat Intelligence Console", page_icon="🛰️", layout="wide")

CHAT_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

CWE_ATTACK_MAP = {
    "CWE-79": [("T1189", "Drive-by Compromise"), ("T1059", "Command and Scripting Interpreter")],
    "CWE-89": [("T1190", "Exploit Public-Facing Application")],
    "CWE-78": [("T1059", "Command and Scripting Interpreter")],
    "CWE-352": [("T1189", "Drive-by Compromise")],
    "CWE-22": [("T1005", "Data from Local System")],
    "CWE-287": [("T1078", "Valid Accounts")],
    "CWE-798": [("T1078", "Valid Accounts")],
    "CWE-502": [("T1190", "Exploit Public-Facing Application")],
    "CWE-119": [("T1203", "Exploitation for Client Execution")],
    "CWE-120": [("T1203", "Exploitation for Client Execution")],
    "CWE-434": [("T1105", "Ingress Tool Transfer")],
    "CWE-306": [("T1190", "Exploit Public-Facing Application")],
    "CWE-611": [("T1190", "Exploit Public-Facing Application")],
    "CWE-918": [("T1190", "Exploit Public-Facing Application")],
}
DEFAULT_ATTACK = [("T1190", "Exploit Public-Facing Application")]

SEVERITY_COLORS = {"CRITICAL": "#FF4D5E", "HIGH": "#F5A524", "MEDIUM": "#F5E36B", "LOW": "#3ADC84", "UNKNOWN": "#6D7E90"}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root{
        --bg:#090D13; --panel:#0F1620; --panel-2:#121A25; --line:#1E2B3A;
        --text:#DCE6EE; --text-dim:#6D7E90;
        --cyan:#2FE2D0; --amber:#F5A524; --red:#FF4D5E; --green:#3ADC84;
    }
    html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
    .stApp{
        background:
            repeating-linear-gradient(0deg, rgba(47,226,208,0.025) 0px, rgba(47,226,208,0.025) 1px, transparent 1px, transparent 3px),
            radial-gradient(1200px 600px at 15% -10%, rgba(47,226,208,0.07), transparent 60%),
            var(--bg);
        color: var(--text);
    }
    section[data-testid="stSidebar"]{ background: var(--panel); border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
        font-family:'IBM Plex Mono', monospace; letter-spacing:0.1em; text-transform:uppercase;
        font-size:0.82rem; color:var(--cyan); border-bottom:1px solid var(--line); padding-bottom:0.4rem;
    }
    section[data-testid="stSidebar"] label{
        font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text-dim);
        text-transform:uppercase; letter-spacing:0.05em;
    }
    .console-header{
        display:flex; align-items:center; gap:18px; padding:20px 26px; margin-bottom:18px;
        background: linear-gradient(135deg, var(--panel-2) 0%, var(--panel) 100%);
        border:1px solid var(--line); border-left:3px solid var(--cyan); border-radius:4px;
    }
    .pulse-wrap{ position:relative; width:40px; height:40px; flex-shrink:0; }
    .pulse-dot{ position:absolute; top:50%; left:50%; width:9px; height:9px; margin:-4.5px 0 0 -4.5px;
        border-radius:50%; background:var(--cyan); box-shadow:0 0 12px 2px var(--cyan); }
    .pulse-ring{ position:absolute; inset:0; border-radius:50%; border:1.5px solid var(--cyan); animation:pulse-out 2.4s ease-out infinite; }
    .pulse-ring.delay{ animation-delay:1.2s; }
    @keyframes pulse-out{ 0%{transform:scale(0.2);opacity:0.9;} 100%{transform:scale(1.6);opacity:0;} }
    @media (prefers-reduced-motion: reduce){ .pulse-ring{ animation:none; opacity:0.35; } }
    .console-title{ font-family:'IBM Plex Mono', monospace; font-weight:700; font-size:1.5rem; letter-spacing:0.04em; margin:0; }
    .console-sub{ font-family:'IBM Plex Mono', monospace; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-dim); }
    .status-chip{ margin-left:auto; font-family:'IBM Plex Mono', monospace; font-size:0.7rem; letter-spacing:0.08em;
        padding:5px 12px; border-radius:20px; text-transform:uppercase; white-space:nowrap; }
    .status-online{ color:var(--green); border:1px solid rgba(58,220,132,0.4); background:rgba(58,220,132,0.08); }
    .status-offline{ color:var(--red); border:1px solid rgba(255,77,94,0.4); background:rgba(255,77,94,0.08); }
    .role-chip{ font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:0.08em; color:var(--cyan);
        border:1px solid var(--line); padding:3px 9px; border-radius:20px; text-transform:uppercase; margin-left:8px; }
    .kb-readout{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; color:var(--text-dim);
        border:1px dashed var(--line); border-radius:4px; padding:9px 14px; margin-bottom:12px; background:var(--panel-2); }
    .kb-readout b{ color:var(--cyan); }
    .alert-banner{ font-family:'IBM Plex Mono', monospace; font-size:0.8rem; color:#FFC168;
        border:1px solid rgba(245,165,36,0.5); background:rgba(245,165,36,0.08); border-radius:4px;
        padding:10px 16px; margin-bottom:14px; }
    .sev-chip{ display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:0.7rem; letter-spacing:0.06em;
        text-transform:uppercase; padding:3px 10px; border-radius:3px; font-weight:600; }
    .sev-critical{ color:#FF8A93; background:rgba(255,77,94,0.12); border:1px solid rgba(255,77,94,0.45); }
    .sev-high{ color:#FFC168; background:rgba(245,165,36,0.12); border:1px solid rgba(245,165,36,0.45); }
    .sev-medium{ color:#F5E36B; background:rgba(245,227,107,0.10); border:1px solid rgba(245,227,107,0.4); }
    .sev-low{ color:#7CEFA8; background:rgba(58,220,132,0.10); border:1px solid rgba(58,220,132,0.4); }
    .sev-unknown{ color:var(--text-dim); background:rgba(255,255,255,0.04); border:1px solid var(--line); }
    .conf-chip{ display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:0.05em;
        padding:2px 8px; border-radius:3px; border:1px solid var(--line); color:var(--text-dim); }
    div[data-testid="stChatMessage"]{ background:var(--panel-2); border:1px solid var(--line); border-radius:4px; }
    .log-time{ font-family:'IBM Plex Mono', monospace; font-size:0.66rem; color:var(--text-dim); margin-bottom:4px; }
    .stButton > button, .stDownloadButton > button{
        font-family:'IBM Plex Mono', monospace; letter-spacing:0.05em; text-transform:uppercase; font-size:0.76rem;
        background:transparent; color:var(--cyan); border:1px solid var(--cyan); border-radius:3px; transition:all .15s ease;
    }
    .stButton > button:hover{ background:rgba(47,226,208,0.12); color:#fff; border-color:#fff; }
    .stTextInput input, .stChatInput textarea, .stSelectbox div[data-baseweb="select"] > div{
        font-family:'IBM Plex Mono', monospace !important; background:var(--panel-2) !important; color:var(--text) !important;
        border:1px solid var(--line) !important;
    }
    section[data-testid="stFileUploaderDropzone"]{ background:var(--panel-2); border:1px dashed var(--line); border-radius:4px; }
    div[data-testid="stMetric"]{ background:var(--panel-2); border:1px solid var(--line); border-radius:4px; padding:10px 14px; }
    div[data-testid="stMetricLabel"]{ font-family:'IBM Plex Mono', monospace; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.06em; color:var(--text-dim); }
    .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"]{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; text-transform:uppercase;
        letter-spacing:0.06em; color:var(--text-dim); background:transparent; }
    .stTabs [aria-selected="true"]{ color:var(--cyan) !important; border-bottom:2px solid var(--cyan) !important; }
    hr{ border-color:var(--line) !important; }
    .section-label{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--cyan); margin:4px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor="#0F1620", plot_bgcolor="#0F1620",
    font=dict(family="IBM Plex Mono, monospace", color="#DCE6EE", size=12),
    colorway=["#2FE2D0", "#F5A524", "#FF4D5E", "#3ADC84", "#6D7E90"],
    xaxis=dict(gridcolor="#1E2B3A"), yaxis=dict(gridcolor="#1E2B3A"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ---------------------------------------------------------------------------
# Resources & session state
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedder()
api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
admin_passcode = st.secrets.get("ADMIN_PASSCODE", "signal-admin") if hasattr(st, "secrets") else "signal-admin"

defaults = {
    "index": None, "chunks": [], "sources": [], "messages": [],
    "bookmarks": {}, "role": "Analyst", "is_admin": False,
    "search_results": [], "compare_ids": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Data helpers — NVD + CISA KEV
# ---------------------------------------------------------------------------
def _severity_from_score(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    return "LOW"


def parse_cve_item(cve: dict) -> dict:
    cve_id = cve.get("id", "UNKNOWN")
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
    metrics = cve.get("metrics", {})
    cvss, vector = None, ""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            cvss = metrics[key][0]["cvssData"].get("baseScore")
            vector = metrics[key][0]["cvssData"].get("vectorString", "")
            break
    cwes = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            if d.get("value", "").startswith("CWE-"):
                cwes.append(d["value"])
    refs = [{"url": r.get("url", ""), "tags": r.get("tags", [])} for r in cve.get("references", [])]
    vendors, products = set(), set()
    for node in cve.get("configurations", []):
        for n in node.get("nodes", []):
            for m in n.get("cpeMatch", []):
                parts = m.get("criteria", "").split(":")
                if len(parts) > 4:
                    vendors.add(parts[3])
                    products.add(parts[4])
    return {
        "id": cve_id, "description": desc, "cvss": cvss, "vector": vector,
        "severity": _severity_from_score(cvss), "published": cve.get("published", "")[:10],
        "cwes": cwes or ["N/A"], "references": refs,
        "vendors": sorted(vendors)[:5], "products": sorted(products)[:5],
    }


@st.cache_data(ttl=600, show_spinner=False)
def nvd_search(keyword: str = "", severity: str = "", days_back: int = 0, results: int = 20):
    params = {"resultsPerPage": results}
    if keyword:
        params["keywordSearch"] = keyword
    if severity and severity != "Any":
        params["cvssV3Severity"] = severity.upper()
    if days_back:
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)
        params["pubStartDate"] = start.strftime("%Y-%m-%dT%H:%M:%S.000")
        params["pubEndDate"] = end.strftime("%Y-%m-%dT%H:%M:%S.000")
    r = requests.get(NVD_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return [parse_cve_item(v["cve"]) for v in data.get("vulnerabilities", [])]


@st.cache_data(ttl=600, show_spinner=False)
def nvd_get_single(cve_id: str):
    r = requests.get(NVD_BASE, params={"cveId": cve_id.strip().upper()}, timeout=15)
    r.raise_for_status()
    vulns = r.json().get("vulnerabilities", [])
    if not vulns:
        return None
    return parse_cve_item(vulns[0]["cve"])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_kev(limit: int = 60):
    r = requests.get(CISA_KEV_URL, timeout=20)
    r.raise_for_status()
    vulns = r.json().get("vulnerabilities", [])
    vulns = sorted(vulns, key=lambda v: v.get("dateAdded", ""), reverse=True)
    return vulns[:limit]


def severity_chip_html(sev: str):
    sev = (sev or "UNKNOWN").upper()
    cls = {"CRITICAL": "sev-critical", "HIGH": "sev-high", "MEDIUM": "sev-medium", "LOW": "sev-low"}.get(sev, "sev-unknown")
    return f'<span class="sev-chip {cls}">{sev.title()}</span>'


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def embed_texts(texts):
    return np.array(embedder.encode(texts, normalize_embeddings=True), dtype="float32")


def embed_query(query: str):
    return np.array(embedder.encode([query], normalize_embeddings=True), dtype="float32")


def add_to_index(text: str, source_label: str):
    new_chunks = chunk_text(text)
    if not new_chunks:
        return 0
    vectors = embed_texts(new_chunks)
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatIP(vectors.shape[1])
    st.session_state.index.add(vectors)
    st.session_state.chunks.extend(new_chunks)
    st.session_state.sources.extend([source_label] * len(new_chunks))
    return len(new_chunks)


def retrieve(query: str, k=TOP_K):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    qvec = embed_query(query)
    scores, idxs = st.session_state.index.search(qvec, min(k, st.session_state.index.ntotal))
    out = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        out.append((st.session_state.chunks[i], st.session_state.sources[i], float(score)))
    return out


def answer_question(groq_client: Groq, query: str, context_triples):
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src, _ in context_triples)
    prompt = f"""You are a cyber security analyst assistant. Answer ONLY using the context below.
If the context doesn't contain the answer, say so rather than guessing. Cite the source label
for each claim.

Context:
{context}

Question: {query}

Answer:"""
    completion = groq_client.chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2,
    )
    return completion.choices[0].message.content


def confidence_label(avg_score: float):
    pct = max(0, min(100, round(avg_score * 100)))
    if pct >= 65:
        return pct, "High", "var(--green)"
    if pct >= 35:
        return pct, "Medium", "var(--amber)"
    return pct, "Low", "var(--red)"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Sidebar — control panel, role gate, ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Identity")
    role_choice = st.radio("Role", ["Analyst", "Administrator"], horizontal=True, label_visibility="collapsed")
    if role_choice == "Administrator":
        pw = st.text_input("Admin passcode", type="password")
        st.session_state.is_admin = pw == admin_passcode
        if pw and not st.session_state.is_admin:
            st.error("Incorrect passcode.")
    else:
        st.session_state.is_admin = False
    st.session_state.role = role_choice if (role_choice == "Analyst" or st.session_state.is_admin) else "Analyst"

    st.markdown("### Uplink")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="console.groq.com/keys")
    client = Groq(api_key=api_key) if api_key else None
    st.markdown(
        f'<div class="kb-readout" style="border-color:{"rgba(58,220,132,0.4)" if client else "rgba(255,77,94,0.4)"};">'
        f'{"● Groq inference online" if client else "● Awaiting Groq API key"}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Ingest sources")
    uploaded_files = st.file_uploader("Advisories / reports (PDF)", type=["pdf"], accept_multiple_files=True)
    cve_id_input = st.text_input("Fetch CVE by ID", placeholder="CVE-2024-3400")
    fetch_cve_btn = st.button("Add CVE to knowledge base", use_container_width=True)

    st.divider()
    if st.session_state.is_admin:
        if st.button("Clear knowledge base (admin)", use_container_width=True):
            st.session_state.index, st.session_state.chunks, st.session_state.sources = None, [], []
            st.success("Knowledge base cleared.")
    else:
        st.caption("Sign in as Administrator to clear the knowledge base.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_html = '<span class="status-chip status-online">● Armed</span>' if client else '<span class="status-chip status-offline">● Standby</span>'
st.markdown(
    f"""
    <div class="console-header">
        <div class="pulse-wrap"><div class="pulse-ring"></div><div class="pulse-ring delay"></div><div class="pulse-dot"></div></div>
        <div>
            <p class="console-title">SIGNAL <span class="role-chip">{st.session_state.role}</span></p>
            <p class="console-sub">Threat Intelligence Console · live feeds + RAG analysis</p>
        </div>
        {status_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Ingest actions
# ---------------------------------------------------------------------------
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.sources:
            with st.spinner(f"Parsing {f.name}..."):
                n = add_to_index(extract_pdf_text(f.read()), f.name)
            st.sidebar.success(f"+{n} chunks — {f.name}")

if fetch_cve_btn:
    if not cve_id_input.strip():
        st.sidebar.error("Enter a CVE ID first.")
    else:
        with st.spinner("Querying NVD..."):
            try:
                rec = nvd_get_single(cve_id_input)
            except Exception as e:
                rec = None
                st.sidebar.error(f"NVD lookup failed: {e}")
        if rec:
            text = f"CVE ID: {rec['id']}\nCVSS: {rec['cvss']} ({rec['severity']})\nDescription: {rec['description']}\nCWEs: {', '.join(rec['cwes'])}"
            n = add_to_index(text, rec["id"])
            st.session_state.bookmarks.setdefault(rec["id"], rec)
            st.sidebar.markdown(f'<div class="kb-readout">+{n} chunks — <b>{rec["id"]}</b> {severity_chip_html(rec["severity"])}</div>', unsafe_allow_html=True)
        elif rec is None:
            st.sidebar.warning("No data found for that CVE ID.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_feed, tab_chat, tab_search, tab_attack, tab_graph, tab_reports = st.tabs(
    ["📊 Dashboard", "📡 Threat Feed", "🤖 AI Chat", "🔍 Search & Compare", "🗺️ ATT&CK Map", "🕸️ Knowledge Graph", "📁 Reports"]
)

# ---- DASHBOARD -------------------------------------------------------------
with tab_dash:
    st.markdown('<p class="section-label">// System Overview</p>', unsafe_allow_html=True)
    try:
        kev = fetch_kev(80)
    except Exception as e:
        kev = []
        st.warning(f"CISA KEV feed unavailable right now ({e}).")
    try:
        recent = nvd_search(days_back=30, results=50)
    except Exception as e:
        recent = []
        st.warning(f"NVD feed unavailable right now ({e}).")

    recent_kev_alerts = [v for v in kev if v.get("dateAdded", "") >= (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")]
    if recent_kev_alerts:
        st.markdown(
            f'<div class="alert-banner">🚨 <b>{len(recent_kev_alerts)}</b> new actively-exploited vulnerabilities added to CISA KEV in the last 7 days</div>',
            unsafe_allow_html=True,
        )

    sev_counts = pd.Series([c["severity"] for c in recent]).value_counts() if recent else pd.Series(dtype=int)
    ransomware_count = sum(1 for v in kev if v.get("knownRansomwareCampaignUse", "") == "Known")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KEV entries tracked", len(kev))
    c2.metric("Linked to ransomware", ransomware_count)
    c3.metric("Critical (30d, NVD)", int(sev_counts.get("CRITICAL", 0)))
    c4.metric("Indexed KB chunks", len(st.session_state.chunks))

    col1, col2 = st.columns([1, 1.3])
    with col1:
        if not sev_counts.empty:
            fig = px.pie(
                names=sev_counts.index, values=sev_counts.values, hole=0.55,
                color=sev_counts.index, color_discrete_map=SEVERITY_COLORS, template=PLOTLY_TEMPLATE,
                title="Severity distribution — CVEs published, last 30 days",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent NVD data to chart.")
    with col2:
        if kev:
            vendor_counts = pd.Series([v.get("vendorProject", "Unknown") for v in kev]).value_counts().head(10)
            fig = px.bar(
                x=vendor_counts.values, y=vendor_counts.index, orientation="h",
                template=PLOTLY_TEMPLATE, title="Top vendors — CISA Known Exploited Vulnerabilities",
                labels={"x": "KEV entries", "y": ""},
            )
            fig.update_traces(marker_color="#2FE2D0")
            st.plotly_chart(fig, use_container_width=True)

    if kev:
        kev_df = pd.DataFrame(kev)
        kev_df["dateAdded"] = pd.to_datetime(kev_df["dateAdded"])
        daily = kev_df.groupby(kev_df["dateAdded"].dt.date).size().reset_index(name="count")
        fig = px.line(daily, x="dateAdded", y="count", template=PLOTLY_TEMPLATE, title="KEV additions over time (trend)")
        fig.update_traces(line_color="#2FE2D0")
        st.plotly_chart(fig, use_container_width=True)

    if recent:
        heat_df = pd.DataFrame(recent)
        heat_df["published"] = pd.to_datetime(heat_df["published"], errors="coerce")
        heat_df["week"] = heat_df["published"].dt.strftime("W%U")
        pivot = heat_df.pivot_table(index="severity", columns="week", values="id", aggfunc="count", fill_value=0)
        if not pivot.empty:
            fig = px.imshow(pivot, template=PLOTLY_TEMPLATE, aspect="auto", color_continuous_scale="Tealgrn", title="Severity heatmap by week (30d)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-label">// Recent CVEs</p>', unsafe_allow_html=True)
    if recent:
        for rec in recent[:8]:
            st.markdown(
                f"**{rec['id']}** {severity_chip_html(rec['severity'])} · CVSS {rec['cvss'] or 'N/A'} · {rec['published']}  \n"
                f"{rec['description'][:220]}{'...' if len(rec['description']) > 220 else ''}",
                unsafe_allow_html=True,
            )
    else:
        st.info("No recent CVE data available.")

# ---- THREAT FEED ------------------------------------------------------------
with tab_feed:
    st.markdown('<p class="section-label">// Live Feed — CISA Known Exploited Vulnerabilities</p>', unsafe_allow_html=True)
    try:
        kev = fetch_kev(60)
    except Exception as e:
        kev = []
        st.error(f"Feed unavailable: {e}")
    if kev:
        df = pd.DataFrame(kev)[["cveID", "vendorProject", "product", "vulnerabilityName", "dateAdded", "dueDate", "knownRansomwareCampaignUse"]]
        df.columns = ["CVE", "Vendor", "Product", "Name", "Added", "Patch due", "Ransomware use"]
        st.dataframe(df, use_container_width=True, height=420)
        st.caption("Source: CISA Known Exploited Vulnerabilities catalog (cisa.gov)")

    st.markdown('<p class="section-label">// Live Feed — NVD Recent Publications</p>', unsafe_allow_html=True)
    kw = st.text_input("Filter recent NVD feed by keyword (optional)", key="feed_kw")
    try:
        feed_recent = nvd_search(keyword=kw, days_back=14, results=30)
    except Exception as e:
        feed_recent = []
        st.error(f"NVD feed unavailable: {e}")
    for rec in feed_recent[:15]:
        st.markdown(
            f"**[{rec['id']}](https://nvd.nist.gov/vuln/detail/{rec['id']})** {severity_chip_html(rec['severity'])} · {rec['published']}  \n"
            f"{rec['description'][:200]}{'...' if len(rec['description']) > 200 else ''}",
            unsafe_allow_html=True,
        )

# ---- AI CHAT ------------------------------------------------------------
with tab_chat:
    st.markdown('<p class="section-label">// Query Log — AI Analyst Chat</p>', unsafe_allow_html=True)
    if st.session_state.chunks:
        st.markdown(
            f'<div class="kb-readout">📡 <b>{len(st.session_state.chunks)}</b> chunks indexed from '
            f'<b>{len(set(st.session_state.sources))}</b> source(s)</div>', unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="kb-readout">📡 Knowledge base empty — add a PDF or CVE from the control panel</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f'<div class="log-time">{msg.get("time", "")}</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("confidence") is not None:
                pct, label, color = msg["confidence"]
                st.markdown(f'<span class="conf-chip" style="color:{color};border-color:{color};">Confidence: {label} ({pct}%)</span>', unsafe_allow_html=True)

    query = st.chat_input("e.g. Is CVE-2024-3400 being actively exploited, and what's the mitigation?")
    if query:
        if not client:
            st.error("Enter a Groq API key in the control panel first.")
        elif not st.session_state.chunks:
            st.error("Add at least one document or CVE before asking a question.")
        else:
            now = timestamp()
            st.session_state.messages.append({"role": "user", "content": query, "time": now})
            with st.chat_message("user"):
                st.markdown(f'<div class="log-time">{now}</div>', unsafe_allow_html=True)
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Scanning knowledge base..."):
                    results = retrieve(query)
                    answer = answer_question(client, query, results)
                    avg_score = float(np.mean([s for _, _, s in results])) if results else 0.0
                    pct, label, color = confidence_label(avg_score)
                    now2 = timestamp()
                    st.markdown(f'<div class="log-time">{now2}</div>', unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown(f'<span class="conf-chip" style="color:{color};border-color:{color};">Confidence: {label} ({pct}%)</span>', unsafe_allow_html=True)
                    with st.expander("🔎 Explainability — why this answer"):
                        st.caption("Retrieved chunks ranked by cosine similarity to your question. The model was instructed to answer only from these.")
                        for chunk, src, score in results:
                            st.markdown(f"**{src}** · similarity `{score:.2f}`")
                            st.caption(chunk[:280] + ("..." if len(chunk) > 280 else ""))
            st.session_state.messages.append({"role": "assistant", "content": answer, "time": now2, "confidence": (pct, label, color)})

# ---- SEARCH & COMPARE ------------------------------------------------------------
with tab_search:
    st.markdown('<p class="section-label">// Advanced Search</p>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    kw = sc1.text_input("Keyword / vendor / product")
    sev_filter = sc2.selectbox("Severity", ["Any", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    days = sc3.slider("Published within (days)", 0, 365, 90)
    n_results = sc4.slider("Max results", 5, 50, 20)

    if st.button("Run search"):
        with st.spinner("Querying NVD..."):
            try:
                st.session_state.search_results = nvd_search(keyword=kw, severity=sev_filter, days_back=days, results=n_results)
            except Exception as e:
                st.session_state.search_results = []
                st.error(f"Search failed: {e}")

    results = st.session_state.search_results
    if results:
        df = pd.DataFrame(results)[["id", "severity", "cvss", "published", "description"]]
        df.columns = ["CVE", "Severity", "CVSS", "Published", "Description"]
        st.dataframe(df, use_container_width=True, height=320)

        ids = [r["id"] for r in results]
        chosen = st.multiselect("Select CVEs to bookmark or compare", ids)
        bc1, bc2 = st.columns(2)
        if bc1.button("⭐ Bookmark selected"):
            for r in results:
                if r["id"] in chosen:
                    st.session_state.bookmarks[r["id"]] = r
            st.success(f"Bookmarked {len(chosen)} CVE(s).")
        if bc2.button("⚖️ Add to comparison"):
            st.session_state.compare_ids = list(set(st.session_state.compare_ids + chosen))
            st.success(f"Comparison set: {', '.join(st.session_state.compare_ids)}")
    else:
        st.info("Run a search to see results here.")

    st.markdown('<p class="section-label">// CVE Comparison</p>', unsafe_allow_html=True)
    if st.session_state.compare_ids:
        pool = {r["id"]: r for r in results}
        pool.update(st.session_state.bookmarks)
        rows = [pool[i] for i in st.session_state.compare_ids if i in pool]
        if rows:
            comp_df = pd.DataFrame(rows)[["id", "severity", "cvss", "published", "cwes", "vendors", "description"]]
            comp_df.columns = ["CVE", "Severity", "CVSS", "Published", "CWE(s)", "Vendors", "Description"]
            st.dataframe(comp_df, use_container_width=True)
        if st.button("Clear comparison set"):
            st.session_state.compare_ids = []
    else:
        st.caption("Select CVEs above and click \"Add to comparison\" to compare them side by side.")

# ---- ATT&CK MAP ------------------------------------------------------------
with tab_attack:
    st.markdown('<p class="section-label">// MITRE ATT&CK Mapping (illustrative)</p>', unsafe_allow_html=True)
    st.caption("Mapped from each CVE's CWE weakness type to a small illustrative CWE→technique lookup table — not the full official MITRE dataset.")
    pool = {**st.session_state.bookmarks, **{r["id"]: r for r in st.session_state.search_results}}
    if not pool:
        st.info("Bookmark or search a CVE first to see its ATT&CK mapping.")
    else:
        pick = st.selectbox("CVE", list(pool.keys()))
        rec = pool[pick]
        st.markdown(f"**{rec['id']}** {severity_chip_html(rec['severity'])} · CVSS {rec['cvss'] or 'N/A'}", unsafe_allow_html=True)
        st.write(rec["description"])
        rows = []
        for cwe in rec["cwes"]:
            techniques = CWE_ATTACK_MAP.get(cwe, DEFAULT_ATTACK if cwe != "N/A" else [])
            for tid, tname in techniques:
                rows.append({"CWE": cwe, "ATT&CK Technique": f"{tid} — {tname}"})
        if rows:
            st.table(pd.DataFrame(rows))
        else:
            st.info("No CWE data available on this CVE to map.")

# ---- KNOWLEDGE GRAPH ------------------------------------------------------------
with tab_graph:
    st.markdown('<p class="section-label">// Interactive Knowledge Graph</p>', unsafe_allow_html=True)
    st.caption("CVE ↔ Vendor/Product ↔ CWE relationships, built from NVD data for your bookmarked/searched CVEs. "
               "Malware/threat-actor nodes need a registered threat-intel feed (e.g. OTX, MISP) — not connected here.")
    pool = {**st.session_state.bookmarks, **{r["id"]: r for r in st.session_state.search_results}}
    if not pool:
        st.info("Bookmark or search CVEs to build the graph.")
    else:
        nodes, edges = {}, []
        for rec in pool.values():
            nodes[rec["id"]] = "cve"
            for v in rec.get("vendors", []):
                nodes[v] = "vendor"
                edges.append((rec["id"], v))
            for cwe in rec.get("cwes", []):
                if cwe != "N/A":
                    nodes[cwe] = "cwe"
                    edges.append((rec["id"], cwe))

        node_list = list(nodes.keys())
        n = len(node_list)
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {node_list[i]: (np.cos(angle[i]), np.sin(angle[i])) for i in range(n)}

        edge_x, edge_y = [], []
        for a, b in edges:
            if a in pos and b in pos:
                edge_x += [pos[a][0], pos[b][0], None]
                edge_y += [pos[a][1], pos[b][1], None]

        color_map = {"cve": "#2FE2D0", "vendor": "#F5A524", "cwe": "#FF4D5E"}
        node_x = [pos[k][0] for k in node_list]
        node_y = [pos[k][1] for k in node_list]
        node_color = [color_map[nodes[k]] for k in node_list]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#1E2B3A", width=1), hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_list, textposition="top center",
            marker=dict(size=16, color=node_color, line=dict(width=1, color="#0F1620")),
            textfont=dict(size=10, color="#DCE6EE"),
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=520,
                           xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 CVE   🟠 Vendor/Product   🔴 CWE weakness")

# ---- REPORTS ------------------------------------------------------------
with tab_reports:
    st.markdown('<p class="section-label">// Bookmarked Investigations</p>', unsafe_allow_html=True)
    bms = list(st.session_state.bookmarks.values())
    if not bms:
        st.info("No bookmarks yet — bookmark CVEs from the Search & Compare tab.")
    else:
        df = pd.DataFrame(bms)[["id", "severity", "cvss", "published", "description"]]
        df.columns = ["CVE", "Severity", "CVSS", "Published", "Description"]
        st.dataframe(df, use_container_width=True)

        st.markdown('<p class="section-label">// Mitigation & Patch Recommendations</p>', unsafe_allow_html=True)
        for rec in bms:
            patch_refs = [r["url"] for r in rec.get("references", []) if "Patch" in r.get("tags", []) or "Vendor Advisory" in r.get("tags", [])]
            st.markdown(f"**{rec['id']}** {severity_chip_html(rec['severity'])}", unsafe_allow_html=True)
            if patch_refs:
                for u in patch_refs[:3]:
                    st.markdown(f"- [{u}]({u})")
            else:
                st.caption("No vendor patch link indexed by NVD — check the vendor's own advisory page.")

        col1, col2 = st.columns(2)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        col1.download_button("⬇️ Export CSV", csv_bytes, file_name="signal_bookmarks.csv", mime="text/csv", use_container_width=True)

        def build_pdf(records):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "SIGNAL — Investigation Report", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(4)
            for rec in records:
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 7, f"{rec['id']}  [{rec['severity']}]  CVSS {rec['cvss'] or 'N/A'}")
                pdf.set_font("Helvetica", "", 10)
                desc = rec["description"].encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 6, desc)
                pdf.ln(3)
            return bytes(pdf.output(dest="S"))

        pdf_bytes = build_pdf(bms)
        col2.download_button("⬇️ Export PDF",pdf_bytes, file_name="signal_report.pdf", mime="application/pdf", use_container_width=True)
