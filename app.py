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
        <
