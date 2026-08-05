"""
SIGNAL — Threat Intelligence Console
Project 29 — RAG-based cyber security Q&A system (Groq edition)

Deploy on Streamlit Community Cloud (share.streamlit.io):
1. Push app.py + requirements.txt to a public GitHub repo.
2. On streamlit.io, "New app" -> point to the repo -> main file: app.py
3. In App settings -> Secrets, add:
   GROQ_API_KEY = "your-key-here"
   (Get a free key at https://console.groq.com/keys)

Note: Groq serves inference only, no embeddings API, so this app embeds
locally with sentence-transformers (all-MiniLM-L6-v2) and uses Groq only
for the final answer generation.
"""

import re
import html
from datetime import datetime

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIGNAL — Threat Intelligence Console", page_icon="🛰️", layout="wide")

CHAT_MODEL = "openai/gpt-oss-120b"  # current Groq model (llama-3.3-70b-versatile is deprecated)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

# ---------------------------------------------------------------------------
# Console theme — dark SOC-terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root{
        --bg:#090D13;
        --panel:#0F1620;
        --panel-2:#121A25;
        --line:#1E2B3A;
        --text:#DCE6EE;
        --text-dim:#6D7E90;
        --cyan:#2FE2D0;
        --amber:#F5A524;
        --red:#FF4D5E;
        --green:#3ADC84;
    }

    html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
    code, .stMarkdown pre, .mono { font-family:'IBM Plex Mono', monospace; }

    .stApp{
        background:
            repeating-linear-gradient(0deg, rgba(47,226,208,0.025) 0px, rgba(47,226,208,0.025) 1px, transparent 1px, transparent 3px),
            radial-gradient(1200px 600px at 15% -10%, rgba(47,226,208,0.07), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    /* ---- Sidebar / control panel ---- */
    section[data-testid="stSidebar"]{
        background: var(--panel);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
        font-family:'IBM Plex Mono', monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.85rem;
        color: var(--cyan);
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] label{
        font-family:'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ---- Header banner ---- */
    .console-header{
        display:flex; align-items:center; gap:18px;
        padding: 22px 26px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--line);
        border-left: 3px solid var(--cyan);
        border-radius: 4px;
    }
    .pulse-wrap{ position:relative; width:44px; height:44px; flex-shrink:0; }
    .pulse-dot{
        position:absolute; top:50%; left:50%; width:10px; height:10px;
        margin:-5px 0 0 -5px; border-radius:50%; background: var(--cyan);
        box-shadow: 0 0 12px 2px var(--cyan);
    }
    .pulse-ring{
        position:absolute; inset:0; border-radius:50%;
        border: 1.5px solid var(--cyan);
        animation: pulse-out 2.4s ease-out infinite;
    }
    .pulse-ring.delay{ animation-delay: 1.2s; }
    @keyframes pulse-out{
        0%   { transform: scale(0.2); opacity: 0.9; }
        100% { transform: scale(1.6); opacity: 0; }
    }
    @media (prefers-reduced-motion: reduce){
        .pulse-ring{ animation: none; opacity: 0.35; }
    }
    .console-title{ font-family:'IBM Plex Mono', monospace; font-weight:700; font-size:1.6rem; letter-spacing:0.04em; color:var(--text); margin:0; }
    .console-sub{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--text-dim); margin-top:2px; }
    .status-chip{
        margin-left:auto; font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.08em;
        padding:5px 12px; border-radius:20px; text-transform:uppercase; white-space:nowrap;
    }
    .status-online{ color:var(--green); border:1px solid rgba(58,220,132,0.4); background:rgba(58,220,132,0.08); }
    .status-offline{ color:var(--red); border:1px solid rgba(255,77,94,0.4); background:rgba(255,77,94,0.08); }

    /* ---- KB status readout ---- */
    .kb-readout{
        font-family:'IBM Plex Mono', monospace; font-size:0.8rem; color: var(--text-dim);
        border:1px dashed var(--line); border-radius:4px; padding:10px 14px; margin-bottom:14px;
        background: var(--panel-2);
    }
    .kb-readout b{ color: var(--cyan); }

    /* ---- Severity chip ---- */
    .sev-chip{
        display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
        letter-spacing:0.08em; text-transform:uppercase; padding:3px 10px; border-radius:3px; font-weight:600;
    }
    .sev-critical{ color:#FF8A93; background:rgba(255,77,94,0.12); border:1px solid rgba(255,77,94,0.45); }
    .sev-high{ color:#FFC168; background:rgba(245,165,36,0.12); border:1px solid rgba(245,165,36,0.45); }
    .sev-medium{ color:#F5E36B; background:rgba(245,227,107,0.10); border:1px solid rgba(245,227,107,0.4); }
    .sev-low{ color:#7CEFA8; background:rgba(58,220,132,0.10); border:1px solid rgba(58,220,132,0.4); }
    .sev-unknown{ color:var(--text-dim); background:rgba(255,255,255,0.04); border:1px solid var(--line); }

    /* ---- Chat log styling ---- */
    div[data-testid="stChatMessage"]{
        background: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    div[data-testid="stChatMessageAvatarUser"]{ background: var(--cyan) !important; }
    div[data-testid="stChatMessageAvatarAssistant"]{ background: var(--panel) !important; border:1px solid var(--cyan); }
    .log-time{ font-family:'IBM Plex Mono', monospace; font-size:0.68rem; color:var(--text-dim); letter-spacing:0.05em; margin-bottom:4px; }

    /* ---- Buttons ---- */
    .stButton > button, .stDownloadButton > button{
        font-family:'IBM Plex Mono', monospace; letter-spacing:0.06em; text-transform:uppercase; font-size:0.78rem;
        background: transparent; color: var(--cyan); border: 1px solid var(--cyan); border-radius:3px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover{ background: rgba(47,226,208,0.12); color:#fff; border-color:#fff; }

    /* ---- Inputs ---- */
    .stTextInput input, .stChatInput textarea{
        font-family:'IBM Plex Mono', monospace !important;
        background: var(--panel-2) !important; color: var(--text) !important;
        border: 1px solid var(--line) !important;
    }

    /* ---- File uploader ---- */
    section[data-testid="stFileUploaderDropzone"]{
        background: var(--panel-2); border: 1px dashed var(--line); border-radius:4px;
    }

    /* ---- Divider ---- */
    hr{ border-color: var(--line) !important; }

    /* ---- Section label ---- */
    .section-label{
        font-family:'IBM Plex Mono', monospace; font-size:0.78rem; letter-spacing:0.12em;
        text-transform:uppercase; color:var(--cyan); margin: 4px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedder()

# ---------------------------------------------------------------------------
# API key setup
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in (("index", None), ("chunks", []), ("sources", []), ("messages", [])):
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — control panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Setup")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com/keys")
    client = Groq(api_key=api_key) if api_key else None
    if api_key:
        st.markdown(
            '<div class="kb-readout" style="border-color:rgba(58,220,132,0.4);">● <b>Link established</b> — Groq inference online</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="kb-readout" style="border-color:rgba(255,77,94,0.4);">● <b>Awaiting key</b> — enter a Groq API key to arm the console</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Ingest sources")
    uploaded_files = st.file_uploader(
        "Security advisories / reports (PDF)", type=["pdf"], accept_multiple_files=True
    )

    st.markdown("**Live CVE lookup — NVD**")
    cve_id = st.text_input("CVE ID", placeholder="CVE-2024-3400")
    fetch_cve = st.button("Fetch CVE", use_container_width=True)

    st.divider()
    if st.button("Clear knowledge base", use_container_width=True):
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.sources = []
        st.success("Knowledge base cleared.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_html = (
    '<span class="status-chip status-online">● Console Armed</span>'
    if client
    else '<span class="status-chip status-offline">● Standby</span>'
)
st.markdown(
    f"""
    <div class="console-header">
        <div class="pulse-wrap">
            <div class="pulse-ring"></div>
            <div class="pulse-ring delay"></div>
            <div class="pulse-dot"></div>
        </div>
        <div>
            <p class="console-title">SIGNAL</p>
            <p class="console-sub">Threat Intelligence Console · RAG-grounded analysis</p>
        </div>
        {status_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def embed_texts(texts):
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return np.array(vectors, dtype="float32")


def embed_query(query: str):
    vector = embedder.encode([query], normalize_embeddings=True)
    return np.array(vector, dtype="float32")


def add_to_index(text: str, source_label: str):
    new_chunks = chunk_text(text)
    if not new_chunks:
        return 0
    vectors = embed_texts(new_chunks)
    dim = vectors.shape[1]
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatIP(dim)
    st.session_state.index.add(vectors)
    st.session_state.chunks.extend(new_chunks)
    st.session_state.sources.extend([source_label] * len(new_chunks))
    return len(new_chunks)


def fetch_cve_from_nvd(cve_id: str):
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.strip().upper()}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None
    cve = vulns[0]["cve"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
    metrics = cve.get("metrics", {})
    cvss = "N/A"
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            cvss = metrics[key][0]["cvssData"]["baseScore"]
            break
    refs = [r_["url"] for r_ in cve.get("references", [])][:5]
    text = f"CVE ID: {cve_id.upper()}\nCVSS Base Score: {cvss}\nDescription: {desc}\nReferences: {', '.join(refs)}"
    return text, cvss


def severity_chip(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return '<span class="sev-chip sev-unknown">Unknown</span>'
    if score >= 9.0:
        return '<span class="sev-chip sev-critical">Critical</span>'
    if score >= 7.0:
        return '<span class="sev-chip sev-high">High</span>'
    if score >= 4.0:
        return '<span class="sev-chip sev-medium">Medium</span>'
    return '<span class="sev-chip sev-low">Low</span>'


def retrieve(query: str, k=TOP_K):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    qvec = embed_query(query)
    _, idxs = st.session_state.index.search(qvec, min(k, st.session_state.index.ntotal))
    results = []
    for i in idxs[0]:
        if i == -1:
            continue
        results.append((st.session_state.chunks[i], st.session_state.sources[i]))
    return results


def answer_question(groq_client: Groq, query: str, context_pairs):
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src in context_pairs)
    prompt = f"""You are a cyber security analyst assistant. Answer the question ONLY using the
context below. If the context doesn't contain the answer, say so clearly rather than guessing.
Cite the source label for each claim you make.

Context:
{context}

Question: {query}

Answer:"""
    completion = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Ingest uploads
# ---------------------------------------------------------------------------
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.sources:
            with st.spinner(f"Parsing {f.name}..."):
                text = extract_pdf_text(f.read())
                n = add_to_index(text, f.name)
            st.sidebar.success(f"+{n} chunks indexed — {f.name}")

if fetch_cve:
    if not cve_id.strip():
        st.sidebar.error("Enter a CVE ID first.")
    else:
        with st.spinner(f"Querying NVD for {cve_id}..."):
            try:
                result = fetch_cve_from_nvd(cve_id)
            except Exception as e:
                result = None
                st.sidebar.error(f"NVD lookup failed: {e}")
        if result:
            text, cvss = result
            n = add_to_index(text, cve_id.upper())
            st.sidebar.markdown(
                f'<div class="kb-readout">+{n} chunks — <b>{html.escape(cve_id.upper())}</b> · {severity_chip(cvss)}</div>',
                unsafe_allow_html=True,
            )
        elif result is None and cve_id.strip():
            st.sidebar.warning("No data found for that CVE ID.")

# ---------------------------------------------------------------------------
# Knowledge base readout
# ---------------------------------------------------------------------------
if st.session_state.chunks:
    st.markdown(
        f'<div class="kb-readout">📡 <b>{len(st.session_state.chunks)}</b> chunks indexed '
        f'from <b>{len(set(st.session_state.sources))}</b> source(s)</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="kb-readout">📡 Knowledge base empty — upload an advisory or fetch a CVE from the control panel</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Chat log
# ---------------------------------------------------------------------------
st.markdown('<p class="section-label">// Query Log</p>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="log-time">{msg.get("time", "")}</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])

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
                now2 = timestamp()
                st.markdown(f'<div class="log-time">{now2}</div>', unsafe_allow_html=True)
                st.markdown(answer)
                with st.expander("📎 Sources referenced"):
                    for chunk, src in results:
                        st.markdown(f"**{src}**")
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer, "time": now2})
