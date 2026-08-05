"""
Cyber Security RAG Assistant
Project 29 — RAG-based threat intelligence Q&A system (Groq edition)

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
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cyber Security RAG Assistant", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Security Threat Intelligence RAG Assistant")
st.caption("Upload security advisories / CVE reports or pull a live CVE, then ask questions grounded in that data.")

CHAT_MODEL = "openai/gpt-oss-120b"  # fast, current Groq model (llama-3.3-70b-versatile is deprecated)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


embedder = load_embedder()

# ---------------------------------------------------------------------------
# API key setup
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
with st.sidebar:
    st.header("⚙️ Setup")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com/keys")
    client = None
    if api_key:
        client = Groq(api_key=api_key)
        st.success("API key configured")
    else:
        st.warning("Enter a Groq API key to enable the assistant.")

    st.divider()
    st.header("📥 Add knowledge")

    uploaded_files = st.file_uploader(
        "Upload security advisories / reports (PDF)", type=["pdf"], accept_multiple_files=True
    )

    st.markdown("**Or fetch a live CVE from NVD**")
    cve_id = st.text_input("CVE ID (e.g. CVE-2024-3400)")
    fetch_cve = st.button("Fetch CVE")

    st.divider()
    if st.button("🗑️ Clear knowledge base"):
        st.session_state.pop("index", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("sources", None)
        st.success("Cleared.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        st.session_state.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized vectors
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


def severity_badge(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "⚪ Unknown"
    if score >= 9.0:
        return "🔴 Critical"
    if score >= 7.0:
        return "🟠 High"
    if score >= 4.0:
        return "🟡 Medium"
    return "🟢 Low"


def retrieve(query: str, k=TOP_K):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    qvec = embed_query(query)
    scores, idxs = st.session_state.index.search(qvec, min(k, st.session_state.index.ntotal))
    results = []
    for i in idxs[0]:
        if i == -1:
            continue
        results.append((st.session_state.chunks[i], st.session_state.sources[i]))
    return results


def answer_question(client: Groq, query: str, context_pairs):
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src in context_pairs)
    prompt = f"""You are a cyber security analyst assistant. Answer the question ONLY using the
context below. If the context doesn't contain the answer, say so clearly rather than guessing.
Cite the source label for each claim you make.

Context:
{context}

Question: {query}

Answer:"""
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Ingest uploads
# ---------------------------------------------------------------------------
if uploaded_files:
    for f in uploaded_files:
        already_added = f.name in st.session_state.sources
        if not already_added:
            with st.spinner(f"Processing {f.name}..."):
                text = extract_pdf_text(f.read())
                n = add_to_index(text, f.name)
            st.sidebar.success(f"Added {n} chunks from {f.name}")

if fetch_cve:
    if not cve_id.strip():
        st.sidebar.error("Enter a CVE ID first.")
    else:
        with st.spinner(f"Fetching {cve_id} from NVD..."):
            try:
                result = fetch_cve_from_nvd(cve_id)
            except Exception as e:
                result = None
                st.sidebar.error(f"NVD lookup failed: {e}")
        if result:
            text, cvss = result
            n = add_to_index(text, cve_id.upper())
            st.sidebar.success(f"Added {cve_id.upper()} — Severity: {severity_badge(cvss)} ({cvss})")
        elif result is None and cve_id.strip():
            st.sidebar.warning("No data found for that CVE ID.")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.subheader("💬 Ask a question")

if st.session_state.chunks:
    st.info(f"Knowledge base: {len(st.session_state.chunks)} chunks from {len(set(st.session_state.sources))} source(s).")
else:
    st.info("Knowledge base is empty — upload a PDF or fetch a CVE from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("e.g. Is CVE-2024-3400 being actively exploited, and what's the mitigation?")

if query:
    if not client:
        st.error("Please enter a Groq API key in the sidebar first.")
    elif not st.session_state.chunks:
        st.error("Add at least one document or CVE before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                results = retrieve(query)
                answer = answer_question(client, query, results)
                st.markdown(answer)
                with st.expander("📎 Sources used"):
                    for chunk, src in results:
                        st.markdown(f"**{src}**")
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})
