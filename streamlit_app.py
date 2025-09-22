import streamlit as st
import pdfplumber
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
MODEL_NAME = "all-MiniLM-L6-v2"  # Auto-downloads from Hugging Face
CHUNK_SIZE = 500
OVERLAP = 100
EMB_DIM = 384

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()
index = faiss.IndexFlatIP(EMB_DIM)
docs_meta = []

# ---------------- HELPERS ----------------
def extract_and_chunk_pdf(file):
    """Extract text and split into chunks."""
    all_chunks = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk = text[start:end]

                if end < len(text):
                    extra_end = min(len(text), end + 50)
                    while end < extra_end and not text[end].isspace():
                        end += 1
                    chunk = text[start:end]

                all_chunks.append(chunk.strip())
                start = end - OVERLAP

            if len(all_chunks) > 2000:
                break

    return all_chunks


def _extract_field_from_text(question, text):
    """Regex extractor for fields like Name, Semester, CGPA, Email, Phone, GitHub."""
    q = (question or "").lower()
    t = (text or "").replace("\n", " ")

    if "name" in q:
        m = re.search(r"(?:name of student|name)\s*[:\-]?\s*([A-Z][a-zA-Z ,.'-]{2,})", t, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    if "semester" in q or "sem" in q:
        m = re.search(r"(?:semester|sem)\s*[:\-]?\s*([0-9]{1,2}|[ivxlcdm]+|[0-9]{1,2}(st|nd|rd|th)?)", t, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    if any(word in q for word in ["gpa", "cgpa", "cpi", "sgpa"]):
        m = re.search(r"(?:CGPA|GPA|CPI|SGPA)\s*[:=\-]?\s*([\d]{1,2}(?:\.\d{1,4})?)", t, re.IGNORECASE)
        if m:
            return m.group(1)

    if "email" in q:
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
        if m:
            return m.group(0)

    if "phone" in q or "contact" in q or "mobile" in q:
        m = re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\d\s-]{7,}\d)", t)
        if m:
            return m.group(0).strip()

    if "github" in q:
        m = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+", t, re.IGNORECASE)
        if m:
            return m.group(0)

    return None


def query(question, k=5):
    """Answer questions using regex + semantic search."""
    if not docs_meta:
        return []

    all_text = " ".join([d["text"] for d in docs_meta])
    direct = _extract_field_from_text(question, all_text)
    if direct:
        return [{"answer": direct, "context": "Extracted directly"}]

    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(docs_meta):
            d = docs_meta[idx]
            text_chunk = d["text"]
            answer = _extract_field_from_text(question, text_chunk)
            results.append({
                "score": float(score),
                "answer": answer if answer else text_chunk[:200],
                "context": text_chunk[:500]
            })
    return results


def summarize_doc(word_count=120):
    """Summarize the document and prepend CGPA if found."""
    combined = " ".join([d["text"] for d in docs_meta])
    words = combined.split()

    m = re.search(r"(?:CGPA|GPA|SGPA|CPI)\s*[:=\-]?\s*([\d]+(?:\.\d{1,4})?)", combined, re.IGNORECASE)
    cgpa_line = f"CGPA: {m.group(1)}\n" if m else ""

    summary_text = " ".join(words[:word_count]) + (" ..." if len(words) > word_count else "")
    return cgpa_line + summary_text


# ---------------- STREAMLIT UI ----------------
st.title("📄 Document Analyzer AI Agent")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    chunks = extract_and_chunk_pdf(uploaded_file)
    if chunks:
        embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        for i, chunk in enumerate(chunks):
            docs_meta.append({"chunk_id": i, "text": chunk})

        st.subheader("📌 Document Summary")
        st.write(summarize_doc())

        st.subheader("🔍 Ask Questions")
        user_q = st.text_input("Enter your question (e.g., 'What is the CGPA?', 'What is the email?')")
        if user_q:
            answers = query(user_q)
            if answers:
                st.write("**Answer:**", answers[0]["answer"])
                with st.expander("Show context"):
                    st.write(answers[0]["context"])
            else:
                st.warning("No answer found.")
