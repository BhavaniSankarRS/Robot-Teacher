# robot_teacher_final_clean.py
import streamlit as st
import fitz                                 # PyMuPDF (fast PDF reading)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from typing import List
import re
import time

# ----------------- Configuration -----------------
MODEL_PATH = r"models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"  # <<-- set your model path
N_CTX = 4096
TOP_K = 5
SIMILARITY_THRESHOLD = 0.18
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
EMBED_BATCH_SIZE = 8
# -------------------------------------------------

# Load local LLM model (llama.cpp)
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX)

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Fast PDF text extraction using PyMuPDF (fitz)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Return numpy array of embeddings (float32)."""
    embs = embed_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    arr = np.array(embs).astype("float32")
    return arr


def build_faiss_index(embeddings_np: np.ndarray) -> faiss.Index:
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    return index


def generate_teacher_answer(context: str, question: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generates teacher-style answer with human-teacher structure, shown sentence by sentence."""
    system_msg = (
        "You are a warm, patient classroom teacher. "
        "Explain naturally, like talking to students face-to-face. "
        "Use clear and short sentences. "
        "Structure your answer into sections: \n"
        "1) Introduction (what we’re learning) \n"
        "2) Core Concept (main definition/idea) \n"
        "3) Step-by-Step Explanation (easy steps) \n"
        "4) Real-world Example (ONE practical example) \n"
        "5) Quick Recap (1–2 lines)."
    )

    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer using exactly this structure:\n"
        "### Introduction\n...\n"
        "### Core Concept\n...\n"
        "### Step-by-Step Explanation\n...\n"
        "### Real-world Example\n...\n"
        "### Quick Recap\n...\n"
        "Keep it natural, warm, and conversational."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    full_text = ""
    stream = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True)

    placeholder = st.empty()

    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                full_text += delta

    # Sentence-by-sentence reveal
    lines = re.split(r'(?<=[.!?]) +', full_text.strip())
    displayed_text = ""
    for line in lines:
        displayed_text += line + " "
        placeholder.markdown(displayed_text)
        time.sleep(1.1)

    return full_text


# -------------------------------------------------
# ---------- Streamlit UI ----------
st.set_page_config(layout="centered", page_title="Robot Teacher", page_icon="🤖")
st.title("📚 Robot Teacher")
st.write("Upload your notes and ask me questions. I’ll explain step by step, just like a real teacher, with a simple real-world example.")

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf_bytes(uploaded_pdf.read())
        raw_text = " ".join(raw_text.split())

        chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunk_embeddings_np = embed_texts(chunks, batch_size=EMBED_BATCH_SIZE)
        faiss.normalize_L2(chunk_embeddings_np)
        index = build_faiss_index(chunk_embeddings_np)

    st.success("✅ PDF processed. Now ask your question:")

    question = st.text_input("Your Question:")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            q_emb = embed_model.encode(question).astype("float32")
            faiss.normalize_L2(q_emb.reshape(1, -1))

            D, I = index.search(q_emb.reshape(1, -1), TOP_K)
            sims = cosine_similarity(q_emb.reshape(1, -1), chunk_embeddings_np)[0]
            best_sim = float(np.max(sims))

            if best_sim < SIMILARITY_THRESHOLD:
                st.warning("I couldn’t find a good answer in this document. Try rephrasing your question.")
            else:
                top_chunks = [chunks[i] for i in I[0]]
                context = "\n\n".join(top_chunks)
                generate_teacher_answer(context, question, max_tokens=600, temperature=0.7)
