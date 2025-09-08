# robot_teacher_final.py
import streamlit as st
import fitz                                 # PyMuPDF (fast PDF reading)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from typing import List

# ----------------- Configuration -----------------
MODEL_PATH = r"models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"  # <<-- set your model path
N_CTX = 4096
TOP_K = 5                    # number of chunks to retrieve for context
SIMILARITY_THRESHOLD = 0.18  # min cosine similarity to accept context (tuneable)
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
    # Use inner product with normalized vectors for cosine similarity via FAISS
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    return index


def generate_teacher_answer(context: str, question: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generates a teacher-style answer with:
      - Short friendly answer
      - Clear explanation
      - ONE relatable real-world example (everyday life)
      - One-line summary
      - Closing prompt
    Uses streaming for faster UI updates.
    """
    system_msg = (
        "You are a warm, patient classroom teacher. Speak in a natural, friendly way, "
        "like you’re explaining to a student sitting in front of you. "
        "Avoid stiff academic tone. Use short, clear sentences. "
        "Always use everyday language. "
        "Give ONE real-world example that feels practical and familiar (cooking, sports, shopping, phone apps, daily life). "
        "Do not add multiple examples, just one strong example. "
        "Make the explanation flow like spoken teaching."
    )

    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer with these labeled sections:\n"
        "1) Short Answer (2 simple sentences max)\n"
        "2) Explanation (friendly step-by-step, like talking to a student)\n"
        "3) Real-world Example (ONE clear, everyday situation)\n"
        "4) One-line Summary\n"
        "5) Ask the student: 'Do you want me to explain it in a simpler way or give another example?'\n\n"
        "Keep it warm, friendly, and conversational."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Streaming for faster, natural response
    full_text = ""
    stream = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True)

    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                full_text += delta
                # Stream to UI progressively
                st.write(delta, unsafe_allow_html=True)

    return full_text


# -------------------------------------------------

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📚 Robot Teacher — Focused Example Mode")

st.markdown(
    """
    **Notes:** The system will provide a short answer, a clear explanation, **one real-world example** for the main idea,
    and a one-line summary. The model is restricted to the uploaded document (it will say if it can't answer from the PDF).
    """
)

uploaded_pdf = st.file_uploader("Upload PDF (lecture notes / textbook)", type="pdf")

if uploaded_pdf is not None:
    st.success("PDF uploaded — extracting and preparing...")

    # 1) Extract text (cached)
    raw_text = extract_text_from_pdf_bytes(uploaded_pdf.read())
    raw_text = " ".join(raw_text.split())  # small cleanup

    # 2) Chunk text
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    st.write(f"Document split into {len(chunks)} chunks.")

    # 3) Embed chunks (cached)
    with st.spinner("Embedding chunks (one-time, cached)..."):
        chunk_embeddings_np = embed_texts(chunks, batch_size=EMBED_BATCH_SIZE)
        # normalize for inner product cosine
        faiss.normalize_L2(chunk_embeddings_np)

    # 4) Build FAISS index
    index = build_faiss_index(chunk_embeddings_np)

    # Sidebar: quick topic selection (first sentence of chunk sample)
    st.sidebar.header("Quick Topics (sample)")
    for i, c in enumerate(chunks[:min(40, len(chunks))]):
        st.sidebar.write(f"{i}: {c.split('.')[:1][0][:80]}")

    # 5) Ask a question
    question = st.text_input("Ask a question based on the uploaded PDF:")

    if st.button("Submit Question") and question.strip():
        with st.spinner("Searching for relevant context..."):
            # 5a) Embed question
            q_emb = embed_model.encode(question).astype("float32")
            faiss.normalize_L2(q_emb.reshape(1, -1))

            # 5b) FAISS top-k search (inner product)
            D, I = index.search(q_emb.reshape(1, -1), TOP_K)
            top_indices = I[0].tolist()

            # 5c) compute cosine similarities precisely for thresholding
            sims = cosine_similarity(q_emb.reshape(1, -1), chunk_embeddings_np)[0]
            best_sim = float(np.max(sims))
            best_idx = int(np.argmax(sims))

            if best_sim < SIMILARITY_THRESHOLD:
                st.warning("I couldn't find a clear answer in the uploaded document. Try rephrasing the question or upload a more relevant document.")
            else:
                # Collect top chunks ordered by similarity
                sorted_by_sim = sorted(top_indices, key=lambda i: sims[i], reverse=True)
                top_chunks = [chunks[i] for i in sorted_by_sim]
                context = "\n\n".join(top_chunks)

                # Generate humanized answer with ONE example for main topic
                with st.spinner("Generating teacher-style answer..."):
                    answer_text = generate_teacher_answer(context, question, max_tokens=600, temperature=0.7)

                st.subheader("Answer (teacher-style)")
                st.write(answer_text)

                # Quick follow-up UI
                st.write("---")
                st.write("If you want a different explanation or simpler example, type a follow-up below.")
                follow = st.text_input("Follow-up (optional):")
                if st.button("Submit Follow-up") and follow.strip():
                    with st.spinner("Generating follow-up answer..."):
                        follow_ans = generate_teacher_answer(context, follow, max_tokens=400, temperature=0.6)
                    st.write(follow_ans)

st.markdown("---")
st.caption("Tuning tips: increase TOP_K for broader context; raise SIMILARITY_THRESHOLD to reduce hallucination; lower temperature for deterministic answers.")
