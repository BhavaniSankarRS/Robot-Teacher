import streamlit as st
import fitz                                 # PyMuPDF (fast PDF reading)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# ----------------- Configuration -----------------
MODEL_PATH = r"models\mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"  # adjust path
N_CTX = 4096
TOP_K = 5               # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.18  # min cosine similarity to accept context (tuneable)
# -------------------------------------------------

# Load local LLM model
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX)

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes):
    """Fast PDF text extraction using PyMuPDF (fitz)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

@st.cache_data(show_spinner=False)
def embed_texts(texts, batch_size=8):
    return embed_model.encode(texts, batch_size=batch_size, show_progress_bar=False)

def build_faiss_index(embeddings_np):
    dim = embeddings_np.shape[1]
    # Use inner product (dot) with normalized vectors for cosine similarity via FAISS
    index = faiss.IndexFlatIP(dim)
    # normalize
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    return index

def generate_teacher_answer_with_examples(context, question, max_tokens=512, temperature=0.7):
    """
    Use create_chat_completion to ask the model to generate:
    - short direct answer
    - step-by-step explanation
    - a real-world example
    - a short analogy
    - one-sentence summary
    - ask if user needs clarification
    """
    system_msg = (
        "You are a friendly, patient teacher. Answer only from the provided context. "
        "If the answer is not present in the context, say: 'I don't know based on the provided material.'\n\n"
        "When answering, always include the following sections (clearly labeled):\n"
        "1) Short Answer (1-2 sentences)\n"
        "2) Explanation (step-by-step, clear, student-friendly)\n"
        "3) Real-world Example (one concrete example showing how it is used)\n"
        "4) Analogy (a short metaphor/analogy to make it intuitive)\n"
        "5) One-line Summary\n"
        "6) 'Do you want a simpler explanation or another example?' (this should be the last line)\n\n"
        "Keep language simple, conversational, and avoid jargon where possible."
    )

    user_msg = (
        f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
        "Answer following the exact section order above. Keep each section concise."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    resp = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp["choices"][0]["message"]["content"]

# -------------------------------------------------

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📚 Robot Teacher — Improved (examples & humanized answers)")

uploaded_pdf = st.file_uploader("Upload PDF (lecture notes / textbook)", type="pdf")

if uploaded_pdf is not None:
    st.success("PDF uploaded — processing (fast extract + embeddings)...")

    # 1) Read text (cached)
    raw_text = extract_text_from_pdf_bytes(uploaded_pdf.read())

    # small cleanup (remove long whitespace)
    raw_text = " ".join(raw_text.split())

    # 2) Chunk text
    chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
    st.write(f"Document split into {len(chunks)} chunks.")

    # 3) Embed chunks (cached)
    with st.spinner("Embedding chunks (one-time)..."):
        chunk_embeddings = embed_texts(chunks, batch_size=8)
        chunk_embeddings_np = np.array(chunk_embeddings).astype("float32")
        # normalize for cosine via inner product
        faiss.normalize_L2(chunk_embeddings_np)

    # 4) Build FAISS index (fast)
    index = build_faiss_index(chunk_embeddings_np)

    # Optional: List of candidate topics (very simple heuristic)
    sample_headlines = []
    for i, c in enumerate(chunks[0: min(40, len(chunks))]):
        sample_headlines.append((i, c.split(".")[0][:80]))
    st.sidebar.header("Quick Topics (first sentences of chunks)")
    for idx, title in sample_headlines:
        st.sidebar.write(f"{idx}. {title}")

    # 5) Ask a question
    question = st.text_input("Ask a question based on the uploaded PDF:")

    if question:
        # 5a) Embed question and check similarity
        q_emb = embed_model.encode(question).astype("float32")
        faiss.normalize_L2(q_emb.reshape(1, -1))

        # FAISS top-k search (inner product because we normalized)
        D, I = index.search(q_emb.reshape(1, -1), TOP_K)
        top_indices = I[0].tolist()

        # compute cosine similarities precisely with sklearn for thresholding
        sims = cosine_similarity(q_emb.reshape(1, -1), chunk_embeddings_np)[0]
        best_sim = float(np.max(sims))
        best_idx = int(np.argmax(sims))

        if best_sim < SIMILARITY_THRESHOLD:
            st.warning("I couldn't find a clear answer in the uploaded document. Try rephrasing or upload a more relevant PDF.")
        else:
            # build context from the top_indices but order them by similarity descending
            sorted_by_sim = sorted(top_indices, key=lambda i: sims[i], reverse=True)
            top_chunks = [chunks[i] for i in sorted_by_sim]
            context = "\n\n".join(top_chunks)

            with st.spinner("Generating a humanized answer with examples..."):
                answer_text = generate_teacher_answer_with_examples(context, question, max_tokens=600, temperature=0.7)

            # present nicely — split by sections if LLM followed the format
            st.subheader("Answer (teacher-style)")
            st.write(answer_text)

            # small UI: allow immediate follow-up
            st.write("---")
            if st.button("👍 That helped — next"):
                st.success("Great — ask another question or select a topic.")
            if st.button("❓ Ask follow-up / simpler example"):
                follow = st.text_input("Type follow-up question (or type: 'simpler') :")
                if follow:
                    with st.spinner("Answering follow-up..."):
                        follow_ans = generate_teacher_answer_with_examples(context, follow, max_tokens=400, temperature=0.7)
                    st.write(follow_ans)

st.markdown("---")
st.caption("Notes: tune SIMILARITY_THRESHOLD and TOP_K for your documents. To improve 'examples' quality, increase context (more top_k) or provide narrower prompts.")
