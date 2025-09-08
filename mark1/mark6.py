import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import PyPDF2
import re

# Load LLM (Mistral 7B - quantized)
llm = Llama(model_path="models/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf", n_ctx=4096)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Session state variables
if 'teaching' not in st.session_state:
    st.session_state.teaching = False
if 'current_chunk_index' not in st.session_state:
    st.session_state.current_chunk_index = 0
if 'interrupted' not in st.session_state:
    st.session_state.interrupted = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# Function to read PDF and return text
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size):
    words = re.findall(r'\w+', text)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Extract headings/topics using regex
def extract_topics(text):
    headings = re.findall(r'(?:\n|^)([A-Z][A-Z\s\d\.:]{5,})\n', text)
    return list(set([h.strip() for h in headings]))

# Generate embedding
def embed_chunk(chunk):
    return embed_model.encode(chunk)

# Retrieve top context chunks based on question embedding
def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return "\n\n".join([chunks[i] for i in top_indices])

# Generate answer from LLM
def generate_answer(context, question):
    messages = [{"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}]
    output = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7)
    return output["choices"][0]["message"]["content"]

# Streamlit UI
st.title("🤖 Robot Teacher")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    pdf_text = read_pdf(uploaded_pdf)
    topics = extract_topics(pdf_text)
    token_count = len(pdf_text.split())
    chunk_size = max(100, min(300, token_count // 10))
    st.write(f"PDF loaded! Auto CHUNK_SIZE = {chunk_size} words per chunk.")

    st.session_state.chunks = chunk_text(pdf_text, chunk_size)
    chunk_embeddings = [embed_chunk(chunk) for chunk in st.session_state.chunks]

    selected_topic = st.selectbox("Select a topic to start teaching:", topics)

    if st.button("Start Teaching"):
        st.session_state.teaching = True
        st.session_state.current_chunk_index = 0

    # Teaching Mode
    if st.session_state.teaching and not st.session_state.interrupted:
        idx = st.session_state.current_chunk_index
        if idx < len(st.session_state.chunks):
            st.write(f"### 📘 Teaching: {selected_topic}")
            st.write(st.session_state.chunks[idx])
            if st.button("Next >>"):
                st.session_state.current_chunk_index += 1
        else:
            st.write("✅ Finished teaching all chunks.")

    # Raise Hand to Ask Doubt
    if st.button("✋ Raise Hand"):
        st.session_state.interrupted = True
        st.session_state.paused_chunk = st.session_state.current_chunk_index

    if st.session_state.interrupted:
        doubt = st.text_input("Ask your doubt:")
        if doubt:
            question_embedding = embed_model.encode(doubt)
            context = retrieve_context(question_embedding, chunk_embeddings, st.session_state.chunks)
            answer = generate_answer(context, doubt)
            st.write("🤖 Answer:", answer)

            clarity = st.radio("Was your doubt clarified?", ["Yes", "No"])
            if clarity == "Yes":
                st.session_state.interrupted = False
                st.session_state.current_chunk_index = max(0, st.session_state.paused_chunk - 1)
            else:
                shorter_answer = generate_answer(context, doubt + " (explain briefly in a different way)")
                st.write("🔁 Clarified Again:", shorter_answer)

    if st.button("🔁 Can you repeat once again?"):
        idx = st.session_state.current_chunk_index
        if idx > 0:
            st.write("Sure, here it is again:")
            st.write(st.session_state.chunks[idx - 1])
