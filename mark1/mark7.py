import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import PyPDF2
import re
import time

# Load LLM
llm = Llama(model_path="models/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf", n_ctx=4096)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF reading and cleaning
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text):
    # Remove headers, page numbers, non-teaching elements
    lines = text.split('\n')
    lines = [line for line in lines if not re.match(r"^(\s*[A-Z\d\s]{2,}\s*)$", line)]
    clean = " ".join(lines)
    clean = re.sub(r"\s+", " ", clean)
    return clean

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Embed a chunk or question
def embed_text(text):
    return embed_model.encode(text)

def retrieve_context(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

def generate_answer(context, question):
    messages = [
        {"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}
    ]
    output = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7)
    return output["choices"][0]["message"]["content"]

def explain_line(line):
    prompt = [
        {"role": "user", "content": f"Explain in a student-friendly way:\n\n'{line}'"}
    ]
    output = llm.create_chat_completion(messages=prompt, max_tokens=200, temperature=0.7)
    return output["choices"][0]["message"]["content"]

# Streamlit UI
st.title("📚 Robot Teacher")

if "teaching_index" not in st.session_state:
    st.session_state.teaching_index = 0
if "sentences" not in st.session_state:
    st.session_state.sentences = []
if "auto_teach" not in st.session_state:
    st.session_state.auto_teach = False
if "paused_for_doubt" not in st.session_state:
    st.session_state.paused_for_doubt = False

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    raw_text = read_pdf(uploaded_pdf)
    cleaned_text = clean_text(raw_text)
    all_sentences = split_into_sentences(cleaned_text)

    # Display topics (first sentence of each ~10 sentences)
    topic_options = [all_sentences[i] for i in range(0, len(all_sentences), 10)]
    selected_topic = st.selectbox("Select a topic to learn:", topic_options)

    if selected_topic:
        # Find where the topic starts
        start_index = all_sentences.index(selected_topic)
        st.session_state.sentences = all_sentences[start_index:]
        st.session_state.teaching_index = 0

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Teaching"):
            st.session_state.auto_teach = True
    with col2:
        if st.button("✋ Raise Hand"):
            st.session_state.paused_for_doubt = True
            st.session_state.auto_teach = False

    if st.session_state.paused_for_doubt:
        st.info("🛑 Teaching paused. Ask your doubt below:")
        user_doubt = st.text_input("Ask your doubt:")
        if user_doubt:
            doubt_ans = generate_answer("\n".join(st.session_state.sentences), user_doubt)
            st.write("🤖 Answer:")
            st.write(doubt_ans)
            satisfied = st.radio("Was your doubt clarified?", ["Yes", "No"])
            if satisfied == "Yes":
                st.session_state.paused_for_doubt = False
                st.session_state.auto_teach = True
            else:
                retry_explanation = explain_line(user_doubt)
                st.write("🔁 Clarification:")
                st.write(retry_explanation)

    if st.button("🔁 Can you repeat once again?"):
        st.session_state.teaching_index = max(0, st.session_state.teaching_index - 3)

    if st.session_state.auto_teach and st.session_state.teaching_index < len(st.session_state.sentences):
        sentence = st.session_state.sentences[st.session_state.teaching_index]
        explanation = explain_line(sentence)
        st.markdown(f"### 📘 Teaching: {sentence}")
        st.write(explanation)
        st.session_state.teaching_index += 1
        time.sleep(3)  # delay for auto
    elif st.session_state.teaching_index >= len(st.session_state.sentences):
        st.success("🎓 Topic teaching complete!")
