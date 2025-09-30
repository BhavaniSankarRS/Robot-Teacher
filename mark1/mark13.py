# robot_teacher_final_clean.py
import json
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


def update_preferred_topics(current_topic, understanding_score):
    """Simple function to track which topics the class prefers"""
    if understanding_score >= 4 and current_topic not in st.session_state.class_profile['preferred_topics']:
        # Only keep the 5 most recent preferred topics
        st.session_state.class_profile['preferred_topics'] = st.session_state.class_profile['preferred_topics'][-4:] + [current_topic]

def generate_lesson_plan(new_topic: str, class_profile: dict) -> dict:
    """
    Acts as the curriculum planner. Analyzes the new topic and CLASS HISTORY
    to create a tailored lesson plan for the overall class level.
    Returns a JSON object with the plan.
    """
    # Build the context string for the prompt based on CLASS profile
    context_str = f"This lesson is for a whole class. "
    
    # Add context about recent topics
    if class_profile.get('recent_topics'):
        context_str += f"The class has recently been learning about {', '.join(class_profile['recent_topics'][-2:])}. "
    
    # Add context about class performance
    context_str += f"The class's average understanding score is {class_profile.get('avg_understanding_score', 4.0)}/5.0 "
    context_str += f"and they find topics with a difficulty rating of {class_profile.get('avg_difficulty_score', 3.0)}/5.0 to be most appropriate. "
    
    # Add context about preferred topics if they exist
    if class_profile.get('preferred_topics'):
        context_str += f"They generally engage well with topics related to: {', '.join(class_profile['preferred_topics'])}. "

    planner_prompt = f"""
# ROLE:
You are an expert educational curriculum designer. Your job is to create a detailed lesson plan for an entire CLASS.

# CLASS PROFILE:
{context_str}

# TASK:
Design a lesson plan for a new topic: "{new_topic}". The plan should be tailored to the overall level and preferences of the class.

Provide a JSON object with the following keys:
- "learning_objective": "A single sentence describing the main goal for this class for this lesson.",
- "central_analogy": "A creative, simple analogy to explain the core idea of {new_topic}. Choose an analogy that matches the class's preferred topics if relevant.",
- "key_terms": ["list", "of", "5", "or", "6", "most", "important", "terms", "prioritize", "fundamental", "concepts"],
- "common_pitfall": "Describe one common way classes at this level misunderstand this topic, and how to preemptively address it.",
- "hook": "An engaging question or statement to begin the lesson that will appeal to the whole class.",
- "pace": "Should be 'slow', 'moderate', or 'fast' based on the class's understanding score."

IMPORTANT: Return ONLY the raw JSON. Do not include any other text before or after.
"""
    # Use your LLM to get the plan. We use a low temperature for structured output.
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": planner_prompt}],
        max_tokens=1024,
        temperature=0.1,
        stop=None
    )
    plan_json = response['choices'][0]['message']['content'].strip()
    
    # Clean the response to extract just the JSON
    try:
        json_match = re.search(r'\{.*\}', plan_json, re.DOTALL)
        if json_match:
            plan_dict = json.loads(json_match.group())
        else:
            plan_dict = json.loads(plan_json)
        return plan_dict
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse lesson plan JSON from LLM. Raw output: {plan_json}")
        # Return a default plan if parsing fails
        return {
            "learning_objective": f"Understand the topic of {new_topic}.",
            "central_analogy": f"Think of {new_topic} like a familiar everyday system.",
            "key_terms": ["concept", "term", "example"],
            "common_pitfall": "A common mistake is to confuse related concepts.",
            "hook": f"Have you ever wondered about {new_topic}?",
            "pace": "moderate"
        }

def generate_adaptive_teacher_answer(context: str, question: str, lesson_plan: dict, max_tokens: int = 700) -> str:
    """
    Generates a teacher-style answer using the personalized lesson plan.
    """
    system_msg = "You are a warm, patient, and adaptive classroom teacher. Explain naturally, like talking to a student one-on-one."

    # Format the lesson plan into the user message
    plan_instructions = f"""
# YOUR LESSON PLAN:
HOOK: {lesson_plan.get('hook', 'Start with an engaging question.')}
CENTRAL ANALOGY: {lesson_plan.get('central_analogy', 'Use a simple analogy.')}
KEY TERMS TO COVER: {', '.join(lesson_plan.get('key_terms', []))}
COMMON PITFALL TO AVOID: {lesson_plan.get('common_pitfall', 'Address a common misunderstanding.')}
LEARNING GOAL: {lesson_plan.get('learning_objective', 'Help the student understand.')}

# INSTRUCTIONS:
- Use the hook to start your answer.
- Weave the central analogy throughout your explanation.
- Naturally define the key terms when they come up.
- Proactively address the common pitfall.
- Keep your sentences clear and short.
- Structure your answer logically but conversationally.
"""

    user_msg = f"""
{plan_instructions}

# CONTEXT FROM THE TEXTBOOK:
{context}

# STUDENT'S QUESTION:
{question}

Now, provide your answer. Speak directly to the student.
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    full_text = ""
    stream = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.7, stream=True)

    placeholder = st.empty()
    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                full_text += delta
                # Simple streaming output
                placeholder.markdown(full_text + "▌")
    placeholder.markdown(full_text) # Final render
    return full_text


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

# ---------- Class Profile & Adaptive Logic ----------

# Initialize session state for CLASS profile (overall class performance)
if 'class_profile' not in st.session_state:
    st.session_state.class_profile = {
        'recent_topics': [],       # List of recently taught topics to the class
        'avg_difficulty_score': 3.0, # Average difficulty (1-5 scale, 5 is hard)
        'avg_understanding_score': 4.0, # Average understanding (1-5 scale, 5 is excellent)
        'preferred_topics': []     # Topics the class generally engages well with
    }

# Helper function to extract the main topic from a question
def extract_topic_from_question(question: str) -> str:
    """A simple function to guess the topic from the question."""
    # Remove question words
    q_words = ['what', 'who', 'how', 'explain', 'define', 'tell me about', 'is', 'are', 'do', 'does']
    topic = question.lower()
    for word in q_words:
        topic = topic.replace(word, '')
    topic = topic.replace('?', '').strip()
    # Return the first few words as the topic
    return ' '.join(topic.split()[:3]).title()

#-----------------------------------
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

    # Add sliders for class feedback (instead of individual difficulty)
    col1, col2 = st.columns(2)
    with col1:
        understanding_score = st.slider(
            "Class Understanding (1-5)",
            min_value=1, max_value=5, value=4,
            help="How well does the class generally understand this material?"
        )
    with col2:
        difficulty_score = st.slider(
            "Topic Difficulty (1-5)", 
            min_value=1, max_value=5, value=3,
            help="How difficult is this topic for the class?"
        )

    if st.button("Get Answer") and question.strip():
        with st.spinner("🧠 Planning class lesson..."):
            # Step 1: Extract topic and update CLASS profile
            current_topic = extract_topic_from_question(question)
            
            # Update class profile with this interaction
            if current_topic and (not st.session_state.class_profile['recent_topics'] or st.session_state.class_profile['recent_topics'][-1] != current_topic):
                st.session_state.class_profile['recent_topics'].append(current_topic)
                
            # Update running averages (simple implementation)
            st.session_state.class_profile['avg_understanding_score'] = (st.session_state.class_profile['avg_understanding_score'] + understanding_score) / 2
            st.session_state.class_profile['avg_difficulty_score'] = (st.session_state.class_profile['avg_difficulty_score'] + difficulty_score) / 2

            # Add this line after updating the averages
            update_preferred_topics(current_topic, understanding_score)

            # Step 2: Retrieve relevant context from PDF
            q_emb = embed_model.encode(question).astype("float32")
            faiss.normalize_L2(q_emb.reshape(1, -1))

            D, I = index.search(q_emb.reshape(1, -1), TOP_K)
            sims = cosine_similarity(q_emb.reshape(1, -1), chunk_embeddings_np)[0]
            best_sim = float(np.max(sims))

            if best_sim < SIMILARITY_THRESHOLD:
                st.warning("I couldn't find a good answer in this document. Try rephrasing your question.")
            else:
                top_chunks = [chunks[i] for i in I[0]]
                context = "\n\n".join(top_chunks)

        with st.spinner("👩‍🏫 Tailoring explanation for class level..."):
            # Step 3: Generate a class-tailored lesson plan
            lesson_plan = generate_lesson_plan(current_topic, st.session_state.class_profile)
            
            # Optional: Display the plan in an expander for transparency
            with st.expander("🧩 See My Teaching Plan for This Class"):
                st.json(lesson_plan)

            # Step 4: Generate and display the adaptive answer
            generate_adaptive_teacher_answer(context, question, lesson_plan, max_tokens=700)

    # Display class profile for transparency
    with st.expander("📊 Current Class Profile"):
        st.write("This represents the overall performance and preferences of your class:")
        st.write(st.session_state.class_profile)