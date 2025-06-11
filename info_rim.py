import os
import json
import tempfile
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------- CONFIGURATION ---------
st.set_page_config(page_title="RAG Assistant", layout="centered", page_icon="images/logo_ministere.jpg")
os.makedirs("conversations", exist_ok=True)

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. 
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.
Always respond in the same language as the question.
Context will be passed as "Context:"
User question will be passed as "Question:"
"""

def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

def safe_file_name(name: str) -> str:
    # Garder uniquement alphanum, _ et -
    return "".join(c for c in name if c.isalnum() or c in ("_", "-")).rstrip()

def save_conversation(conv_history, name):
    if conv_history:
        safe_name = safe_file_name(name)
        file_path = os.path.join("conversations", f"{safe_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv_history, f, indent=2, ensure_ascii=False)
        return safe_name
    return None

def load_saved_conversations():
    return sorted([f[:-5] for f in os.listdir("conversations") if f.endswith(".json")])

def load_conversation(name):
    path = os.path.join("conversations", f"{name}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def delete_conversation(name):
    path = os.path.join("conversations", f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    embeddings = embed_texts(documents)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

def load_and_process_all_pdfs(dataset_dir: str = "datasets", force: bool = False):
    if os.path.exists("./demo-rag-chroma") and not force:
        return
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(dataset_dir, file_name)
            with open(file_path, "rb") as f:
                temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
                temp_file.write(f.read())
                temp_file.close()
                loader = PyMuPDFLoader(temp_file.name)
                docs = loader.load()
                os.unlink(temp_file.name)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400, chunk_overlap=100,
                    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
                )
                all_splits = text_splitter.split_documents(docs)
                clean_name = file_name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
                add_to_vector_collection(all_splits, clean_name)

def query_collection(prompt: str, n_results: int = 10):
    query_embedding = embed_texts([prompt])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results

def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = encoder_model.predict([(prompt, doc) for doc in documents])
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    relevant_text = "\n\n".join([documents[i] for i in sorted_indices])
    return relevant_text, sorted_indices

def call_llm(context: str, prompt: str):
    full_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
    ]
    response = ollama.chat(model="llama3.2:3b", stream=True, messages=full_prompt)
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]

# --------- INIT ---------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
collection = chroma_client.get_or_create_collection(name="rag_app", metadata={"hnsw:space": "cosine"})

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "conversation_name" not in st.session_state:
    st.session_state.conversation_name = None
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "open_menu" not in st.session_state:
    st.session_state.open_menu = {}

# --------- SIDEBAR ---------
st.sidebar.title("📂 Documents & Conversations")

force_reload = st.sidebar.checkbox("🔁 Forcer rechargement PDF")
with st.spinner("📂 Indexation des documents..."):
    load_and_process_all_pdfs(force=force_reload)
st.sidebar.success("✅ PDF chargés.")

if st.sidebar.button("🆕 Nouvelle conversation"):
    # Sauvegarde conversation actuelle si existante
    if st.session_state.conversation_history and st.session_state.conversation_name:
        save_conversation(st.session_state.conversation_history, st.session_state.conversation_name)
    st.session_state.conversation_history = []
    st.session_state.conversation_name = None
    st.session_state.show_about = False
    st.rerun()

if st.sidebar.button("ℹ️ À propos de nous"):
    st.session_state.show_about = True
    st.rerun()

st.sidebar.markdown("### 💬 Conversations enregistrées")

for conv in load_saved_conversations():
    is_open = st.session_state.open_menu.get(conv, False)
    with st.sidebar.container():
        cols = st.columns([5, 1])
        if cols[0].button(f"📂 {conv}", key=f"load_{conv}"):
            st.session_state.conversation_history = load_conversation(conv)
            st.session_state.conversation_name = conv
            st.session_state.show_about = False
            st.sidebar.info(f"✅ Conversation `{conv}` chargée")
            st.rerun()
        if cols[1].button("⋮", key=f"menu_{conv}"):
            st.session_state.open_menu[conv] = not is_open
            st.rerun()
        if is_open:
            new_name = st.text_input(f"Renommer '{conv}'", value=conv, key=f"rename_input_{conv}")
            rename_col, delete_col = st.columns([1, 1])
            if rename_col.button("✏️ Renommer", key=f"rename_btn_{conv}"):
                old_path = os.path.join("conversations", f"{conv}.json")
                new_safe_name = safe_file_name(new_name)
                new_path = os.path.join("conversations", f"{new_safe_name}.json")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    st.session_state.open_menu.pop(conv, None)
                    st.rerun()
            if delete_col.button("🗑️ Supprimer", key=f"delete_btn_{conv}"):
                delete_conversation(conv)
                st.session_state.open_menu.pop(conv, None)
                st.rerun()

# --------- PAGE PRINCIPALE ---------
st.title("📚 Assistant Journal Officiel")

if st.session_state.show_about:
    st.subheader("👨‍💻 Équipe de développement")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/23216.png", caption="Aliou Diallo", use_container_width=True)
        st.markdown("**Statisticien-Économiste / Data Scientist**")
    with col2:
        st.image("images/23215.jpeg", caption="Mouhamed M. A. YAHYA", use_container_width=True)
        st.markdown("**Statisticien-Économiste / Data Scientist / Chef de projet**")
    with col3:
        st.image("images/23093.webp", caption="Mariem CHEIKH SIDIYA", use_container_width=True)
        st.markdown("**Data Scientist**")
    st.markdown("---")
    if st.button("⬅️ Retour à l'application"):
        st.session_state.show_about = False
        st.rerun()
    st.stop()

st.write("Posez une ou plusieurs questions. Les réponses sont tirées des documents PDF extrait du site https://www.msgg.gov.mr/ar/journal-officiel.")

# Affiche la conversation existante
for turn in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

# Nouvelle question
if prompt := st.chat_input("Posez votre question ici..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Si pas encore de nom de conversation, on en crée un à partir de la question (ex: 1er mots)
    if not st.session_state.conversation_name:
        # Extraire 4 premiers mots (ou moins) de la question, sans caractères spéciaux
        words = prompt.split()
        candidate_name = " ".join(words[:4])
        candidate_name = safe_file_name(candidate_name) or "conversation"
        st.session_state.conversation_name = candidate_name

    results = query_collection(prompt)
    context_docs = results.get("documents", [[]])[0]

    if not context_docs:
        with st.chat_message("assistant"):
            st.error("Aucun contexte trouvé.")
    else:
        relevant_text, _ = re_rank_cross_encoders(prompt, context_docs)
        full_answer = ""
        with st.chat_message("assistant"):
            response_area = st.empty()
            for chunk in call_llm(relevant_text, prompt):
                full_answer += chunk
                response_area.markdown(full_answer)

        # Sauvegarder dans l'historique
        st.session_state.conversation_history.append({
            "question": prompt,
            "answer": full_answer
        })

        # Sauvegarder la conversation sur disque avec le nom généré
        save_conversation(st.session_state.conversation_history, st.session_state.conversation_name)
