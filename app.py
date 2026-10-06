import os
import uuid
import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

APP_NAME = "Intelligent Personalised Assistant"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "personalised_assistant"
UPLOAD_FOLDER = "./uploaded_files"

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
load_dotenv()

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def initialize_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    return client, collection

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def split_text(text, chunk_size=700, overlap=100):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap

    return chunks

def add_document_to_chromadb(text, filename, model, collection):
    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [
        {
            "filename": filename,
            "chunk": index + 1
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)

def search_knowledge(question, model, collection, number_of_results=5):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas

def generate_answer(question, documents):
    if not documents:
        return "I could not find relevant information in your uploaded materials."

    parts = [
        "Based on your uploaded study material:",
        ""
    ]

    for document in documents[:3]:
        parts.append("• " + document)

    return "\n\n".join(parts)

try:
    with st.spinner("Loading the Intelligent Personalised Assistant..."):
        embedding_model = load_embedding_model()
        chroma_client, collection = initialize_chromadb()
except Exception as error:
    st.error("The application could not start.")
    st.code(str(error))
    st.stop()

st.title("🤖 Intelligent Personalised Assistant")
st.write(
    "An AI-powered assistant that learns from your personalised "
    "study materials and provides relevant information."
)

st.divider()

st.sidebar.title("⚙️ Assistant Settings")

st.sidebar.subheader("👤 Personal Information")

user_name = st.sidebar.text_input(
    "Your Name",
    value="Student"
)

course = st.sidebar.text_input(
    "Course",
    value="Data Science"
)

year = st.sidebar.selectbox(
    "Year",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)

st.sidebar.divider()
st.sidebar.subheader("📚 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF study materials",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("📥 Add Materials"):
        total_chunks = 0
        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            status.write(f"Processing: {uploaded_file.name}")

            try:
                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    uploaded_file.name
                )

                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                text = extract_text_from_pdf(file_path)

                if not text.strip():
                    st.warning(
                        f"No readable text found in {uploaded_file.name}"
                    )
                    continue

                chunks = add_document_to_chromadb(
                    text,
                    uploaded_file.name,
                    embedding_model,
                    collection
                )

                total_chunks += chunks
                progress.progress((index + 1) / len(uploaded_files))

                st.success(
                    f"✅ {uploaded_file.name} added successfully."
                )

            except Exception as error:
                st.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )

        status.empty()
        st.info(
            f"📚 Total knowledge chunks added: {total_chunks}"
        )

st.sidebar.divider()

try:
    total_documents = collection.count()
    st.sidebar.metric("Knowledge Chunks", total_documents)
except Exception:
    total_documents = 0

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        st.cache_resource.clear()
        st.success("Knowledge base cleared successfully.")
        st.rerun()
    except Exception as error:
        st.error(f"Could not clear database: {error}")

st.header(f"👋 Welcome, {user_name}!")

st.write(
    f"You are currently studying **{course}** "
    f"and you are in **{year}**."
)

st.header("💬 Ask Your Personalised Assistant")

question = st.text_area(
    "What would you like to know?",
    placeholder="Example: Explain Data Science in simple words.",
    height=100
)

if st.button("🤖 Ask Assistant", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Your knowledge base is empty.")
        st.info(
            "Please upload a PDF and click 'Add Materials' first."
        )

    else:
        with st.spinner("Thinking and searching your materials..."):
            try:
                documents, metadatas = search_knowledge(
                    question,
                    embedding_model,
                    collection,
                    number_of_results=5
                )
            except Exception as error:
                st.error(f"Search error: {error}")
                st.stop()

        st.subheader("🤖 Assistant Answer")
        st.write(generate_answer(question, documents))

        st.divider()
        st.subheader("📖 Sources")

        if metadatas:
            displayed_sources = set()

            for metadata in metadatas:
                if metadata:
                    filename = metadata.get("filename", "Unknown file")
                    chunk = metadata.get("chunk", "Unknown")
                    source = (filename, chunk)

                    if source not in displayed_sources:
                        st.write(
                            f"📄 **{filename}** — Chunk {chunk}"
                        )
                        displayed_sources.add(source)

st.divider()

st.header("✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Personalised Learning")
    st.write(
        "Uses your uploaded learning materials to provide "
        "relevant information."
    )

with col2:
    st.subheader("🧠 Intelligent Search")
    st.write(
        "Sentence Transformers create embeddings and ChromaDB "
        "finds relevant information."
    )

with col3:
    st.subheader("👤 Student Profile")
    st.write(
        "Stores basic profile information such as name, course "
        "and year during the current session."
    )

st.divider()

st.caption(
    "Intelligent Personalised Assistant | AI + NLP + Vector Database"
)
