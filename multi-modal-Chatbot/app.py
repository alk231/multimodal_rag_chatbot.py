import streamlit as st
import os
from dotenv import load_dotenv
import base64

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity import ChatPerplexity

load_dotenv()

st.set_page_config(page_title="RAG Multimodal Chatbot", page_icon="🤖", layout="wide")

UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "faiss_index"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

# Session initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"sender": "bot", "text": "Hello! Upload files and ask questions."}]

if "vector_store" not in st.session_state:
    if os.path.exists(VECTOR_STORE_PATH):
        st.session_state.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        st.session_state.vector_store = None

if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None


# --------------------------
# Vectorstore functions
# --------------------------

def add_to_vectorstore(text, source):
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(documents, embedding_model)
    else:
        st.session_state.vector_store.add_documents(documents)

    st.session_state.vector_store.save_local(VECTOR_STORE_PATH)


def extract_text_from_pdf(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_txt(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    loader = TextLoader(path)
    text = "\n".join(p.page_content for p in loader.load())
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_docx(file):
    path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    loader = Docx2txtLoader(path)
    docs = loader.load()
    text = docs[0].page_content
    add_to_vectorstore(text, file.name)
    return text


def extract_text_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n".join(d.page_content for d in docs)
    add_to_vectorstore(text, url)
    return text


def get_youtube_text(url):
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    else:
        video_id = url.split("youtu.be/")[1].split("?")[0]

    try:
        snippets = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
        text = " ".join(s.text for s in snippets)
    except:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcripts.find_generated_transcript(["en", "hi"])
        parts = transcript.fetch()
        text = " ".join(p["text"] for p in parts)

    add_to_vectorstore(text, url)
    return text


# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("📤 Upload Knowledge Base")

file = st.sidebar.file_uploader("Upload File", type=["pdf", "txt", "docx"])
if file:
    with st.spinner("Processing..."):
        # Reset stored vectors
        st.session_state.vector_store = None
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)

        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            extract_text_from_pdf(file)
        elif ext == ".txt":
            extract_text_from_txt(file)
        elif ext == ".docx":
            extract_text_from_docx(file)

        st.sidebar.success("✅ File processed!")

image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image:
    path = os.path.join(UPLOAD_FOLDER, image.name)
    with open(path, "wb") as f:
        f.write(image.getbuffer())

    st.session_state.last_image_path = path
    st.sidebar.success("🖼️ Image ready!")

url = st.sidebar.text_input("Webpage URL")
if st.sidebar.button("Add URL") and url:
    with st.spinner("Fetching webpage..."):
        extract_text_from_url(url)
        st.sidebar.success("🌐 Webpage added!")

yt = st.sidebar.text_input("YouTube Link")
if st.sidebar.button("Fetch Transcript") and yt:
    with st.spinner("Fetching transcript..."):
        get_youtube_text(yt)
        st.sidebar.success("🎬 Transcript added!")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# --------------------------
# Chat Display
# --------------------------

st.write("## 🤖 Chat")

for msg in st.session_state.messages:
    align = "flex-end" if msg["sender"] == "user" else "flex-start"
    bg = "#007BFF" if msg["sender"] == "user" else "#FFFFFF"
    color = "white" if msg["sender"] == "user" else "black"

    st.markdown(f"""
        <div style="display:flex; justify-content:{align}; margin:6px 0;">
            <div style="background:{bg}; color:{color}; padding:12px 16px;
                        border-radius:16px; max-width:72%; font-size:16px;">
                {msg["text"]}
            </div>
        </div>
    """, unsafe_allow_html=True)


# --------------------------
# Chat Input + Logic
# --------------------------

prompt = st.chat_input("Ask a question...", key="chat_input_main")

if prompt:
    st.session_state.messages.append({"sender": "user", "text": prompt})

    # No document uploaded
    if st.session_state.vector_store is None:
        reply = "Please upload a PDF / TXT / DOCX file first."
        st.session_state.messages.append({"sender": "bot", "text": reply})
        st.rerun()

    # Detect image usage
    image_keywords = ["image", "photo", "picture", "see", "detect", "describe the image"]
    use_image = any(kw in prompt.lower() for kw in image_keywords)

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(prompt)
    context = "\n".join(d.page_content for d in docs)

    llm = ChatPerplexity()

    # Image mode
    if use_image and st.session_state.last_image_path:
        with open(st.session_state.last_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\n\nContext:\n" + context},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                ],
            }
        ]

        reply = llm.invoke(messages).content

    else:
        # Pure RAG
        template = PromptTemplate(
            template=(
                "Use ONLY the context to answer.\n"
                "If not found in context, reply 'I don't know'.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\nAnswer:"
            ),
            input_variables=["context", "question"]
        )

        chain = template | llm | StrOutputParser()
        reply = chain.invoke({"context": context, "question": prompt})

    st.session_state.messages.append({"sender": "bot", "text": reply})
    st.rerun()
