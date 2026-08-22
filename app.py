import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found inside .env")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY


DATA_FOLDER = 'docs'       
DB_FOLDER = 'chromadb'


EMBEDDING_MODEL = "gemini-embedding-2"
LLM_NAME = "gemini-3.5-flash-lite"  

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 2

os.makedirs(DATA_FOLDER, exist_ok=True)


embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

vectordb = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=embeddings,
)

existing_sources = set()
try:
    stored = vectordb.get(include=["metadatas"])
    if "metadatas" in stored:
        existing_sources = {meta["source"] for meta in stored["metadatas"] if "source" in meta}
except Exception:
    existing_sources = set()

new_docs = []
for file in os.listdir(DATA_FOLDER):
    path = os.path.join(DATA_FOLDER, file)

    # Handle .md and .txt
    if file.endswith(".md") or file.endswith(".txt"):
        if file not in existing_sources:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            new_docs.append(Document(
                page_content=text,
                metadata={
                    "source": file,
                    "category": "wellness",
                    "target_issue": "general",
                    "clinical_source": "student_resources",
                    "urgency_level": "low"
                }
            ))
            print("New text/markdown file added:", file)

    # Handle .pdf
    elif file.endswith(".pdf"):
        if file not in existing_sources:
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = file
                d.metadata["category"] = "wellness"
                d.metadata["target_issue"] = "general"
                d.metadata["clinical_source"] = "student_resources"
                d.metadata["urgency_level"] = "low"
            new_docs.extend(pdf_docs)
            print("New PDF added:", file)

if new_docs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(new_docs)
    vectordb.add_documents(chunks)
    print("New documents embedded and added to ChromaDB.")
else:
    print("No new documents to add.")

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatGoogleGenerativeAI(model=LLM_NAME, temperature=0.25, max_output_tokens=300)

prompt = ChatPromptTemplate.from_template(
    """
    You are a supportive student wellness assistant.
    Respond ONLY with coping techniques or institutional resources from the provided context.
    Do NOT give medical diagnoses, prescriptions, or unverified advice.
    If no relevant technique is found, respond empathetically and ask the student to describe their environment.

    Context:
    {context}

    Student Message:
    {question}

    Answer:
    """
)

def format_docs(docs):
    return "\n\n".join(
        f"[source: {doc.metadata.get('source','')}] "
        f"(category: {doc.metadata.get('category','')})\n{doc.page_content}"
        for doc in docs
    )

rag_chain = prompt | llm | StrOutputParser()


st.set_page_config(
    page_title='Student Wellness RAG',
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Student Wellness Assistant")
st.write("Ask for coping strategies, grounding exercises, or campus resources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input("Share how you're feeling...")

exit_commands = ["quit", "exit", "clear", "bye", "goodbye"]

if question and question.strip().lower() in exit_commands:
    with st.chat_message("assistant"):
        st.markdown("Take care! 🌸 The app will now close.")
    st.stop()

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    if not docs:
        answer = "I couldn't find that information in the provided documents."
    else:
        context = format_docs(docs)
        answer = rag_chain.invoke({"context": context, "question": question})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if docs and "i couldn't find that information" not in answer.lower():
            st.markdown("**Sources**")
            shown = []
            for doc in docs:
                source = doc.metadata["source"]
                if source not in shown:
                    st.write(f"- {source} (category: {doc.metadata.get('category','')})")
                    shown.append(source)

    st.session_state.messages.append({"role": "assistant", "content": answer})
