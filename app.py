import os
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader


# -------------------------------
# Setup
# -------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found inside .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

DATA_FOLDER = "docs"
DB_FOLDER = "chromadb"

EMBEDDING_MODEL = "gemini-embedding-2"
LLM_NAME = "gemini-3.5-flash-lite"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 2

os.makedirs(DATA_FOLDER, exist_ok=True)


# -------------------------------
# Hugging Face Sentiment Model
# -------------------------------
sentiment_analyzer = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)


# -------------------------------
# RAG Setup
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

vectordb = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": TOP_K}
)

llm = ChatGoogleGenerativeAI(
    model=LLM_NAME,
    temperature=0.25,
    max_output_tokens=300
)


prompt = ChatPromptTemplate.from_template(
    """
    You are a supportive student wellness assistant.

    Respond ONLY with coping techniques or institutional resources
    from the provided context.

    Do NOT give medical diagnoses, prescriptions,
    or unverified advice.

    If no relevant technique is found, respond empathetically
    and ask the student to describe their environment.

    Context:
    {context}

    Student Message:
    {question}

    Answer:
    """
)

rag_chain = prompt | llm | StrOutputParser()


# -------------------------------
# Combiner Layer
# -------------------------------
class MessageFormat:

    def __init__(
        self,
        msg_type,
        content,
        source=None,
        sentiment=None,
        audio=None
    ):
        self.msg_type = msg_type
        self.content = content
        self.source = source
        self.sentiment = sentiment
        self.audio = audio


# -------------------------------
# Chatbot Display
# -------------------------------
def display_chat(question, sentiment_msg, rag_msg):

    # User Message
    with st.chat_message("user"):
        st.write(question)

    # Assistant Response
    with st.chat_message("assistant", avatar="🌱"):

        # Retrieved Information
        st.markdown("### 🌱 Here's something that might help")

        st.write(rag_msg.content)

        # Source
        if rag_msg.source:
            st.caption(f"📄 Source: {rag_msg.source}")

        # Sentiment Information
        label = sentiment_msg.sentiment.get(
            "label",
            "Unknown"
        )

        score = sentiment_msg.sentiment.get(
            "score",
            0.0
        )

        emoji_map = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😟",
            "love": "❤️",
            "surprise": "😲"
        }

        emoji = emoji_map.get(
            label.lower(),
            "🤔"
        )

        st.divider()

        st.caption(
            f"💭 Emotion detected: "
            f"{emoji} **{label.capitalize()}** "
            f"({score:.0%} confidence)"
        )


# -------------------------------
# Streamlit App
# -------------------------------
def main():

    # Page Configuration
    st.set_page_config(
        page_title="Wellness Assistant",
        page_icon="🌱",
        layout="centered"
    )

    # App Header
    st.title("🌱 Unified Student Wellness Assistant")

    st.caption(
        "A safe space to share how you're feeling "
        "and explore helpful wellness resources."
    )

    st.divider()


    # -------------------------------
    # Initialize Chat History
    # -------------------------------
    if "messages" not in st.session_state:

        st.session_state.messages = []


    # -------------------------------
    # Display Previous Conversations
    # -------------------------------
    for message in st.session_state.messages:

        display_chat(
            message["question"],
            message["sentiment"],
            message["rag"]
        )


    # -------------------------------
    # Chat Input
    # -------------------------------
    question = st.chat_input(
        "Share how you're feeling..."
    )


    # -------------------------------
    # Process User Input
    # -------------------------------
    if question:

        # -------------------------------
        # Sentiment Analysis
        # -------------------------------
        sentiment_result = sentiment_analyzer(
            question
        )[0]

        sentiment_msg = MessageFormat(
            msg_type="sentiment",
            content=question,
            sentiment={
                "label": sentiment_result["label"],
                "score": sentiment_result["score"]
            }
        )


        # -------------------------------
        # RAG Retrieval
        # -------------------------------
        docs = retriever.invoke(question)


        if not docs:

            rag_answer = (
                "I couldn't find that information "
                "in the provided documents."
            )

            rag_msg = MessageFormat(
                msg_type="rag",
                content=rag_answer
            )


        else:

            # Combine retrieved documents
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )


            # Generate RAG Response
            rag_answer = rag_chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )


            # Create RAG Message
            rag_msg = MessageFormat(
                msg_type="rag",
                content=rag_answer,
                source=", ".join(
                    doc.metadata.get(
                        "source",
                        ""
                    )
                    for doc in docs
                )
            )


        # -------------------------------
        # Store Conversation
        # -------------------------------
        st.session_state.messages.append(
            {
                "question": question,
                "sentiment": sentiment_msg,
                "rag": rag_msg
            }
        )


        # -------------------------------
        # Display New Conversation
        # -------------------------------
        display_chat(
            question,
            sentiment_msg,
            rag_msg
        )


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()





# import os
# import streamlit as st
# from dotenv import load_dotenv
# from transformers import pipeline

# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader

# # -------------------------------
# # Setup
# # -------------------------------
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if GOOGLE_API_KEY is None:
#     raise ValueError("GOOGLE_API_KEY not found inside .env")
# os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# DATA_FOLDER = "docs"
# DB_FOLDER = "chromadb"
# EMBEDDING_MODEL = "gemini-embedding-2"
# LLM_NAME = "gemini-3.5-flash-lite"
# CHUNK_SIZE = 400
# CHUNK_OVERLAP = 50
# TOP_K = 2
# os.makedirs(DATA_FOLDER, exist_ok=True)

# # -------------------------------
# # Hugging Face Sentiment Model
# # -------------------------------
# sentiment_analyzer = pipeline("text-classification", 
#                               model="bhadresh-savani/distilbert-base-uncased-emotion")

# # -------------------------------
# # RAG Setup
# # -------------------------------
# embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
# vectordb = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)

# retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
# llm = ChatGoogleGenerativeAI(model=LLM_NAME, temperature=0.25, max_output_tokens=300)

# prompt = ChatPromptTemplate.from_template(
#     """
#     You are a supportive student wellness assistant.
#     Respond ONLY with coping techniques or institutional resources from the provided context.
#     Do NOT give medical diagnoses, prescriptions, or unverified advice.
#     If no relevant technique is found, respond empathetically and ask the student to describe their environment.

#     Context:
#     {context}

#     Student Message:
#     {question}

#     Answer:
#     """
# )
# rag_chain = prompt | llm | StrOutputParser()

# # -------------------------------
# # Combiner Layer
# # -------------------------------
# class MessageFormat:
#     def __init__(self, msg_type, content, source=None, sentiment=None, audio=None):
#         self.msg_type = msg_type
#         self.content = content
#         self.source = source
#         self.sentiment = sentiment
#         self.audio = audio

# def display_message(msg: MessageFormat):
#     if msg.msg_type == "rag":
#         st.markdown("### 📘 Retrieved Information")
#         st.write(msg.content)
#         if msg.source:
#             st.caption(f"Source: {msg.source}")

#     elif msg.msg_type == "sentiment":
#         label = msg.sentiment.get("label", "Unknown")
#         score = msg.sentiment.get("score", 0.0)
#         emoji_map = {
#             "joy": "😊", "sadness": "😢", "anger": "😠",
#             "fear": "😟", "love": "❤️", "surprise": "😲"
#         }
#         emoji = emoji_map.get(label.lower(), "🤔")
#         st.markdown(f"**Sentiment:** {emoji} {label} (Confidence: {score:.2f})")

#     elif msg.msg_type == "voice":
#         st.markdown("🎙️ Voice Response (placeholder)")
#         st.write(msg.content if msg.content else "No voice model integrated yet.")

# # -------------------------------
# # Streamlit App
# # -------------------------------
# def main():
#     st.set_page_config(page_title="Unified Wellness Assistant", page_icon="🌱", layout="wide")
#     st.title("🌱 Unified Student Wellness Assistant")

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     question = st.chat_input("Share how you're feeling...")

#     if question:
#         # Sentiment Analysis
#         sentiment_result = sentiment_analyzer(question)[0]
#         sentiment_msg = MessageFormat(
#             msg_type="sentiment",
#             content=question,
#             sentiment={"label": sentiment_result["label"], "score": sentiment_result["score"]}
#         )

#         # RAG Retrieval
#         docs = retriever.invoke(question)
#         if not docs:
#             rag_answer = "I couldn't find that information in the provided documents."
#             rag_msg = MessageFormat(msg_type="rag", content=rag_answer)
#         else:
#             context = "\n\n".join(doc.page_content for doc in docs)
#             rag_answer = rag_chain.invoke({"context": context, "question": question})
#             rag_msg = MessageFormat(
#                 msg_type="rag",
#                 content=rag_answer,
#                 source=", ".join([doc.metadata.get("source", "") for doc in docs])
#             )

#         # Store & Display
#         st.session_state.messages.append(sentiment_msg)
#         st.session_state.messages.append(rag_msg)

#     # Render all messages
#     for msg in st.session_state.messages:
#         display_message(msg)
#         st.divider()

# if __name__ == "__main__":
#     main()
