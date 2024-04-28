import streamlit as st

from langchain_community.chat_models import ChatOllama
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# from templates.htmlTemplates import css, bot_template, user_template
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import Chroma

# from data.vector import *
import chromadb
from chromadb.config import Settings
import config as cfg

from components.sidebar import sidebar
# TODO: Setup prompts
# TODO: Setup pre-populated vector db


def get_vectordb():
    """
    Connect to VectorDB (Chroma)
    """

    client = chromadb.HttpClient(
        host=cfg.CHROMA_HOST,
        port=cfg.CHROMA_PORT,
        settings=Settings(
            allow_reset=True,
        ),
    )

    # TODO: Delete these once working
    #
    # vectorstore = Chroma(
    #     persist_directory=cfg.RAG_PERSIST_DIR,
    #     embedding_function=embedding_function,
    # )

    #    embedding_function = SentenceTransformerEmbeddings()

    embeddings_model = SentenceTransformerEmbeddings()

    vectorstore = Chroma(
        client=client,
        embedding_function=embeddings_model,
        collection_name=cfg.RAG_COLLECTION_NAME,
    )

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOllama(
        model=cfg.MODEL,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content).replace(
                    "{{USER_AVATAR}}", cfg.USER_AVATAR
                ),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content).replace(
                    "{{BOT_AVATAR}}", cfg.BOT_AVATAR
                ),
                unsafe_allow_html=True,
            )


# def setup_streamlit_ui():
#     st.set_page_config(
#         page_title="Chat App: Pure LLM Demo",
#         page_icon="📖",
#         layout="wide",
#         initial_sidebar_state="collapsed",
#     )
#     # st.header("📖 Chat App: Non RAG Demo")
#     sidebar()


def main():
    # load_dotenv()
    # st.set_page_config(page_title=cfg.STREAMLIT_PAGE_TITLE)
    st.set_page_config(
        page_title=cfg.STREAMLIT_PAGE_TITLE,
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    sidebar()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(cfg.STREAMLIT_BOT_TITLE)
    # st.header("RAG BOT")

    user_question = st.text_input(cfg.USER_PROMPT)
    if user_question:
        handle_userinput(user_question)

    vectorstore = get_vectordb()
    st.session_state.conversation = get_conversation_chain(
        vectorstore
    )  # create conversation chain


if __name__ == "__main__":
    main()
