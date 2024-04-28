import streamlit as st

# from ragnar.components.faq import faq
from components.faq_llm import faq


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "- Test the chatbot by typing in the input box with questions outside its knowledge.\n"
            "  - Events since the LLM's last training\n"
            "  - **Safe** Internal or private information\n"
        )

        faq()

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            """
            📖 This App is an AI Chatbot based on:
            - LangChain
            - Streamlit
            - Chromadb (RAG)
            """
        )
        # st.markdown("Made by [DR. AMJAD RAZA](https://www.linkedin.com/in/amjadraza/)")
        # st.markdown("Credits for Template [hwchase17](https://github.com/hwchase17/langchain-streamlit-template)")
        st.markdown("---")
