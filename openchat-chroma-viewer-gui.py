import chromadb
from chromadb.config import Settings
import pandas as pd
import streamlit as st
import config as cfg


def view_collections(collection_name):
    st.markdown(f"### DB Path: {cfg.RAG_COLLECTION_NAME}")

    client = chromadb.HttpClient(
        host=cfg.CHROMA_HOST,
        port=cfg.CHROMA_PORT,
        settings=Settings(allow_reset=True),
    )

    print(client.list_collections())

    st.header("Collections")

    for collection in client.list_collections():
        data = collection.get()

        # ids = data["ids"]
        # embeddings = data["embeddings"]
        # metadata = data["metadatas"]
        # documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        st.markdown("### Collection: **%s**" % cfg.RAG_COLLECTION_NAME)
        st.dataframe(df)


if __name__ == "__main__":
    try:
        print(f"Opening database: {cfg.RAG_COLLECTION_NAME}")
        view_collections(cfg.RAG_COLLECTION_NAME)
    except:
        pass
