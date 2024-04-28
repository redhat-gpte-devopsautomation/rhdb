from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

import os
import uuid
import config as cfg


def load_chunk_persist_pdf() -> Chroma:
    """
    load, chunk, embed the .pdf contents of a directory
    """

    pdf_folder_path = cfg.RAG_INPUTS
    print(f"RAG INPUTS: {pdf_folder_path}")
    documents = []

    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            print(f"Processing {file}")
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(
        chunk_size=cfg.RAG_CHUNK_SIZE,
        chunk_overlap=cfg.RAG_CHUNK_OVERLAP,
    )

    chunked_documents = text_splitter.split_documents(documents)

    client = chromadb.HttpClient(
        host=cfg.CHROMA_HOST,
        port=cfg.CHROMA_PORT,
        settings=Settings(
            allow_reset=True,
        ),
    )

    collection = client.get_or_create_collection(cfg.RAG_COLLECTION_NAME)

    for chunk in chunked_documents:
        collection.add(
            ids=[str(uuid.uuid1())],
            metadatas=chunk.metadata,
            documents=chunk.page_content,
        )

    embeddings_model = SentenceTransformerEmbeddings()

    vectordb = Chroma(
        client=client,
        embedding_function=embeddings_model,
        collection_name=cfg.RAG_COLLECTION_NAME,
    )

    return vectordb


def main():
    load_chunk_persist_pdf()


if __name__ == "__main__":
    main()
