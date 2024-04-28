import chromadb
from chromadb.config import Settings
import config as cfg
import pandas as pd


def chromadb_connect():
    """
    Connect to chroma instance
    """

    client = chromadb.HttpClient(
        host=cfg.CHROMA_HOST,
        port=cfg.CHROMA_PORT,
        settings=Settings(
            allow_reset=True,
        ),
    )

    return client


def view_collections(client):
    """
    View collection on client
    """

    # print(f"RAG Persist Path: {cfg.RAG_COLLECTION_NAME}")
    # TODO: This function should return only ONE collection

    print(client.list_collections())
    print(client.tenant)

    print("Collections")

    for collection in client.list_collections():
        print(collection.name)
        data = collection.get()

        # ids = data["ids"]
        # embeddings = data["embeddings"]
        # metadata = data["metadatas"]
        # documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        print(f"### Collection: {collection.name}")
        print(df)
        df["ids"] = df["ids"].str.slice(0, 9)
        # df["data"] = df["documents"].str.slice(0, 24)
        print(f"Shape/dimensions: {df.shape}")
        print(df.columns)
        print(f'{df["ids"]}')
        # embedding_length = df["embeddings"].apply(len)
        # embedding length')  # = {embedding_length}')
        #' = embeddings"].apply(len)}')
    #' {df["documents"]}')


if __name__ == "__main__":
    try:
        print(f"Opening database: {cfg.RAG_COLLECTION_NAME}")
        client = chromadb_connect()
        view_collections(client)
    except:
        pass
