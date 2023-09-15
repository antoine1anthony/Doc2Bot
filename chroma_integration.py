#chroma_integration.py

import os
import logging
import openai


from dotenv import load_dotenv
from openai.datalib.pandas_helper import pandas as pd
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb import Client

from typing import List, Optional

# Setting up logging
logging.basicConfig(filename="chroma_integration.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info("Chroma Integration module started.")

# Load environment variables
load_dotenv()

# Load OPENAI API Key environment variable and set it within OpenAI API
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

# Embedding model for Chroma
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize Chroma client
chroma_client = Client(Settings(anonymized_telemetry=False, allow_reset=True))
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name=EMBEDDING_MODEL)


def create_chroma_collection(collection_name: str) -> Client:
    """Create or get a collection in Chroma."""
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_function)
    logging.info("Chroma Collection created or retrieved! The collection name is: %s", collection_name)
    return collection

def add_embeddings_to_chroma(df: pd.DataFrame, collection: Client):
    """Add embeddings from the dataframe to Chroma."""
    try:
        # Extract embeddings, documents, and ids
        embeddings = df["embeddings"].tolist()
        documents = df["text"].tolist()

        # Convert the range of integers to a list of strings
        ids = [str(i) for i in range(len(df))]

        # Add to Chroma collection
        collection.add(embeddings=embeddings, documents=documents, ids=ids)
        logging.info("Embeddings added to Chroma.")
    except Exception as e:
        logging.error(f"Error adding embeddings to Chroma: {e}")

def index_data_to_chroma(df: pd.DataFrame, collection_name: str = "default_collection"):

    """Index processed data to ChromaDB."""
    # Create or get the specified collection
    collection = create_chroma_collection(collection_name)

    # add_embeddings_to_chroma(df, collection)
    # logging.info("Data indexed to ChromaDB successfully!")

    try:
        df["embeddings"] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine="text-embedding-ada-002")["data"][0]["embedding"])

        add_embeddings_to_chroma(df, collection)
        logging.info("Data indexed to ChromaDB successfully!")
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")

def create_context(collection_name: str, question: str, max_len: int = 1800) -> Optional[str]:

    collection =  chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
    results = query_chroma_collection(collection, question, n_results=5)
    

    # For now, let's just return None since the documents are not available
    if not results.get("documents"):
        return None

    # If the documents were available, the below line would extract them
    contexts = results["documents"]

    flattened_contexts = [doc for sublist in contexts for doc in sublist]
    return "\n\n###\n\n".join(flattened_contexts) if flattened_contexts else None


def query_chroma_collection(collection: Client, question: str, n_results: int = 5) -> List[dict]:
    """Query a Chroma collection and return the results."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["distances", "documents"],
    )
    logging.debug(f"Chroma Query Results: {results}")
    return results

