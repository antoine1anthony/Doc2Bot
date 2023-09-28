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
    try:
        logging.info(f'Creating or retrieving Chroma collection: {collection_name}')
        print(f'Creating or retrieving Chroma collection: {collection_name}')
        collection = chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_function)
        logging.info(f'Chroma Collection created or retrieved! The collection name is: {collection_name}')
        print(f'Chroma Collection created or retrieved! The collection name is: {collection_name}')
        return collection
    except Exception as e:
        logging.error(f"Error in create_chroma_collection: {e}")
        print(f"Error: {e}")
        return None  # Return None in case of an error


def add_embeddings_to_chroma(df: pd.DataFrame, collection: Client):
    try:
        logging.info('Adding embeddings to Chroma.')
        print('Adding embeddings to Chroma.')
        embeddings = df["embeddings"].tolist()
        documents = df["text"].tolist()
        ids = [str(i) for i in range(len(df))]
        collection.add(embeddings=embeddings, documents=documents, ids=ids)
        logging.info('Embeddings added to Chroma.')
        print('Embeddings added to Chroma.')
    except Exception as e:
        logging.error(f"Error in add_embeddings_to_chroma: {e}")
        print(f"Error: {e}")


def index_data_to_chroma(df: pd.DataFrame, collection_name: str = "default_collection"):
    try:
        logging.info(f'Indexing data to Chroma with collection name: {collection_name}')
        print(f'Indexing data to Chroma with collection name: {collection_name}')
        collection = create_chroma_collection(collection_name)
        df["embeddings"] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine="text-embedding-ada-002")["data"][0]["embedding"])
        add_embeddings_to_chroma(df, collection)
        logging.info('Data indexed to ChromaDB successfully!')
        print('Data indexed to ChromaDB successfully!')
    except Exception as e:
        logging.error(f"Error in index_data_to_chroma: {e}")
        print(f"Error: {e}")


def query_chroma_collection(collection: Client, question: str, n_results: int = 5) -> List[dict]:
    try:
        logging.info(f'Querying Chroma collection with question: {question}')
        print(f'Querying Chroma collection with question: {question}')
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["distances", "documents"],
        )
        logging.debug(f'Chroma Query Results: {results}')
        print(f'Chroma Query Results: {results}')
        return results
    except Exception as e:
        logging.error(f"Error in query_chroma_collection: {e}")
        print(f"Error: {e}")
        return []  # Return an empty list in case of an error


def create_context(collection_name: str, question: str, max_len: int = 1800) -> Optional[str]:
    try:
        logging.info(f'Entering create_context with collection_name: {collection_name}, question: {question}, max_len: {max_len}')
        print(f'Entering create_context with collection_name: {collection_name}, question: {question}, max_len: {max_len}')

        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        logging.info('Retrieved collection from chroma_client.')
        print('Retrieved collection from chroma_client.')

        results = query_chroma_collection(collection, question, n_results=5)
        logging.info(f'Query results: {results}')
        print(f'Query results: {results}')

        # For now, let's just return None since the documents are not available
        if not results.get("documents"):
            logging.warning('No documents available in query results.')
            print('No documents available in query results.')
            return None

        # If the documents were available, the below line would extract them
        contexts = results["documents"]
        logging.info(f'Contexts extracted: {contexts}')
        print(f'Contexts extracted: {contexts}')

        flattened_contexts = [doc for sublist in contexts for doc in sublist]
        final_context = "\n\n###\n\n".join(flattened_contexts) if flattened_contexts else None
        logging.info(f'Final context: {final_context}')
        print(f'Final context: {final_context}')

        return final_context

    except Exception as e:
        logging.error(f"Error in create_context: {e}")
        print(f"Error: {e}")
        return None  # Return None in case of an error

