#chroma_integration.py

import os
import logging
import openai
# import time
from dotenv import load_dotenv
import pandas as pd
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from chromadb import Client
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(filename="chroma_integration.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OPENAI API Key environment variable and set it within OpenAI API
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model for Chroma
OPENAI_API_EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize Chroma client
CHROMA_CLIENT = Client(Settings(anonymized_telemetry=False, allow_reset=True))
OPENAI_API_EMBEDDING_FUNCTION = OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name=OPENAI_API_EMBEDDING_MODEL)

# Constants
BATCH_SIZE = 5461  # Set the batch size to the maximum allowed number of embeddings
EMBEDDING_CONVERSION_BATCH_SIZE = 100  # Set the batch size to a suitable number for your use case
MAX_THREADS = 15  # Limit the number of threads to prevent overloading

def create_chroma_collection(collection_name: str) -> Optional[Client]:
    """Create or retrieve a Chroma collection."""
    try:
        logger.info(f'Creating or retrieving Chroma collection: {collection_name}')

        collection = CHROMA_CLIENT.get_or_create_collection(collection_name, embedding_function=OPENAI_API_EMBEDDING_FUNCTION)

        logger.info(f'Chroma Collection created or retrieved! The collection name is: {collection_name}')
        return collection
    except Exception as e:
        logger.error(f"Error in create_chroma_collection: {e}")
        return None

def add_embeddings_to_chroma(df: pd.DataFrame, collection):
    """Add embeddings to Chroma in batches."""
    n_batches = -(-len(df) // BATCH_SIZE)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []

        for i in range(n_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_df = df.iloc[start_idx:end_idx]

            # Asynchronously submit data to Chroma
            futures.append(executor.submit(_submit_batch_to_chroma, batch_df, collection, i, n_batches))

        # Log results of the futures
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in future: {e}")

def _submit_batch_to_chroma(batch_df, collection, batch_index, total_batches):
    """Submit a single batch of data to Chroma and log the status."""
    try:
        logger.info(f'Submitting batch {batch_index + 1} of {total_batches}')
        documents = batch_df.text.tolist()
        ids = [str(idx) for idx in batch_df.index.tolist()]

        collection.add(
            documents=documents,
            ids=ids
        )
        logger.info(f'Successfully submitted batch {batch_index + 1} of {total_batches}')
    except Exception as e:
        logger.error(f'Error submitting batch {batch_index + 1} of {total_batches}: {e}')


def index_data_to_chroma(df: pd.DataFrame, collection_name: str = "default_collection"):
    try:
        logging.info(f'Indexing data to Chroma with collection name: {collection_name}')
        print(f'Indexing data to Chroma with collection name: {collection_name}')

        collection = create_chroma_collection(collection_name)

        # logging.info('Starting process of turning document text into embedding.')
        # print('Starting process of turning document text into embedding.')

        # embedding_conversion_start_time = time.time()  # Record the start time

        # # Split the DataFrame into batches
        # n_batches = -(-len(df) // EMBEDDING_CONVERSION_BATCH_SIZE)
        # embeddings = []

        # df['embeddings'] = embeddings

        # embedding_conversion_end_time = time.time()  # Record the end time
        # embedding_conversion_elapsed_time = (embedding_conversion_end_time - embedding_conversion_start_time) / 60 # Calculate the elapsed time

        # logging.info('Data embedding conversion finished successfully!')
        # print('Data embedding conversion finished successfully!')
        # print(f'Time taken: {embedding_conversion_elapsed_time:.2f} minutes.')

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
        # print(f'Chroma Query Results: {results}')
        return results
    except Exception as e:
        logging.error(f"Error in query_chroma_collection: {e}")
        print(f"Error: {e}")
        return []  # Return an empty list in case of an error


def create_context(collection_name: str, question: str, max_len: int = 1800) -> Optional[str]:
    try:
        logging.info(f'Entering create_context with collection_name: {collection_name}, question: {question}, max_len: {max_len}')
        print(f'Entering create_context with collection_name: {collection_name}, question: {question}, max_len: {max_len}')

        collection = CHROMA_CLIENT.get_collection(name=collection_name, embedding_function=OPENAI_API_EMBEDDING_FUNCTION)
        logging.info('Retrieved collection from CHROMA_CLIENT.')
        print('Retrieved collection from CHROMA_CLIENT.')

        results = query_chroma_collection(collection, question, n_results=5)
        logging.info(f'Query results: {results}')
        # print(f'Query results: {results}')

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

