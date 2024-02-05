# embeddings_helper.py

import time
import random
import openai
import logging

from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from gpu_helper import DEVICE

MODEL_ID: str = "openai/clip-vit-base-patch32"

MODEL = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
TOKENIZER = CLIPTokenizerFast.from_pretrained(MODEL_ID)
PROCESSOR = CLIPProcessor.from_pretrained(MODEL_ID)

MAX_RETRIES: int = 5  # Maximum number of retries if a request fails
BASE_SLEEP: int = 60 # Base sleep time (in seconds) between retries (will be multiplied in each iteration)

def convert_text_to_embedding(text: str) -> list:
    """Convert a text to embedding using OpenAI API with rate limit handling."""
    retries = 0  # Counter for retries
    
    while retries < MAX_RETRIES:
        try:
            return openai.Embedding.create(input=text, engine="text-embedding-ada-002")["data"][0]["embedding"]
        except openai.error.OpenAIError as e:
            # Log the error message
            logging.error(f"Error converting text to embedding: {str(e)}")
            
            # Check if the error is due to rate limiting
            if "rate limit" in str(e).lower():
                # Calculate sleep time with exponential backoff and random jitter
                sleep_time = (2 ** retries) * BASE_SLEEP + random.uniform(0, 0.1 * (2 ** retries))
                
                logging.info(f"Rate limit reached. Retrying in {sleep_time:.2f} seconds...")
                
                # Sleep for the calculated duration
                time.sleep(sleep_time)
                
                # Increment the retries counter
                retries += 1
            else:
                # If the error is not due to rate limiting, re-raise the exception
                raise
        except Exception as e:
            logging.error(f"Unexpected error converting text to embedding: {str(e)}")
            raise
    # If we've exhausted retries, return an empty list to indicate failure
    logging.error("Maximum retries reached. Returning an empty embedding.")
    return []

def convert_text_to_embedding_CLIP(text: str) -> list:
    """Converts text to embedding using OpenAI CLIP model"""

    try:
        # Tokenize and truncate the text to the maximum length that the model can handle
        inputs = TOKENIZER(text, truncation=True, max_length=77, return_tensors="pt")

        # Move the tensor to the same device as the model
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        # Get text features (embeddings) from the model
        text_embedding = MODEL.get_text_features(**inputs)

        print(f"Number of Dimensional Vectors: {text_embedding.shape}")
        return text_embedding

    except Exception as e:
        logging.error(f"Unexpected error converting text to embedding with CLIP: {str(e)}")
        print(f"Unexpected error converting text to embedding with CLIP: {str(e)}")
        return torch.Tensor()

class CLIPEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = convert_text_to_embedding_CLIP(texts).cpu().detach().numpy()

        return embeddings