import time
import random
import openai
import logging

MAX_RETRIES = 5  # Maximum number of retries if a request fails
BASE_SLEEP = 60 # Base sleep time (in seconds) between retries (will be multiplied in each iteration)

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
