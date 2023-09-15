#index_data.py

from chroma_integration import index_data_to_chroma
from data_processing import process_files, tokenize_dataframe
from cli_animations import loading_animation
import os
import logging
import time
import threading


from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create constants
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")




def process_data_for_bot_context_injection():
    directory = input("Enter the path to your data directory with PDF and/or TXT files (IMPORTANT! THIS WILL BE YOUR BOT'S EMBEDDINGS AND CONTEXT!): ")
    if os.path.exists(directory):

        # Start the loading animation in a separate thread
        stop_animation = threading.Event()
        animation_thread = threading.Thread(target=loading_animation, args=(stop_animation,))
        animation_thread.start()

        # Start timer to calculate processing time
        start_time = time.time()

        # Process the files
        df = process_files(directory)

        # Tokenize the dataframe
        df = tokenize_dataframe(df)

        # Stop the loading animation
        stop_animation.set()
        animation_thread.join()

        # Calculate and print processing time
        processing_time = (time.time() - start_time) / 60
        print(f"Processing completed in {processing_time:.2f} minutes.")

        index_data_to_chroma(df, COLLECTION_NAME)
    else:
        logging.error(f"Directory {directory} not found.")
        print(f"Directory {directory} not found.")

