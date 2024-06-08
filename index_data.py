# index_data.py

from chroma_integration import index_data_to_chroma
from data_processing import process_files, tokenize_dataframe
from cli_animations import loading_animation
from dotenv import load_dotenv
import os
import logging
import time
import threading
import pandas as pd

# Load environment variables
load_dotenv()

# Create constants
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")


def process_data_for_bot_context_injection():
    # directories_input = input("Enter the paths to your data directories separated by commas (IMPORTANT! THIS WILL BE YOUR BOT'S EMBEDDINGS AND CONTEXT!): ")
    # directories = directories_input.split(',')

    # directories = [
    #     "python",
    #     "node.js",
    #     "typescript",
    #     "javascript",
    #     "unity_manual_data",
    #     "unity_script_reference_data",
    #     "react_native_data",
    #     "godot",
    #     "postgresql",
    #     "mongodb",
    #     "c++_data",
    #     "blender_data",
    #     "aws_sage_maker",
    #     "aws_s3",
    #     "aws_lambda",
    #     "aws_documentdb",
    #     "webassembly"
    #     ]

    directories = ["c++_data", "react_native_data"]

    all_dataframes = []  # List to store dataframes from each directory

    for directory in directories:
        directory = directory.strip()  # Remove any leading/trailing whitespace
        if os.path.exists(directory):

            # Start the loading animation in a separate thread
            stop_animation = threading.Event()
            animation_thread = threading.Thread(
                target=loading_animation, args=(stop_animation,))
            animation_thread.start()

            # Start timer to calculate processing time
            start_time = time.time()

            # Process the files
            df = process_files(directory)

            # Stop the loading animation
            stop_animation.set()
            animation_thread.join()

            # Calculate and print processing time
            processing_time = (time.time() - start_time) / 60
            print(f"Processing {directory} completed in {processing_time:.2f} minutes.")

            # Append the processed dataframe to the list
            all_dataframes.append(df)
        else:
            logging.error(f"Directory {directory} not found.")
            print(f"Directory {directory} not found.")

    # Combine all dataframes into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Tokenize the combined dataframe
    combined_df = tokenize_dataframe(combined_df)

    index_data_to_chroma(combined_df, COLLECTION_NAME)
