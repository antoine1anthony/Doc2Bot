#chatbot.py

import os
import logging
import time
import threading
from chatgpt import IntegratedChatGPT
from index_data import process_data_for_bot_context_injection
from cli_animations import loading_animation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(filename="chatbot.log", level=logging.INFO)

# Define constants
BOT_NAME = "Assistant"
LOG_FILE = "chat_log.txt"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
CHATBOT_SYSTEM_MESSAGE = "This is a chatbot that uses integrated GPT-4 and Chroma embeddings."

def setup_chatbot():
    # Start the loading animation in a separate thread
    stop_animation = threading.Event()
    animation_thread = threading.Thread(target=loading_animation, args=(stop_animation,))
    animation_thread.start()

    # Start timer to calculate setup time
    start_time = time.time()

    # Initialize the integrated chatbot class
    global chatbot_instance
    chatbot_instance = IntegratedChatGPT(CHATBOT_SYSTEM_MESSAGE, CHROMA_COLLECTION_NAME)

    # Stop the loading animation
    stop_animation.set()
    animation_thread.join()

    # Calculate and print setup time
    setup_time = (time.time() - start_time) 
    print(f"Chatbot setup completed in {setup_time:.2f} seconds.")


def chat_with_user():
    """
    Interact with the user in a chat-like interface.
    """
    print(f"{BOT_NAME}: Hi! How can I assist you today?")

    while True:
        user_input = input("You: ").strip()

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"{BOT_NAME}: Goodbye! Have a great day.")
            break

        # Get the response from the integrated chatbot
        response = chatbot_instance.chat(user_input, LOG_FILE, BOT_NAME)
        print(f"{BOT_NAME}: {response}")

if __name__ == "__main__":
    process_data_for_bot_context_injection()
    setup_chatbot()
    chat_with_user()
