#Doc2Bot.py

import os
import tiktoken
import openai
import PyPDF2
import logging
import threading
import time
import sys
import chromadb

from dotenv import load_dotenv
from chatgpt import IntegratedChatGPT
from openai.datalib.pandas_helper import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY

# Setting up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('Application started.')


# Initialize Chroma client
# chroma_client = chromadb.Client()
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))

# Create a collection in Chroma to store embeddings
# chroma_client.reset()
collection_name = "document_embeddings"
collection = chroma_client.get_or_create_collection(collection_name)
logging.info('Chroma Collection created! The collection name is: %s', collection_name)

def add_embeddings_to_chroma(df):
    """Add embeddings from the dataframe to Chroma."""
    try:
        # Extract embeddings, documents, and ids
        embeddings = df['embeddings'].tolist()
        documents = df['text'].tolist()

        # Convert the range of integers to a list of strings
        ids = [str(i) for i in range(len(df))]

        # Add to Chroma collection
        collection.add(embeddings=embeddings, documents=documents, ids=ids)
        logging.info("Embeddings added to Chroma.")
    except Exception as e:
        logging.error(f"Error adding embeddings to Chroma: {e}")

# Function to show a console loading animation
def loading_animation(event):
    animation_chars = ["|", "/", "-", "\\"]
    i = 0
    sys.stdout.write(' Processing documents: ')
    while not event.is_set():
        if i > 3:
            i = 0
        sys.stdout.write(animation_chars[i] + "\r")
        sys.stdout.flush()
        i += 1
        time.sleep(0.2)
    sys.stdout.write('Done processing documents!\n')
    sys.stdout.flush()

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a given TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading TXT {txt_path}: {e}")
        return ""

def remove_newlines(serie):
    """Remove newlines and unnecessary spaces from the given pandas series."""
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def process_files(directory):
    """Process files in a given directory and convert them to a dataframe."""
    texts = []

    for file in os.listdir(directory):
        if file.endswith(".txt"):
            text = extract_text_from_txt(os.path.join(directory, file))
            texts.append((file[:-4], text))
        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(directory, file))
            texts.append((file[:-4], text))

    df = pd.DataFrame(texts, columns=['fname', 'text'])
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    
    # Check if the 'processed' directory exists, if not, create it
    if not os.path.exists('processed'):
        os.makedirs('processed')
    
    df.to_csv('processed/scraped.csv')
    logging.info("Files processed and saved to 'processed/scraped.csv'.")
    return df

# Revised process_files function to include the loading animation
def process_files_with_animation(directory):
    """Process files in a given directory and convert them to a dataframe with a loading animation."""
    stop_animation_event = threading.Event()
    t = threading.Thread(target=loading_animation, args=(stop_animation_event,))
    t.start()
    
    texts = process_files(directory)  # Call the original process_files function

    stop_animation_event.set()
    t.join()
    return texts

def tokenize_dataframe(df):
    """Tokenize the dataframe using OpenAI's tiktoken."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x, allowed_special="all")) if x.strip() != "" else 0)

    max_tokens = 500

    def split_into_many(text, max_tokens=max_tokens):
        """Split the text into many based on the given max tokens."""
        sentences = text.split('. ')
        n_tokens = [len(tokenizer.encode(" " + sentence, allowed_special="all")) for sentence in sentences]
        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence, token in zip(sentences, n_tokens):
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0
            if token > max_tokens:
                continue
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    shortened = []

    for row in df.iterrows():
        if row[1]['text'] is None:
            continue
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x, allowed_special="all")))

    try:
        # df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
        # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        #         api_key="YOUR_API_KEY",
        #         model_name="text-embedding-ada-002"
        #     )
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        df['embeddings'] = df.text.apply(lambda x: default_ef(x))
        # Ensure 'processed' directory exists
        # if not os.path.exists('processed'):
        #     os.makedirs('processed')

        add_embeddings_to_chroma(df)
        logging.info("Dataframe tokenized and embeddings saved to chromadb.")
        # df.to_csv('processed/embeddings.csv')
        # logging.info("Dataframe tokenized and embeddings saved to 'processed/embeddings.csv'.")
        return df
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return pd.DataFrame()

# New bot() function that uses the IntegratedChatGPT class
def bot():
    """Main bot loop to answer questions using IntegratedChatGPT."""

    # Check if the 'processed/embeddings.csv' file exists
    # `if not os.path.exists('processed/embeddings.csv'):
    #     logging.error("The 'processed/embeddings.csv' file does not exist. Please ensure files have been processed correctly.")
    #     print("The 'processed/embeddings.csv' file does not exist. Please ensure files have been processed correctly.")
    #     `return  # Exit the function if the file doesn't exist

    chat_bot = IntegratedChatGPT( chatbot="I am a bot trained on the provided documents.", collection=collection)

    while True:
        question = input("Ask me a question or type 'exit' to quit: ")
        
        if question.lower() == 'exit':
            break
        
        answer = chat_bot.chat(user_input=question, log_file='chat_log.txt', bot_name="DocBot")
        print(f"Answer: {answer}")

# Main execution
if __name__ == "__main__":
    directory = input("Enter the directory path containing your PDF and TXT files: ")
    
    
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        print("Directory does not exist. Please check the path and try again.")
    else:
        df = process_files_with_animation(directory)
        tokenize_dataframe(df)
        bot()
