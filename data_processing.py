#data_processing.py

import os
import logging
import tiktoken
import openai
import PyPDF2
from dotenv import load_dotenv
from openai.datalib.pandas_helper import pandas as pd
from openai.embeddings_utils import distances_from_embeddings

from typing import List, Union, Tuple

# Load environment variables
load_dotenv()

# Setting up logging
logging.basicConfig(filename='data_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('Data Processing module started.')

def extract_text_from_pdf(pdf_path: str) -> str:
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

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a given TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading TXT {txt_path}: {e}")
        return ""

def remove_newlines(serie: pd.Series) -> pd.Series:
    """Remove newlines and unnecessary spaces from the given pandas series."""
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def process_files(directory: str) -> pd.DataFrame:
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

def tokenize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tokenize the dataframe using OpenAI's tiktoken."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x, allowed_special="all")) if x.strip() != "" else 0)

    max_tokens = 500

    def split_into_many(text: str, max_tokens: int = max_tokens) -> List[str]:
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

    return df

