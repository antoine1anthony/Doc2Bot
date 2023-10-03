# data_processing.py

import os
import logging
import tiktoken
import PyPDF2
from openai.datalib.pandas_helper import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_THREADS = 15  # Limit the number of threads to prevent overloading

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a given PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages])
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a given TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading TXT {txt_path}: {e}")
        return ""

def extract_text_from_html(html_path: str) -> str:
    """Extract text from a given HTML file."""
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            result = soup.get_text()
            return result
    except Exception as e:
        logging.error(f"Error reading HTML {html_path}: {e}")
        return ""

def remove_newlines(serie: pd.Series) -> pd.Series:
    """Remove newlines and unnecessary spaces from the given pandas series."""
    serie = serie.str.replace(r'(\n|\\n|  +)', ' ', regex=True)
    return serie

def process_files(directory: str) -> pd.DataFrame:
    """Process files in a given directory and convert them to a dataframe."""
    logger.info(f'Processing directory: {directory}')
    texts = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []

        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Asynchronously extract text from files
            futures.append(executor.submit(_extract_text, file, file_path))

        # Append results of the futures to texts
        for future in futures:
            try:
                texts.append(future.result())
            except Exception as e:
                logger.error(f"Error in future: {e}")

    df = pd.DataFrame(texts, columns=['fname', 'text'])
    df['text'] = df.fname + ". " + remove_newlines(df.text)

    # Check if the 'processed' directory exists, if not, create it
    if not os.path.exists('processed'):
        os.makedirs('processed')
    
    df.to_csv('processed/scraped.csv')
    logger.info("Files processed and saved to 'processed/scraped.csv'.")
    return df

def _extract_text(file, file_path):
    """Extract text from a single file and log the status."""
    try:
        if file.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.endswith('.html'):
            text = extract_text_from_html(file_path)
        elif file.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            text = ""
        return (file, text)
    except Exception as e:
        logger.error(f"Error processing file {file}: {e}")
        return (file, "")

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

