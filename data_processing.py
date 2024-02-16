# data_processing.py

import os
import logging
import tiktoken
import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from concurrent.futures import ThreadPoolExecutor
import moviepy.editor as mp
from openai import OpenAI

client = OpenAI()
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Setup logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_THREADS = 15  # Limit the number of threads to prevent overloading

SUPPORTED_EXTENSIONS = [
    '.ts', '.tsx', '.js', '.jsx', '.cpp', '.css', '.json',
    '.kt', '.swift', '.md', '.java', '.php', '.txt', '.mov', '.mp4',
    '.py', '.go', '.rs', '.rb', '.sh', '.html', '.pdf', '.avi', '.wmv'
    # Add other supported file extensions here
]

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


def traverse_directory(directory):
    """Traverse the directory and list all supported code files."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                yield os.path.join(root, file)


def process_files(directory: str) -> pd.DataFrame:
    """Process files in a given directory and convert them to a dataframe."""
    logger.info(f'Processing directory: {directory}')
    texts = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []

        for file_path in traverse_directory(directory):
            # Asynchronously extract text from files
            futures.append(executor.submit(_extract_text, os.path.basename(file_path), file_path))

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


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Extract audio from a given video file.
    """
    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        clip.close()
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Whisper API and delete the audio file afterward.
    """
    try:
        print('running transcribe_audio')
        print('audio_path: %s' % audio_path)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcribe("whisper-1", audio_file)
            print(f'Transcript: {transcript}')
        # Delete the temporary audio file after transcription
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return transcript.text
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {e}")
        return ""


def transcribe_large_audio(audio_path: str, chunk_length: int = 10000) -> str:
    """
    Transcribe audio that might be larger than the API's maximum limit.
    
    :param audio_path: Path to the audio file.
    :param chunk_length: Length of chunks to split the audio (in ms). Default is 10000ms (10s).
    :return: Transcribed text.
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="mp3")
        chunks = split_on_silence(
            audio,
            min_silence_len=500,  # must be silent for at least 500ms
            silence_thresh=-40,    # consider it silent if quieter than -40 dBFS
            keep_silence=500,      # keep 500ms of leading/trailing silence
        )
        
        # Fallback: If silence splitting creates too large chunks, split audio into fixed-size chunks
        if any(len(chunk) > chunk_length for chunk in chunks):
            chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        
        transcriptions = []
        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.mp3"
            chunk.export(chunk_path, format="mp3")
            transcriptions.append(transcribe_audio(chunk_path))
        
        return " ".join(transcriptions)
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {e}")
        return ""


def extract_text_from_video(video_path: str) -> str:
    """
    Extract and transcribe text from a given video file.
    """
    # Specify path for temporary audio file
    audio_path = "temp_audio.mp3"

    # Extract Audio
    extract_audio_from_video(video_path, audio_path)

    # Transcribe Audio
    transcript_text = transcribe_large_audio(audio_path)
    
    # print(f'Transcribe Audio from {transcript_text}')
    return transcript_text


def _extract_text(file, file_path):
    """Extract text from a single file and log the status."""
    try:
        extension = os.path.splitext(file)[1].lower()

        if extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif extension == '.html':
            text = extract_text_from_html(file_path)
        elif extension in ['.txt', '.md', '.css', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.json', '.kt', '.swift', '.java', '.php', '.py', '.go', '.rs', '.rb', '.sh']:
            text = extract_text_from_txt(file_path)
        elif extension in ['.mov', '.mp4', '.avi', '.wmv']:
            text = extract_text_from_video(file_path)
        else:
            logger.info(f"Unsupported file extension {extension} for file {file}")
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

