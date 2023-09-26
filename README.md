# Doc2Bot - Context-aware Chatbot

An integrated chatbot system that leverages the power of GPT-4 and Chroma embeddings to answer questions based on provided context.

## Features:

- **PDF and TXT File Processing**: Easily extract and process text from PDF and TXT files.
- **Tokenization**: Breaks down large documents into manageable tokens using OpenAI's tiktoken.
- **ChromaDB Integration**: Enhances the chatbot's context-awareness by indexing and querying data using ChromaDB.
- **Logging**: Comprehensive logging for troubleshooting and monitoring purposes.
- **CLI Loading Animations**: Provides visual feedback during time-consuming operations.

## Modules:

- `data_processing.py`: Methods for text extraction and file processing.
- `chatbot.py`: Main chatbot interaction module with user interface.
- `index_data.py`: Handles the indexing of data for bot context injection.
- `chroma_integration.py`: Functions for ChromaDB integration and context creation.
- `chatgpt.py`: Core module for interacting with GPT-4, with added error handling.

## Setup:

1. Clone the repository.
2. Install required dependencies from `requirements.txt`.
3. Set up environment variables for OpenAI and ChromaDB in a `.env` file.
4. Run `chatbot.py` and follow the prompts.

