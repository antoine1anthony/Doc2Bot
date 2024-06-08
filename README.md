# Doc2Bot - Context-aware Chatbot

An innovative chatbot system that leverages the power of GPT-4 and Chroma embeddings to deliver context-aware responses based on provided documents.

## Features

- **PDF and TXT File Processing**: Efficiently extract and process text from PDF and TXT files.
- **HTML File Processing**: Capability to extract text from HTML files, enhancing source data variety.
- **Tokenization**: Decomposes large documents into manageable tokens using OpenAI's tiktoken.
- **ChromaDB Integration**: Augments the chatbot's context-awareness by indexing and querying data using ChromaDB.
- **Multi-threading**: Utilizes multi-threading to handle concurrent data processing and API requests, improving speed and efficiency.
- **Logging**: Provides comprehensive logging for troubleshooting and monitoring purposes.
- **CLI Loading Animations**: Delivers visual feedback during lengthy operations.

## Modules

- `data_processing.py`: Includes methods for text extraction and file processing, including PDF, TXT, and HTML formats.
- `chatbot.py`: Serves as the main chatbot interaction module, presenting a user interface.
- `index_data.py`: Manages the indexing of data for enhancing bot context injection.
- `chroma_integration.py`: Contains functions for integrating ChromaDB and creating context.
- `chatgpt.py`: Acts as the core module for interacting with GPT-4, integrating additional error handling mechanisms.
- `cli_animations.py`: Manages CLI animations to enhance user interaction experience.
- `embeddings_helper.py`: Utilizes OpenAI API to convert text to embeddings, with handling for API rate limits and retries.

## Setup

1. **Clone the Repository**: Acquire the codebase to your local system.
2. **Install Dependencies**: Ensure all dependencies from `requirements.txt` are installed.
3. **Environment Variables**: Configure environment variables for OpenAI and ChromaDB in a `.env` file.
4. **Run the Bot**: Execute `chatbot.py` and follow the prompts to interact with the chatbot.

## Code Structure

### chatbot.py

Manages chatbot initialization, presenting a user interface for interaction, and manages user input and chatbot responses, incorporating loading animations and logging.

### chatgpt.py

Defines `Doc2BotGPT`, a class that manages interactions with GPT-4, handling conversation history, API interactions, and managing responses with and without context from ChromaDB.

### chroma_integration.py

Manages the integration with ChromaDB, providing functionality to create collections, add embeddings, query collections, and generate context for the chatbot.

### cli_animations.py

Handles console animations, providing visual feedback during processing phases.

### data_processing.py

Defines methods to extract text from files (PDF, TXT, HTML), perform tokenization, and process and combine data from multiple sources.

### embeddings_helper.py

Facilitates conversion of text to embeddings using OpenAI API, with robust error handling and retry mechanisms to deal with API rate limits and errors.

### index_data.py

Handles data processing and indexing to ChromaDB, including tokenization, batch processing, and asynchronous interactions.

## Notes for Users

- Ensure API keys for OpenAI and configurations for ChromaDB are securely stored and accessed.
- Be mindful of API usage, especially for embedding generation, and optimize data processing to avoid excessive API calls.
- Adjust thread counts, batch sizes, and retry/sleep parameters as per your system capabilities and API limits.
