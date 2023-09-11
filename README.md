Embeddings-Based Question Answering System
------------------------------------------

This application processes textual content from PDF and TXT files, converting them into embeddings using OpenAI's API. It then provides a question answering system, which utilizes these embeddings to answer user queries.

### Requirements:

-   Python 3.7+
-   Libraries: `pandas`, `tiktoken`, `numpy`, `openai`, `PyPDF2`, and `logging`
-   OpenAI API Key

### Setup:

1.  Ensure you have Python 3.7 or newer installed.
2.  Install the required libraries:

    bash

    `pip install pandas tiktoken numpy openai PyPDF2`

3.  Clone this repository or download the application script.
4.  Set up your OpenAI API key. You can do this by setting an environment variable or modifying the script to include your API key.

### Usage:

1.  Run the application:

    bash

    `python app.py`

2.  When prompted, provide the directory path containing your PDF and TXT files.
3.  The application will process these files, convert the content into embeddings, and initiate the question answering system.
4.  Type your questions into the command line interface. The system will attempt to provide relevant answers based on the content of the provided files.
5.  To exit the system, type `exit`.

### Features:

-   File Processing: Supports both PDF and TXT file formats.
-   Embeddings: Uses OpenAI's API to create embeddings from the textual content.
-   Question Answering: Uses the embeddings to retrieve relevant content and answer user queries.

### Logging:

Errors and important activities are logged in `app.log`. If you encounter any issues, check this log file for more detailed information.

### Limitations:

-   The system's knowledge is based on the content of the provided files.
-   Some very specific or nuanced questions might not get accurate answers if the information isn't present in the files.
-   Ensure your OpenAI API key is valid and has the necessary permissions.

### Future Improvements:

-   Support for more file formats.
-   Enhanced error handling for different types of content and file structures.
-   Integration with a database for storing and retrieving embeddings.

### Contributions:

Contributions, bug reports, and enhancements are always welcome. Please open an issue or submit a pull request on our repository.