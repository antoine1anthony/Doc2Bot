#chatgpt.py
import openai
import logging

from openai.embeddings_utils import distances_from_embeddings
from openai.datalib.numpy_helper import numpy as np
from openai.datalib.pandas_helper import pandas as pd

# Define an exception for the ChatGPT class for better error handling.
class ChatGPTError(Exception):
    pass

class IntegratedChatGPT:
    DEFAULT_PARAMS = {
        "temperature": 0.75,
        "frequency_penalty": 0.2,
        "presence_penalty": 0
    }

    def __init__(self, chatbot, embeddings_csv_path, retries=3):
        """
        Initialize the IntegratedChatGPT class.

        Parameters:
        - api_key (str): OpenAI API key.
        - chatbot (str): System message for chatbot.
        - embeddings_csv_path (str): Path to the embeddings CSV file.
        - retries (int): Number of retries for the chat API call in case of failure.
        """
        self.chatbot = chatbot
        self.conversation = []  # To store the conversation history
        self.retries = retries
        self.df = self.load_embeddings(embeddings_csv_path)  # Load embeddings from the CSV

    def load_embeddings(self, embeddings_csv_path):
        """
        Load embeddings from a CSV file.

        Parameters:
        - embeddings_csv_path (str): Path to the embeddings CSV file.

        Returns:
        - DataFrame: DataFrame containing the loaded embeddings.
        """
        df = pd.read_csv(embeddings_csv_path, index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        return df

    def chat(self, user_input, log_file, bot_name):
        """
        Facilitate a chat interaction using user input.

        Parameters:
        - user_input (str): Input provided by the user.
        - log_file (str): File to log the conversation.
        - bot_name (str): Name of the chatbot to display in logs.

        Returns:
        - str: Response from the chatbot.
        """
        self.conversation.append({"role": "user", "content": user_input})

        # Create context from embeddings
        context = self.create_context(user_input, self.df)

        # If a context is available, answer using that context, otherwise revert to a standard GPT chat.
        if context:
            response = self.answer_using_embeddings(user_input, context)
        else:
            response = self.chatgpt_with_retry(self.conversation, self.chatbot, user_input)

        # Log the chatbot's response and store it in the conversation history.
        self.conversation.append({"role": "assistant", "content": response})

        # Log the conversation to the specified log file.
        with open(log_file, 'a') as f:
            f.write(f"User: {user_input}\n")
            f.write(f"{bot_name}: {response}\n\n")

        # Keep the conversation history to a fixed length.
        if len(self.conversation) > 4:
            self.conversation.pop(0)
        return response

    def is_repetitive(self, response):
        """
        Check if a response is repetitive in the last three messages.

        Parameters:
        - response (str): The response to check.

        Returns:
        - bool: True if repetitive, otherwise False.
        """
        for message in self.conversation[-3:]:
            if message['content'] == response:
                return True
        return False

    def chatgpt(self, conversation, chatbot, user_input, **kwargs):
        """
        Facilitate a chat interaction using OpenAI's chat API.

        Parameters:
        - conversation (list): History of the conversation.
        - chatbot (str): System message for chatbot.
        - user_input (str): Input provided by the user.
        - **kwargs: Additional parameters for the chat API call.

        Returns:
        - str: Response from the chatbot.
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}
        
        messages_input = conversation.copy()
        messages_input.insert(0, {"role": "system", "content": chatbot})

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=params["temperature"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"],
                messages=messages_input)
            
            chat_response = completion['choices'][0]['message']['content']
        except openai.api.error.APIError as e:
            logging.warning(f"Error during chat completion: {e}")
            raise ChatGPTError("Error during chat completion.") from e

        # If the response is repetitive, try again.
        if self.is_repetitive(chat_response):
            return self.chatgpt(conversation, chatbot, user_input, **kwargs)
        else:
            return chat_response

    def chatgpt_with_retry(self, conversation, chatbot, user_input, **kwargs):
        """
        Retry the chatgpt function in case of failures.

        Parameters:
        - conversation (list): History of the conversation.
        - chatbot (str): System message for chatbot.
        - user_input (str): Input provided by the user.
        - **kwargs: Additional parameters for the chat API call.

        Returns:
        - str: Response from the chatbot.
        """
        for i in range(self.retries):
            try:
                return self.chatgpt(conversation, chatbot, user_input, **kwargs)
            except openai.api.error.APIError as e:
                logging.warning(f"Error in chatgpt attempt {i + 1}: {e}. Retrying...")
            except Exception as e:
                logging.error(f"Unexpected error in chatgpt attempt {i + 1}: {e}. No more retries.")
                raise ChatGPTError from e
        return None

    def create_context(self, question, df, max_len=1800, size="ada"):
        """
        Create a context using embeddings for the chatbot.

        Parameters:
        - question (str): Question provided by the user.
        - df (DataFrame): DataFrame containing embeddings.
        - max_len (int): Maximum token length for context.
        - size (str): Model size for embeddings.

        Returns:
        - str: Context for the chatbot.
        """
        try:
            q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
            df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
        except Exception as e:
            logging.error(f"Error creating context: {e}")
            raise ChatGPTError("Error creating context.") from e

        returns = []
        cur_len = 0

        for i, row in df.sort_values('distances', ascending=True).iterrows():
            cur_len += row['n_tokens'] + 4
            if cur_len > max_len:
                break
            returns.append(row["text"])

        return "\n\n###\n\n".join(returns)

    def answer_using_embeddings(self, question, context, **kwargs):
        """
        Generate an answer using a provided context.

        Parameters:
        - question (str): Question provided by the user.
        - context (str): Context for the chatbot.
        - **kwargs: Additional parameters for the chat API call.

        Returns:
        - str: Response from the chatbot.
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}
        
        # Add the specific prompt related to the context and question to the conversation
        conversation_with_context = self.conversation.copy()
        prompt_message = {
            "role": "user",
            "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        }
        conversation_with_context.append(prompt_message)

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=params["temperature"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"],
                messages=conversation_with_context)
            
            chat_response = completion['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error during context-based completion: {e}")
            raise ChatGPTError("Error during context-based completion.") from e

        # If the response is repetitive, try again.
        if self.is_repetitive(chat_response):
            return self.answer_using_embeddings(question, context, **kwargs)
        else:
            return chat_response
