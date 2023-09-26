#chatgpt.py
import openai
import logging
from chroma_integration import  create_context
from typing import List, Optional, Dict, Any

# Define an exception for the ChatGPT class for better error handling.
class ChatGPTError(Exception):
    pass

class IntegratedChatGPT:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "temperature": 0.75,
        "frequency_penalty": 0.2,
        "presence_penalty": 0
    }

    def __init__(self, chatbot: str, collection_name: str, retries: int = 3):
        self.chatbot = chatbot
        self.conversation: List[Dict[str, str]] = []  # To store the conversation history
        self.retries = retries
        self.collection_name = collection_name

    def chat(self, user_input: str, log_file: str, bot_name: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})


        # Create context from embeddings
        context = create_context(self.collection_name, user_input)

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

    def is_repetitive(self, response: str) -> bool:
        for message in self.conversation[-3:]:
            if message['content'] == response:
                return True
        return False

    def chatgpt(self, conversation: List[Dict[str, str]], chatbot: str, user_input: str, **kwargs) -> str:
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

    def chatgpt_with_retry(self, conversation: List[Dict[str, str]], chatbot: str, user_input: str, **kwargs) -> Optional[str]:
        for i in range(self.retries):
            try:
                return self.chatgpt(conversation, chatbot, user_input, **kwargs)
            except openai.api.error.APIError as e:
                logging.warning(f"Error in chatgpt attempt {i + 1}: {e}. Retrying...")
            except Exception as e:
                logging.error(f"Unexpected error in chatgpt attempt {i + 1}: {e}. No more retries.")
                raise ChatGPTError from e
        return None

    

    def answer_using_embeddings(self, question: str, context: str, **kwargs) -> str:
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
