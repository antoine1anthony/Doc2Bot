# chatgpt.py
import openai
import logging
import tiktoken
from chroma_integration import create_context
from typing import List, Dict, Optional, Any
import json
from tools.tools import create_documentation, save_documents, tools
from tenacity import retry, stop_after_attempt, wait_random_exponential

class ChatGPTError(Exception):
    pass

class Doc2BotGPT:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "temperature": 0.75,
        "frequency_penalty": 0.2,
        "presence_penalty": 0
    }
    MAX_TOKENS: int = 64000

    def __init__(self, api_key: str, chatbot: str, collection_name: str, retries: int = 3):
        self.api_key = api_key
        self.chatbot = chatbot
        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation: List[Dict[str, str]] = []
        self.retries = retries
        self.collection_name = collection_name
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.tools = tools
        openai.api_key = self.api_key

    def trim_conversation_to_fit_token_limit(self, conversation, max_tokens=128000):
        encoded_convo = [self.encoder.encode(
            msg["content"]) for msg in conversation]
        total_tokens = sum(len(tokens) for tokens in encoded_convo)

        while total_tokens > max_tokens and len(conversation) > 2:
            removed_message = conversation.pop(1)
            removed_tokens = len(self.encoder.encode(
                removed_message["content"]))
            total_tokens -= removed_tokens

        return conversation

    def process_chunks(self, chunks, log_file, bot_name):
        full_response = ""
        total_chunks = len(chunks)
        chunks_processed = 0

        for i, chunk in enumerate(chunks):
            chunk_text = self.encoder.decode(chunk)
            header = f"Chunk: {i+1} out of {total_chunks}\n"
            footer = "Please respond now, all chunks have been sent." if i == total_chunks - 1 else f"respond with an empty string until Chunk {total_chunks}"
            chunk_text = f"{header}{chunk_text}\n{footer}"
            self.conversation.append({"role": "user", "content": chunk_text})

            chunks_processed += 1
            logging.info(f"Processing chunk {chunks_processed}/{total_chunks}")
            print(f"Processing chunk {chunks_processed}/{total_chunks}...")

            response = self.chatgpt_with_retry(conversation=self.conversation)

            self.conversation.append(
                {"role": "assistant", "content": response.content})
            full_response += response.content

            logging.info(f"Chunk {chunks_processed} processed. User: {chunk_text[:50]}... Assistant: {response.content[:50]}...")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"User: {chunk_text}\n{bot_name}: {response.content}\n\n")

            print(f"Chunk {chunks_processed}/{total_chunks} processed.")

            if len(self.conversation) > 4:
                self.conversation.pop(2)
                self.conversation.pop(2)

        logging.info("All chunks processed successfully.")
        print("All chunks processed successfully.")

        return full_response

    def chat(self, user_input: str, log_file: str, bot_name: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})

        # Create context from embeddings
        context = create_context(self.collection_name, user_input)

        response = self.chatgpt_with_retry(self.conversation, self.chatbot, user_input, context)

        self.conversation.append({"role": "assistant", "content": response})
        with open(log_file, 'a') as f:
            f.write(f"User: {user_input}\n{bot_name}: {response}\n\n")
        
        if len(self.conversation) > 4:
            self.conversation.pop(0)
        
        return response

    def chatgpt(self, conversation: List[Dict[str, str]], chatbot: str, user_input: str, context: str = "", tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        params = {**self.DEFAULT_PARAMS, **kwargs}
        messages_input = conversation.copy()
        
        if self.tools:
            print(f'ping...{self.tools}')
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=params["temperature"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"],
                messages=messages_input,
                tools=self.tools,
                tool_choice="auto"
            )
        else:
            print(f'ping...no tools...{self.tools}')
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=params["temperature"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"],
                messages=messages_input
            )

        chat_response = completion.choices[0].message
        
        tool_calls = chat_response.tool_calls
        # create_documentation, create_unit_tests
        if tool_calls:
            available_functions = {
                "create_documentation": create_documentation,
                "save_documents": save_documents
            }

            messages_input.append(chat_response)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                # Check and ensure all required arguments are present
                if function_name == "create_documentation":
                    function_response = function_to_call(function_args["code"])
                elif function_name == "save_documents":
                    function_response = function_to_call(function_args["document_content"], function_args["file_name"], function_args["file_extension"])

                # Add function response as a message with the role 'assistant'
                messages_input.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),  # Ensure the content is a JSON object
                })

            second_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages_input
            )

            print(f'Second message: {second_response.choices[0].message.content}')
            return second_response.choices[0].message.content

        return chat_response.content

    @retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(10), reraise=True)
    def chatgpt_with_retry(self, conversation: List[Dict[str, str]], chatbot: str, user_input: str, context: str = "", tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Optional[str]:
        try:
            conversation = self.trim_conversation_to_fit_token_limit(
                conversation, 128000)
            return self.chatgpt(conversation, chatbot, user_input, context, tools, **kwargs)
        except openai.APIStatusError as e:
            logging.error(f"Error during chat completion: {e}")
            raise ChatGPTError from e

    def answer_using_embeddings(self, question: str, context: str, **kwargs) -> str:
        params = {**self.DEFAULT_PARAMS, **kwargs}
        
        conversation_with_context = self.conversation.copy()
        prompt_message = {
            "role": "user",
            "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
        }
        conversation_with_context.append(prompt_message)

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=params["temperature"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"],
            messages=conversation_with_context,
            tools=tools,
            tool_choice="auto"
        )

        chat_response = completion.choices[0].message
        return chat_response
