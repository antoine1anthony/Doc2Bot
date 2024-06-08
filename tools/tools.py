from openai import OpenAI
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OpenAI_API_KEY'))


def create_documentation( code: str) -> str:
    prompt = f"Create documentation for the following code:\n\n{code}\n\nDocumentation:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    )
    documentation = response.choices[0].message.content
    return documentation

# Function to save documents of any file type
def save_documents(document_content, file_name, file_extension):
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{file_name}_{date_today}.{file_extension}"

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(document_content)

    return filename

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_documentation",
            "description": "Create documentation for the provided code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_documents",
            "description": "Save documents of any file type. Use this to write new code, content, documentation, or anything else the user needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_content": {"type": "string"},
                    "file_name": {"type": "string"},
                    "file_extension": {"type": "string"}
                },
                "required": ["document_content", "file_name", "file_extension"]
            }
        }
    }
]
