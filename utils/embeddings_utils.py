# utils/embeddings_utils.py
import os
import uuid
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

load_dotenv()
api_key = os.getenv("AZURE_AI_KEY")
endpoint = os.getenv("AZURE_AI_ENDPOINT_EMBEDDINGS")
embeddings_model_deployment = os.getenv("AZURE_AI_EMBEDDINGS_MODEL_DEPLOYMENT")

def get_embedding(text: str) -> list[float]:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview"
    )
    response = client.embeddings.create(
        input=[text],
        model=embeddings_model_deployment
    )
    return response.data[0].embedding
def get_client():
    client = EmbeddingsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    return client

def get_embeddings_vector(text):
    response = get_client().embed(
    input=text,
    model=embeddings_model_deployment
    )
    embedding = response.data[0].embedding
    return embedding

def get_chunk_object(chapter:dict, input_directory)-> dict:
    with open(f"{input_directory}/{chapter['file']}", "r") as f:
        chunk_content = f.read()
        vector = get_embeddings_vector(chunk_content)
    return {
        "id": str(uuid.uuid4()),
        'book': chapter["book"],
        'book_name': chapter["book_name"],
        'chapter': chapter["chapter"],
        'chapter_name': chapter["chapter_name"],
        'file': chapter["file"],
        'chunk_content': chunk_content,
        'vector': vector
    }
