# agents/document_agent.py
import os
from openai import AzureOpenAI
from utils.search_utils import search_documents

client = AzureOpenAI(
    api_key=os.getenv("AZURE_AI_KEY"),
    azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    api_version="2024-12-01-preview"
)
deployment = os.getenv("AI_MODEL_DEPLOYMENT")

def call_document_agent(user_query: str) -> str:
    results = search_documents(user_query)
    context = "\n\n".join([f"Titre: {doc['title']}\nContenu: {doc['content']}" for doc in results])

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": f"Voici des procédures extraites d'une base documentaire :\n{context}"},
            {"role": "user", "content": user_query}
        ],
        max_tokens=1024,
        temperature=0.5
    )

    return response.choices[0].message.content




import os
import json
import pyodbc
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv("./.env", override=True)

# Azure OpenAI configuration
api_key = os.getenv("AZURE_AI_KEY")
endpoint_ai = os.getenv("AZURE_AI_ENDPOINT")
deployment = os.getenv("AI_MODEL_DEPLOYMENT_NAME")
api_version = "2024-12-01-preview"

# Azure AI Search configuration
api_key_ai_search = os.getenv("AI_SEARCH_KEY")
endpoint_ai_search = os.getenv("AI_SEARCH_ENDPOINT")
search_index_name = os.getenv("AI_SEARCH_INDEX")

# SQL Server configuration
sql_server = os.getenv("SQL_SERVER")
sql_database = os.getenv("SQL_DATABASE")
sql_user = os.getenv("SQL_USER")
sql_password = os.getenv("SQL_PASSWORD")

# Initialize Azure OpenAI client
def get_openai_client():
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint_ai,
        api_key=api_key,
    )

# Query SQL Server for metadata
def query_sql_metadata():
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={sql_server};DATABASE={sql_database};UID={sql_user};PWD={sql_password}"
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 5 * FROM Books")  # Example query
        rows = cursor.fetchall()
        metadata = "\n".join([str(row) for row in rows])
        return metadata
    except Exception as e:
        return f"Error querying SQL Server: {e}"

# Get response from Azure OpenAI with Azure Search integration
def get_response(messages):
    client = get_openai_client()
    response = client.chat.completions.create(
        messages=messages,
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": endpoint_ai_search,
                        "index_name": search_index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": api_key_ai_search,
                        },
                        "top_n_documents": 3,
                        "fields_mapping": {
                            "title_field": "chapter_name",
                            "filepath_field": "file",
                            "content_fields": ["chunk_content", "book", "chapter_name"],
                            "vector_fields": ["vector"],
                        },
                    },
                }
            ]
        },
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        model=deployment,
    )
    return response

# Main interactive loop
def main():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in economic theory and document analysis.",
        }
    ]

    while True:
        user_input = input("User: What is your question? (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Optionally enrich with SQL metadata
        metadata = query_sql_metadata()
        if metadata:
            messages.append({"role": "system", "content": f"Here is some metadata from SQL Server:\n{metadata}"})

        # Get response from OpenAI
        response = get_response(messages)
        answer = response.choices[0].message.content
        print("\nAssistant:", answer)

        # Show citations
        citations = response.choices[0].message.context.get("citations", [])
        if citations:
            print("\nCitations:")
            for citation in citations:
                print(f" - {citation['title']}: {citation['filepath']}")
        else:
            print("\nNo citations found.")

        # Append assistant message
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

