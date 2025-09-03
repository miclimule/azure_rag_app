# utils/search_utils.py
import os
import json

from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from utils.embeddings_utils import get_embedding
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticSearch,
    SemanticField
)

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

credential = AzureKeyCredential(search_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

def index_documents(documents: list):
    docs_with_vector = []
    for doc in documents:
        docs_with_vector.append({
            "id": str(doc["id"]),
            "title": doc["title"],
            "content": doc["content"],
            "content_vector": get_embedding(doc["content"])
        })

    result = search_client.upload_documents(documents=docs_with_vector)
    return result

def search_documents(query: str, k: int = 3):
    vector = get_embedding(query)

    results = search_client.search(
        search_text=None,
        vectors=[
            {
                "value": vector,
                "fields": "content_vector",
                "k": k,
                "kind": "vector"
            }
        ],
        select=["id", "title", "content"]
    )

    return [doc for doc in results]

api_key = os.getenv("AI_SEARCH_KEY")
endpoint = os.getenv("AI_SEARCH_ENDPOINT")

credential = AzureKeyCredential(api_key)
azure_search_service_endpoint = endpoint

def get_search_index_client(search_index_name):

    return SearchIndexClient(
        endpoint=azure_search_service_endpoint, 
        index_name=search_index_name, 
        credential=credential
    )

# create search index

def create_search_index(search_index_name):

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            sortable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="book", type=SearchFieldDataType.String),
        SearchableField(name="book_name", type=SearchFieldDataType.String),
        SearchableField(name="chapter", type=SearchFieldDataType.String),
        SearchableField(name="chapter_name", type=SearchFieldDataType.String),
        SearchableField(name="file", type=SearchFieldDataType.String),
        SearchableField(name="chunk_content", type=SearchFieldDataType.String),
        SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072, 
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="chapter_name"),
            content_fields=[SemanticField(field_name="chunk_content")]
        )
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])
    search_index = SearchIndex(name=search_index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = get_search_index_client(search_index_name).create_or_update_index(search_index)
    print(f' {result.name} created')

def index_exists(client, index_name):
    indexes = client.list_index_names()
    return index_name in indexes

def delete_index_if_exists(search_index_name):
    client = get_search_index_client(search_index_name)
    try:
        if index_exists(client, search_index_name):
            print(f"Index '{search_index_name}' exists.")
            client.delete_index(search_index_name)
            print(f"Index '{search_index_name}' deleted.")
        else:
            print(f"Index '{search_index_name}' does not exist.")
    except Exception as e:
        print(f"Error deleting index: {e}")


def get_search_client(search_index_name):
    return  SearchClient(
        endpoint=endpoint,
        index_name=search_index_name,
        credential=credential
    )


def upload_chunk_document(filepath, search_index_name):
    search_client = get_search_client(search_index_name)
    filename = os.path.basename(filepath)

    if filename.endswith('.json'):
        with open(filepath, 'r') as file:
            document = json.load(file)
            print(f"Uploading {filename} to Azure Search Index...")

            result = search_client.upload_documents(documents=document)
            print(f"Upload of {filename} succeeded: { result[0].succeeded }")

