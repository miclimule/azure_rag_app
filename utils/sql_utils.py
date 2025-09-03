import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# Connexion
def get_connection():
    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DB")
    username = os.getenv("AZURE_SQL_USER")
    password = os.getenv("AZURE_SQL_PASSWORD")
    conn_str = (
        'DRIVER={ODBC Driver 18 for SQL Server};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        'Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    )
    return pyodbc.connect(conn_str)

# CRUD sur la table documents
def create_document(doc):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (starting_line, book, book_name, chapter, chapter_name)
        VALUES (?, ?, ?, ?, ?)
    """, doc['starting_line'], doc['book'], doc['book_name'], doc['chapter'], doc['chapter_name'])
    conn.commit()
    cursor.close()
    conn.close()

def read_documents():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    cursor.close()
    conn.close()

def update_document(document_id, new_title):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE documents SET chapter_name = ? WHERE id = ?
    """, new_title, document_id)
    conn.commit()
    cursor.close()
    conn.close()

def delete_document(document_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", document_id)
    conn.commit()
    cursor.close()
    conn.close()

# Exemple d'utilisation
if __name__ == "__main__":
    document = {
        "starting_line": 837,
        "book": "I",
        "book_name": "WAGES AND CAPITAL",
        "chapter": "I",
        "chapter_name": "The current doctrine of wages—its insufficiency"
    }

    print(pyodbc.drivers())
    create_document(document)
    read_documents()
    update_document(1, "Updated Chapter Title")
    # delete_document(1)
