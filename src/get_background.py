import json
import sqlite3
import spacy
import numpy as np
from sklearn.neighbors import BallTree
from collections import Counter
from typing import List, Dict, Tuple, Any
from bottle import Bottle, run, request, static_file
import random
import env
import os

def setup_database() -> Tuple[sqlite3.Connection, List[Any]]:
    """
    Set up the SQLite database and load metadata from a JSON file.

    Returns:
        Tuple[sqlite3.Connection, List[Any]]: A tuple containing the database connection and the loaded metadata.
    """
    # Load the JSON file
    with open(os.path.join(env.raw_metadata, "metadata_clean.json"), "r") as f:
        metadata = json.load(f)

    # Initialize SQLite database
    conn = sqlite3.connect(os.path.join(env.raw_metadata, 'metadata.db'))
    cursor = conn.cursor()

    # Create a table with full-text search capabilities
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS metadata_fts USING fts5(
            id, text
        )
    ''')

    # Process metadata and insert into the database
    for item in metadata:
        id = item[0]
        text = ' '.join([' '.join(sublist) for sublist in item[1:5]])  # Concatenate all text fields
        cursor.execute("INSERT INTO metadata_fts (id, text) VALUES (?, ?)", (id, text))

    conn.commit()
    return conn, metadata

def text_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Perform a text-based search on the metadata.

    Args:
        conn (sqlite3.Connection): The database connection.
        query (str): The search query.
        limit (int, optional): The maximum number of results to return. Defaults to 10.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing search results.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM metadata_fts WHERE text MATCH ? ORDER BY rank LIMIT ?", (query, limit))
    results = cursor.fetchall()
    return [{'id': r[0], 'text': r[1]} for r in results]

def create_vector_index(nlp: spacy.language.Language, metadata: List[Any]) -> Tuple[BallTree, np.ndarray]:
    """
    Create a vector index for the metadata using spaCy embeddings.

    Args:
        nlp (spacy.language.Language): The spaCy language model.
        metadata (List[Any]): The metadata to index.

    Returns:
        Tuple[BallTree, np.ndarray]: A tuple containing the BallTree index and the vector array.
    """
    vectors = []
    for item in metadata:
        text = ' '.join([' '.join(sublist) for sublist in item[1:5]])
        doc = nlp(text)
        if len(doc.vector) > 0:
            vectors.append(doc.vector)

    vectors = np.array(vectors)
    tree = BallTree(vectors, leaf_size=40)
    return tree, vectors

def vector_search(nlp: spacy.language.Language, tree: BallTree, vectors: np.ndarray, metadata: List[Any], query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a vector-based search on the metadata.

    Args:
        nlp (spacy.language.Language): The spaCy language model.
        tree (BallTree): The BallTree index.
        vectors (np.ndarray): The vector array.
        metadata (List[Any]): The metadata.
        query (str): The search query.
        k (int, optional): The number of nearest neighbors to retrieve. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing search results with distances.
    """
    query_vector = nlp(query).vector.reshape(1, -1)
    distances, indices = tree.query(query_vector, k=k)

    results = []
    for i, idx in enumerate(indices[0]):
        item = metadata[idx]
        id = item[0]
        text = ' '.join([' '.join(sublist) for sublist in item[1:5]])
        results.append({
            'id': id,
            'text': text,
            'distance': distances[0][i]
        })

    return results


# Load the large English word vector model
nlp = spacy.load("en_core_web_lg")

# Setup the database and get metadata
conn, metadata = setup_database()

# Create vector index
tree, vectors = create_vector_index(nlp, metadata)

# EXAMPLE SEARCH OF EACH TYPE
query = "technology"  # You can change this to any search term
'''
print("Text-based search results:")
text_results = text_search(conn, query)
for result in text_results[:5]:  # Print top 5 results
    print(f"ID: {result['id']}")
    print(f"Text: {result['text'][:200]}...")
    print()

print("\nVector-based search results:")
vector_results = vector_search(nlp, tree, vectors, metadata, query)
for result in vector_results[:5]:  # Print top 5 results
    print(f"ID: {result['id']}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Distance: {result['distance']}")
    print()
'''
def combined_query(query: str = query) -> List[Any]:
    """
    Perform a combined text-based and vector-based search on the metadata.

    Args:
        query (str): The search query.

    Returns:
        List[Any]: A list of unique result IDs sorted by frequency.
    """
    results = []

    try:
        text_results = text_search(conn, query)
        for result in text_results:
            results.append(result['id'])
    except:
        print("error in fts")
    try:
        vector_results = vector_search(nlp, tree, vectors, metadata, query)
        for result in vector_results:
            results.append(result['id'])
    except:
        print("error in vector search")
    deny = [49,56,80,111,152,311,383,389,429,439] #various images that were failures and deleted


    return [x for x in Counter(results) if x not in deny]



###Enriched Articles Example
# enriched = open(os.path.join(env.raw_json_output, "enriched_articles_20240823_133829.json")).read()
# articles = json.loads(enriched)
