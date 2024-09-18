import json
import sqlite3
import spacy
import numpy as np
from sklearn.neighbors import BallTree
from typing import List, Tuple, Any
import os

def setup_database() -> Tuple[sqlite3.Connection, List[Any]]:
    """
    Set up the SQLite database and load metadata from a JSON file.
    """
    # Load the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "raw_data", "metadata_clean.json")
    with open(file_path, "r") as f:
        metadata = json.load(f)

    # Initialize SQLite database
    db_path = os.path.join(current_dir, "..", "raw_data", "metadata.db")
    conn = sqlite3.connect(db_path)
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

def text_search(conn: sqlite3.Connection, query: str, limit: int = 1) -> List[str]:
    """
    Perform a text-based search on the metadata, returning a list of IDs.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM metadata_fts WHERE text MATCH ? ORDER BY rank LIMIT ?", (query, limit))
    results = cursor.fetchall()
    return [r[0] for r in results]

def create_vector_index(nlp: spacy.language.Language, metadata: List[Any]) -> Tuple[BallTree, np.ndarray]:
    """
    Create a vector index for the metadata using spaCy embeddings.
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

def vector_search(nlp: spacy.language.Language, tree: BallTree, vectors: np.ndarray, metadata: List[Any], query: str, k: int = 1) -> List[str]:
    """
    Perform a vector-based search on the metadata, returning a list of IDs.
    """
    query_vector = nlp(query).vector.reshape(1, -1)
    distances, indices = tree.query(query_vector, k=k)
    
    results = []
    for idx in indices[0]:
        item = metadata[idx]
        results.append(item[0])
    
    return results

def combined_query(query: str, conn, nlp, tree, vectors, metadata) -> str:
    """
    Perform a combined text-based and vector-based search on the metadata.
    Return the ID of the most relevant result.
    """
    result_id = None

    # Try text search first
    try:
        text_results = text_search(conn, query)
        if text_results:
            result_id = text_results[0]
    except Exception as e:
        print("Error in text search:", e)

    # If no result from text search, try vector search
    if not result_id:
        try:
            vector_results = vector_search(nlp, tree, vectors, metadata, query)
            if vector_results:
                result_id = vector_results[0]
        except Exception as e:
            print("Error in vector search:", e)
    
    return result_id

def get_background(keyword: str) -> str:
    """
    Search for the most relevant image based on the keyword and return its path.
    """
    # Load the spaCy model
    nlp = spacy.load("en_core_web_lg")

    # Set up the database and get the metadata
    conn, metadata = setup_database()

    # Create the vector index
    tree, vectors = create_vector_index(nlp, metadata)

    # Perform the query to get the most relevant ID
    result_id = combined_query(keyword, conn, nlp, tree, vectors, metadata)

    # If a result was found, return the path to the corresponding image
    if result_id:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the relative path to the static folder
        static_path = os.path.join(current_dir, "..", "static")
        # Construct the image path
        image_path = os.path.join(static_path, f'{result_id}.avif')
        if os.path.exists(image_path):
            return image_path
        else:
            return f"Image {result_id}.avif not found in the static folder."
    else:
        return "No relevant image found."

