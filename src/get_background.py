import json
import sqlite3
import spacy
import numpy as np
from sklearn.neighbors import BallTree
from collections import Counter
from typing import List, Dict, Tuple, Any
import random
import os
import shutil

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

def text_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Perform a text-based search on the metadata.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM metadata_fts WHERE text MATCH ? ORDER BY rank LIMIT ?", (query, limit))
    results = cursor.fetchall()
    return [{'id': r[0], 'text': r[1]} for r in results]

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

def vector_search(nlp: spacy.language.Language, tree: BallTree, vectors: np.ndarray, metadata: List[Any], query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a vector-based search on the metadata.
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

def combined_query(query: str, conn, nlp, tree, vectors, metadata) -> List[str]:
    """
    Perform a combined text-based and vector-based search on the metadata.
    """
    results = []

    try:
        text_results = text_search(conn, query)
        for result in text_results:
            results.append(result['id'])
    except Exception as e:
        print("Error in text search:", e)

    try:
        vector_results = vector_search(nlp, tree, vectors, metadata, query)
        for result in vector_results:
            results.append(result['id'])
    except Exception as e:
        print("Error in vector search:", e)

    # Remove duplicate result IDs
    unique_results = list(set(results))
    
    return unique_results

def save_image(image_id: str, destination_folder: str):
    """
    Copy images from the 'static' folder to the specified folder.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the relative path to the static folder
    static_path = os.path.join(current_dir, "..", "static")
    
    # Construct the target folder path, which is parallel to the src folder
    destination_folder = os.path.join(current_dir, "..", "saved_images")
    os.makedirs(destination_folder, exist_ok=True)  # Create the target folder if it doesn't exist

    # Construct the source image path
    source_path = os.path.join(static_path, f'{image_id}.avif')
    
    if os.path.exists(source_path):
        # Construct the target path
        destination_path = os.path.join(destination_folder, f'{image_id}.avif')
        shutil.copyfile(source_path, destination_path)
        print(f"Image {image_id}.avif has been saved to {destination_path}")
    else:
        print(f"Image {image_id}.avif not found in the static folder.")


# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Set up the database and get the metadata
conn, metadata = setup_database()

# Create the vector index
tree, vectors = create_vector_index(nlp, metadata)

# Load the article data
current_dir = os.path.dirname(os.path.abspath(__file__))
articles_file = os.path.join(current_dir, "..", "raw_json", "enriched_articles_20240828_144730.json")

with open(articles_file, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Local logic (perform search and save results directly)
query = input("Please enter the keywords for search: ")

if query:
    print(f"Keywords for search: {query}")
    
    # Perform the query and return results
    result_ids = combined_query(query, conn, nlp, tree, vectors, metadata)
    if result_ids:
        print(f"Found {len(result_ids)} results, saving the top 5 related images:")
        for image_id in result_ids[:5]:
            save_image(image_id, 'saved_images')
    else:
        print("No related images found.")

print("Execution complete.")
