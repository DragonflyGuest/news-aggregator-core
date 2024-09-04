import get_background as gb
import generate_image as gi
import download_img as di
import RSS2NLP as RS
import utils

import spacy
import json

def main():
    input_url = ["https://theconversation.com/articles.atom?language=en"]
    query = "technology"  # You can change this to any search term

    # Load the large English word vector model
    nlp = spacy.load("en_core_web_lg")

    # Setup the database and get metadata
    conn, metadata = gb.setup_database()

    # Create vector index
    tree, vectors = gb.create_vector_index(nlp, metadata)

    text_results = text_search(conn, query)

    # get enriched article
    extracted_data = understand_url("https://edition.cnn.com/2024/03/11/politics/takeaways-cnn-exclusive-interview-trump-employee-5-mar-a-lago/index.html")
    entities = count_and_sort_entities(extracted_data['entities'])
    if len(entities) == 0:
        return

    # for entity in entities:
    #     di.download_entity_image(entity)
    gi.image_process()

main()
