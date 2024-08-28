import get_background as gb
import generate_image as gi
import download_img as di
import RSS2NLP as RS
import utils

import spacy
import json

def main():
    input_url = ["https://theconversation.com/articles.atom?language=en"]

    # Load the large English word vector model
    nlp = spacy.load("en_core_web_lg")

    # Setup the database and get metadata
    conn, metadata = gb.setup_database()

    # Create vector index
    tree, vectors = gb.create_vector_index(nlp, metadata)

    # EXAMPLE SEARCH OF EACH TYPE
    query = "technology"  # You can change this to any search term

    # text_results = text_search(conn, query)

    # get enriched article
    enriched_entries = RS.process_feeds(input_url)
    articles = json.dumps(enriched_entries, ensure_ascii=False, indent=4)
    articles_json = json.loads(articles) # you should change 0 to the exact number if you use multi input_url

    entities = utils.extract_by_label(articles_json[0]['named_entities'], "PERSON")
    # for entity in entities:
    #     di.download_entity_image(entity)
    gi.image_process()

main()
