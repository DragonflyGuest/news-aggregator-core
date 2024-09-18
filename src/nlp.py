import sqlite3
from typing import Dict, List, Tuple, Any, Set
import requests
from goose3 import Goose
import spacy
from spacy.tokens import Doc
import textacy
import textacy.extract
from textacy import make_spacy_doc
import nltk
from textblob import TextBlob
from operator import itemgetter
import pprint  # for pprint.pprint(results)
import json
from collections import defaultdict, Counter
import numpy as np
from requests.exceptions import JSONDecodeError
from keybert import KeyBERT


# Part 1: Fetching and Structuring Article Content
def fetch_article(url: str, timeout: int = 10) -> str:
    """
    Fetches the content of a webpage using Goose3 with a strong timeout.
    Returns the cleaned article text.

    :param url: URL of the webpage to fetch.
    :param timeout: Timeout for the request in seconds.
    :return: Goose article object
    """
    try:
        # Attempt to fetch the content using requests with timeout
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Use Goose to extract article
        with Goose() as goose:
            article = goose.extract(raw_html=response.content)
            return article
    except Exception as e:
        print(f"Failed to fetch article content/parse article: {e}")
        return ""


# Part 2: Article Text Processing with NLP
def process_article_text_with_summary_and_sentiment(text: str) -> Dict[str, List[str]]:
    """
    Processes the article text using NLP to extract named entities,
    keywords, perform sentiment analysis, and generate an extractive summary.

    :param text: The text of the article to process.
    :return: A dictionary with named entities, keywords, sentiment analysis, and summary.
    """
    # Load Spacy model
    nlp = spacy.load("en_core_web_lg")
    doc = make_spacy_doc(text, lang=nlp)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract keywords using Textacy
    keywords = [keyword for keyword, weight in textacy.extract.keyterms.textrank(doc, normalize="lemma")]

    # Perform sentiment analysis on the entire article using TextBlob
    blob = TextBlob(text)
    overall_sentiment = blob.sentiment

    # Generate extractive summary based on keywords
    summary_sentences = []
    for keyword in keywords:
        for sentence in doc.sents:
            if keyword in sentence.text:
                summary_sentences.append(sentence.text)
                break  # Move to the next keyword after finding the first matching sentence

    summary = list(set(summary_sentences))  # Ensure unique sentences

    best_match_results = find_most_entity(doc, nlp)

    return {
        "entities": entities,
        "keywords": keywords,
        "overall_sentiment": overall_sentiment,
        "summary": summary,
        "real_entity_ID": best_match_results
    }


def get_data(text):
    nlp = spacy.load("en_core_web_lg")
    doc = make_spacy_doc(text, lang=nlp)
    best_match_results = find_most_entity(doc, nlp)
    keywords = get_keywords(text)
    return {
        'real_entity_ID': best_match_results,
        'keywords': keywords
    }


def get_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text,
                                         stop_words='english',
                                         use_mmr=True,
                                         diversity=0.1,
                                         top_n=7)
    keyword_list = [keyword[0] for keyword in keywords]
    return keyword_list


# Part 3: Entity Clustering and Frequent Entity Identification

def find_most_entity(doc, nlp):
    """
    Finds the most frequent entity in the document and returns its best matching label and ID from Wikidata.

    :param doc: The spaCy document to process.
    :param nlp: The spaCy NLP model to use for processing.
    :return: A tuple containing the best matching entity label and ID, or (None, None) if no match is found.
    """
    relations = find_entity_relations(doc)

    ents_list = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    ent_clusters = cluster_entities(ents_list, nlp)

    # Count frequency of each entity in the cluster
    entity_freq = defaultdict(int)
    for cluster_key, cluster_members in ent_clusters.items():
        for member in cluster_members:
            entity_freq[cluster_key] += 1

    # print("1")

    # Find the most frequent entities
    most_frequent_entities = sorted(entity_freq, key=entity_freq.get, reverse=True)[:3]  # Here's how many to get
    # print('2')
    # print(most_frequent_entities)

    results = []
    for entity in most_frequent_entities:
        # Merge relations for the current entity
        merged_relations = []
        for person, related_entities in relations.items():
            if person in ent_clusters.get(entity, [entity]):
                merged_relations.extend(related_entities)

        # Extract unique entity names from merged_relations
        unique_entities = set(entity['related_entity'] for entity in merged_relations)

        best_match_label, best_match_id = find_best_match(entity, list(unique_entities), nlp)
        if best_match_label and best_match_id:
            results.append((best_match_label, best_match_id))
        else:
            results.append((None, None))

    return results


def cluster_entities(ents_list, nlp, threshold=0.7):
    """
    Clusters entities based on their similarity.

    :param ents_list: A list of spaCy spans representing the entities to cluster.
    :param nlp: The spaCy NLP model to use for processing.
    :param threshold: The similarity threshold for clustering entities.
    :return: A dictionary mapping representative entities to their cluster members.
    """
    clusters = defaultdict(list)
    processed = set()  # To track processed entities and avoid redundancy

    for ent1 in ents_list:
        found_cluster = False
        for key, members in clusters.items():
            # Use an example member from current cluster to measure similarity
            example_member = nlp(members[0])
            if ent1.similarity(example_member) >= threshold:
                members.append(ent1.text)
                found_cluster = True
                break

        if not found_cluster:
            # Start a new cluster with this entity
            clusters[ent1.text].append(ent1.text)

        processed.add(ent1.text)

    # Post-processing to ensure all cluster members are listed under one representative
    final_clusters = defaultdict(list)
    for members in clusters.values():
        representative = members[0]  # You can select representative by other criteria as well
        for member in members:
            final_clusters[representative].append(member)

    return final_clusters


def find_entity_relations(doc):
    """
    Finds relations between entities in the document.

    :param doc: The spaCy document to process.
    :return: A dictionary mapping person entities to their related entities and relations.
    """
    entities = list(doc.ents)
    relations = defaultdict(list)
    # Filter to keep only PERSON entities
    person_entities = [ent for ent in entities if ent.label_ == "PERSON"]

    for person in person_entities:
        for other in entities:
            if person == other:
                continue  # Skip self-relations
            person_root = person.root
            other_root = other.root
            path1 = get_dependency_path(person_root)
            path2 = get_dependency_path(other_root)
            common_ancestor = None
            for token in path1:
                if token in path2:
                    common_ancestor = token
                    break
            if common_ancestor:
                # Only add relations where the key is a PERSON entity
                relations[person.text].append({
                    'related_entity': other.text,
                    'relation': common_ancestor.dep_
                })
    return relations


def get_dependency_path(token):
    """
    Retrieves the dependency path from the token to the root of the sentence.

    :param token: The token to start the path from.
    :return: A list of tokens representing the dependency path.
    """
    path = []
    while token.head != token:
        path.append(token)
        token = token.head
    path.append(token)  # Add the root token
    return path


# Part 4: Entity Matching and Information Retrieval from Wikidata

def find_best_match(name, context, nlp):
    """
    Finds the best matching entity for a given name based on context using Wikidata.

    :param name: The name of the entity to find a match for.
    :param context: The context in which the entity appears.
    :param nlp: The spaCy NLP model to use for processing.
    :return: A tuple containing the best matching entity label and ID, or (None, None) if no match is found.
    """
    candidates = query_wikidata(name)
    if candidates:
        best_match_id = match_entity(candidates, context, nlp)
        if best_match_id:
            best_match_label = get_label(best_match_id)
            return best_match_label, best_match_id
        else:
            return None, None
    else:
        return None, None


def query_wikidata(entity_name):
    """
    Queries Wikidata for entities with the given name.

    :param entity_name: The name of the entity to query for.
    :return: A list of Wikidata entity IDs that match the given name.
    """
    # Use simplified SPARQL query
    sparql_query = f"""
    SELECT ?person ?personLabel WHERE {{
      ?person wdt:P31 wd:Q5; # The entity is a human
              rdfs:label "{entity_name}"@en.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 20 # Limit to at most 20 results
    """
    url = 'https://query.wikidata.org/sparql'
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = {
        'query': sparql_query,
        'format': 'json'
    }
    response = requests.get(url, headers=headers, params=data)
    response.raise_for_status()  # Ensure the response status is 200
    data = response.json()

    results = data['results']['bindings']
    if results:
        candidates = [result['person']['value'].split('/')[-1] for result in results]
        return candidates
    else:
        return []


def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    :param vec1: The first vector.
    :param vec2: The second vector.
    :return: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def match_entity(candidates, context, nlp):
    """
    Matches the most relevant entity based on context information, using cosine similarity to compare document vectors.

    :param candidates: A list of candidate entity IDs.
    :param context: The context in which the entity appears.
    :param nlp: The spaCy NLP model to use for processing.
    :return: The ID of the best matching entity, or None if no match is found.
    """
    best_similarity = 0
    best_match = None

    # Convert context to vector
    context_doc = nlp(' '.join(context))
    context_vector = context_doc.vector

    for qid in candidates:
        entity_info = get_entity_info(qid)
        if entity_info:
            # Merge entity info into a single text, then convert to vector
            entity_text = ' '.join(entity_info)
            entity_doc = nlp(entity_text)
            entity_vector = entity_doc.vector

            # Calculate cosine similarity
            similarity = cosine_similarity(context_vector, entity_vector)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = qid

    return best_match


def get_entity_info(qid):
    """
    Retrieves key information about an entity from Wikidata.

    :param qid: The Wikidata ID of the entity.
    :return: A list of strings containing the entity's description and property values.
    """
    # Get key information about the entity
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract entity description information
        description = data['entities'][qid]['descriptions']['en']['value']

        # Get all property values
        claims = data['entities'][qid]['claims']

        # Define a list of properties to exclude
        excluded_properties = ['P18', 'P31', 'P21', 'P27', 'P569', 'P19', 'P20', 'P1559', 'P1477', 'P735', 'P734',
                               'P1889', 'P4464']

        # Extract all property values, excluding specified properties
        property_values = []
        for prop, claim_list in claims.items():
            if prop not in excluded_properties:
                for claim in claim_list:
                    if claim['mainsnak']['snaktype'] == 'value':
                        if claim['mainsnak']['datavalue']['type'] == 'wikibase-entityid':
                            prop_qid = claim['mainsnak']['datavalue']['value']['id']
                            prop_label = get_label(prop_qid)
                            if prop_label:
                                property_values.append(prop_label)
                        elif claim['mainsnak']['datavalue']['type'] == 'time':
                            prop_value = claim['mainsnak']['datavalue']['value']['time']
                            property_values.append(prop_value)
                        elif claim['mainsnak']['datavalue']['type'] == 'string':
                            prop_value = claim['mainsnak']['datavalue']['value']
                            property_values.append(prop_value)
                        elif claim['mainsnak']['datavalue']['type'] == 'quantity':
                            prop_value = claim['mainsnak']['datavalue']['value']['amount']
                            property_values.append(prop_value)
                        elif claim['mainsnak']['datavalue']['type'] == 'monolingualtext':
                            prop_value = claim['mainsnak']['datavalue']['value']['text']
                            property_values.append(prop_value)
                        elif claim['mainsnak']['datavalue']['type'] == 'globecoordinate':
                            prop_value = claim['mainsnak']['datavalue']['value']['latitude'], \
                                claim['mainsnak']['datavalue']['value']['longitude']
                            property_values.append(prop_value)

        return [description] + property_values
    except (JSONDecodeError, KeyError, requests.exceptions.RequestException):
        return []


def get_label(qid):
    """
    Retrieves the label of an entity from Wikidata.

    :param qid: The Wikidata ID of the entity.
    :return: The label of the entity, or None if not found.
    """
    # Get the label of an entity
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        label = data['entities'][qid]['labels']['en']['value']
        return label
    except (JSONDecodeError, KeyError, requests.exceptions.RequestException):
        return None


# Load a pre-computed list of top entities.
def load_entities_from_json(filename: str) -> List[str]:
    """
    Loads the entities from a JSON file.

    :param filename: The name of the JSON file to load entities from.
    :return: A list of entities loaded from the JSON file.
    """
    with open(filename, 'r') as f:
        entities = json.load(f)
    return list(entities.keys())


# test between article entities and pre-computed entities
def test_entities_in_list(article_entities: List[Tuple[str, str]], loaded_entities: List[str]) -> Set[str]:
    """
    Tests if the entities extracted from the article are in the list of loaded entities.

    :param article_entities: A list of entities extracted from the article, where each entity is a tuple of (entity_text, entity_label).
    :param loaded_entities: A list of entities loaded from the JSON file.
    :return: A set of unique entities from the article that are found in the loaded entities list.
    """
    matching_entities = set()
    for entity_text, entity_label in article_entities:
        combined_entity = f"{entity_text} {entity_label}"
        if combined_entity in loaded_entities:
            matching_entities.add(combined_entity)
    return matching_entities


# Function to use on the REPL for getting URLs and testing entities.
def understand_url(url: str) -> Dict[str, Any]:
    article = fetch_article(url)

    if not article:
        return {
            "title": "",
            "meta_description": "",
            "cleaned_text": "",
            "entities": [],
            "keywords": [],
            "summary": [],
            "overall_sentiment": None,
            "matching_entities": [],
            "real_entity_ID": []
        }

    try:
        processed_data = process_article_text_with_summary_and_sentiment(article.cleaned_text)
        loaded_entities = load_entities_from_json('entities.json')
        matching_entities = test_entities_in_list(processed_data["entities"], loaded_entities)

        return {
            "title": article.title,
            "meta_description": article.meta_description.replace('\xa0', ' '),
            "cleaned_text": article.cleaned_text,
            "entities": processed_data["entities"],
            "keywords": processed_data["keywords"],
            "summary": processed_data["summary"],
            "overall_sentiment": processed_data["overall_sentiment"],
            "matching_entities": matching_entities,
            "real_entity_ID": processed_data["real_entity_ID"]
        }
    except Exception as e:
        print(f"Error processing article: {e}")
        return {
            "title": article.title,
            "meta_description": article.meta_description.replace('\xa0', ' '),
            "cleaned_text": article.cleaned_text,
            "entities": [],
            "keywords": [],
            "summary": [],
            "overall_sentiment": None,
            "matching_entities": [],
            "real_entity_ID": []
        }


def get_name(url):
    article = fetch_article(url)
    #data = get_data(article.cleaned_text)
    data = 'Trump'
    return data

def get_keyword(url):
    #article = fetch_article(url)
    keyword = "Technology"
    return keyword



