from typing import List, Dict, Any, Optional
import feedparser
import spacy
import textacy
from textacy.extract import keyterms
import nltk
from textblob import TextBlob
import json
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import env


def parse_feed(feed_url: str) -> feedparser.FeedParserDict:
    """
    Parse an RSS feed from a given URL.

    Args:
        feed_url (str): The URL of the RSS feed.

    Returns:
        feedparser.FeedParserDict: The parsed feed.
    """
    return feedparser.parse(feed_url)


def extract_article_text(entry: feedparser.FeedParserDict) -> Optional[str]:
    """
    Extract the full article text from a feed entry.

    Args:
        entry (feedparser.FeedParserDict): A single entry from the parsed feed.

    Returns:
        Optional[str]: The extracted article text, or None if not found.
    """
    # Try different common locations for the full content
    content_locations = [
        lambda e: e.get('content', [{}])[0].get('value'),
        lambda e: e.get('summary'),
        lambda e: e.get('description'),
    ]

    for get_content in content_locations:
        content = get_content(entry)
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)

    return None


def process_entry(entry: feedparser.FeedParserDict, nlp: spacy.language.Language) -> Dict[str, Any]:
    """
    Process a single feed entry with various NLP tasks.

    Args:
        entry (feedparser.FeedParserDict): A single entry from the parsed feed.
        nlp (spacy.language.Language): The loaded spaCy model.

    Returns:
        Dict[str, Any]: The enriched entry with NLP analysis results.
    """
    article_text = extract_article_text(entry)
    if not article_text:
        return {}  # Skip this entry if we couldn't extract the text

    enriched_entry = {
        'title': entry.get('title', ''),
        'summary': entry.get('summary', ''),
        'link': entry.get('link', ''),
        'published': entry.get('published', ''),
        'article_text': article_text
    }

    full_text = f"{enriched_entry['title']}. {article_text}"
    doc = nlp(full_text)

    enriched_entry['noun_chunks'] = [chunk.text for chunk in textacy.extract.noun_chunks(doc)]
    enriched_entry['named_entities'] = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    enriched_entry['textrank_keyterms'] = [term for term, _ in keyterms.textrank(doc, topn=10)]
    enriched_entry['textrank_summary'] = generate_summary(doc)
    enriched_entry['tokens'] = [token.text for token in doc[:100]]  # Use spaCy tokenization
    enriched_entry['pos_tags'] = [(token.text, token.pos_) for token in doc[:100]]  # Use spaCy POS tagging

    blob = TextBlob(full_text)
    enriched_entry['sentiment'] = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

    return enriched_entry


def generate_summary(doc: spacy.tokens.Doc) -> str:
    """
    Generate a summary of the document using TextRank.

    Args:
        doc (spacy.tokens.Doc): The spaCy document to summarize.

    Returns:
        str: The generated summary.
    """
    sentences = list(doc.sents)
    ranked_sentences = textacy.extract.keyterms.textrank(doc)
    top_sentences = sorted(
        [(sent.text, score) for sent in sentences for term, score in ranked_sentences if term in sent.text],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    return ' '.join([sent[0] for sent in top_sentences])


def process_feeds(feed_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple RSS feeds and perform NLP analysis on their entries.

    Args:
        feed_urls (List[str]): A list of RSS feed URLs to process.

    Returns:
        List[Dict[str, Any]]: A list of enriched entries from all feeds.
    """
    nlp = spacy.load("en_core_web_lg")
    enriched_entries = []

    # Progress bar for feed loading
    for feed_url in tqdm(feed_urls, desc="Loading feeds"):
        feed = parse_feed(feed_url)

        # Progress bar for processing entries within each feed
        for entry in tqdm(feed.entries, desc=f"Processing entries from {feed_url}", leave=False):
            enriched_entry = process_entry(entry, nlp)
            if enriched_entry:
                enriched_entries.append(enriched_entry)

    return enriched_entries


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def save_enriched_entries(enriched_entries: List[Dict[str, Any]]) -> None:
    """
    Save the enriched entries as a JSON file.

    Args:
        enriched_entries (List[Dict[str, Any]]): The list of enriched entries to save.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enriched_articles_{timestamp}.json"

    with open(os.path.join(env.raw_json_output, filename), 'w', encoding='utf-8') as f:
        json.dump(enriched_entries, f, ensure_ascii=False, indent=4, cls=JSONEncoder)

    print(f"Enriched articles saved to {filename}")


def main() -> None:
    """
    Main function to run the RSS feed processing pipeline.
    """
    feed_urls = [
        "https://theconversation.com/articles.atom?language=en",
        "https://www.sbs.com.au/news/topic/world/feed",
        "https://www.sbs.com.au/news/topic/australia/feed",
        # Add more feed URLs here
    ]
    enriched_entries = process_feeds(feed_urls)
    save_enriched_entries(enriched_entries)



if __name__ == "__main__":
    main()
