"""TF-IDF keyword extraction as second signal for topic grouping (Week 1, Step 2)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp.preprocessor import CleanedItem


def extract_keywords(
    items: list[CleanedItem],
    top_n: int = 10,
    max_features: int = 5000,
) -> list[list[str]]:
    """Return top-N TF-IDF keywords for each item in the corpus.

    Args:
        items: List of cleaned items whose `lemmas` field is used as input.
        top_n: Number of keywords to return per item.
        max_features: Vocabulary size cap for the TF-IDF vectorizer.

    Returns:
        A list of keyword lists, one per item, in the same order as `items`.
    """
    if not items:
        return []

    corpus = [" ".join(item["lemmas"]) for item in items]

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    results: list[list[str]] = []
    for row_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(row_idx).toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if row[i] > 0]
        results.append(keywords)

    return results


def attach_keywords(
    items: list[CleanedItem],
    top_n: int = 10,
) -> list[CleanedItem]:
    """Mutate items in-place to add a `keywords` field and return them."""
    keyword_lists = extract_keywords(items, top_n=top_n)
    for item, kws in zip(items, keyword_lists):
        item["keywords"] = kws  # type: ignore[typeddict-unknown-key]
    return items
