"""
Keyword-based sentiment classification using Bayes' Theorem, computed
by hand from word counts - no scikit-learn, no NLTK, no vectorizers.

P(Positive | keyword) = P(keyword | Positive) * P(Positive) / P(keyword)
"""

import csv
import re

WORD_RE = re.compile(r"[a-z']+")


def load_reviews(path):
    """Returns a list of (review_text, sentiment) tuples."""
    reviews = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append((row["review"].lower(), row["sentiment"]))
    return reviews


def tokenize(text):
    return set(WORD_RE.findall(text))


def build_word_counts(reviews):
    """
    Counts, for every review, whether each word appears at least once
    (presence/absence per review, not raw frequency - this keeps a single
    review that repeats a word 20 times from dominating the estimate).
    """
    pos_doc_count = 0
    neg_doc_count = 0
    word_pos_count = {}
    word_neg_count = {}

    for text, label in reviews:
        words = tokenize(text)
        if label == "positive":
            pos_doc_count += 1
            for w in words:
                word_pos_count[w] = word_pos_count.get(w, 0) + 1
        else:
            neg_doc_count += 1
            for w in words:
                word_neg_count[w] = word_neg_count.get(w, 0) + 1

    return {
        "pos_doc_count": pos_doc_count,
        "neg_doc_count": neg_doc_count,
        "word_pos_count": word_pos_count,
        "word_neg_count": word_neg_count,
    }


def bayes_breakdown(keyword, counts):
    """
    Returns the four Bayes' Theorem quantities for a single keyword,
    all derived from raw document counts:

      prior      = P(Positive)
      likelihood = P(keyword | Positive)
      marginal   = P(keyword)                 [law of total probability]
      posterior  = P(Positive | keyword)       [Bayes' theorem]
    """
    keyword = keyword.lower()
    total_docs = counts["pos_doc_count"] + counts["neg_doc_count"]

    prior_pos = counts["pos_doc_count"] / total_docs

    docs_with_keyword_pos = counts["word_pos_count"].get(keyword, 0)
    docs_with_keyword_neg = counts["word_neg_count"].get(keyword, 0)

    likelihood_pos = docs_with_keyword_pos / counts["pos_doc_count"]

    marginal = (docs_with_keyword_pos + docs_with_keyword_neg) / total_docs

    posterior_pos = (likelihood_pos * prior_pos) / marginal

    return {
        "keyword": keyword,
        "prior_P_positive": prior_pos,
        "likelihood_P_keyword_given_positive": likelihood_pos,
        "marginal_P_keyword": marginal,
        "posterior_P_positive_given_keyword": posterior_pos,
        "docs_with_keyword_pos": docs_with_keyword_pos,
        "docs_with_keyword_neg": docs_with_keyword_neg,
    }
