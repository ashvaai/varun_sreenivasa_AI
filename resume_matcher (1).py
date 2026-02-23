"""
Resume Matcher
This script provides a simple proof-of-concept for evaluating the similarity
between a job description (JD) and a candidate résumé. It performs basic
tokenization, computes a Jaccard similarity score, generates a short summary
of the résumé, and outputs a recommended action (Interview or Reject) based
on a configurable threshold.

Usage:
    python resume_matcher.py --jd path/to/job_description.txt --resume path/to/resume.txt

This will print a JSON object containing the score, summary, and recommended action.

The script can be extended or integrated into a web backend or automation tool
like n8n. It demonstrates how unstructured text can be processed programmatically
without relying on large language models.
"""

import argparse
import json
import re
from typing import List, Tuple

try:
    # sklearn is available in the runtime and provides more robust text similarity via TF‑IDF.
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SKLEARN = True
except ImportError:
    # Fall back to manual cosine similarity if sklearn is unavailable.  This ensures the script
    # still functions in minimal environments but will yield less nuanced scores.
    _HAS_SKLEARN = False


def tokenize(text: str) -> List[str]:
    """
    Convert text into a list of lowercase alphanumeric tokens and remove common stop words.

    Stop words are very common words (e.g. "the", "and") that provide little value in measuring
    similarity.  Removing them yields more meaningful scores.
    """
    # Basic lowercase alphanumeric tokenization
    tokens = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()
    # Define a small stop word list.  This could be expanded or replaced with a more comprehensive list.
    stopwords = {
        'the', 'and', 'a', 'to', 'of', 'in', 'for', 'on', 'with', 'is', 'that', 'by', 'or',
        'as', 'an', 'be', 'are', 'from', 'at', 'this', 'use', 'using', 'we', 'you', 'your',
        'it', 'our', 'his', 'her', 'they', 'them', 'their', 'i', 'me', 'my', 'mine'
    }
    return [token for token in tokens if token and token not in stopwords]


def manual_cosine_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Compute cosine similarity between two lists of tokens using term frequency.

    When sklearn is unavailable, this fallback computes a simple cosine similarity between term
    frequency vectors.  A similarity of 1.0 means the texts are identical in terms of token
    distribution, whereas 0.0 indicates no overlap.
    """
    from collections import Counter
    import math

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    # Compute dot product for common tokens
    common_tokens = set(counter_a.keys()) & set(counter_b.keys())
    dot_product = sum(counter_a[t] * counter_b[t] for t in common_tokens)

    # Compute magnitudes
    magnitude_a = math.sqrt(sum(val * val for val in counter_a.values()))
    magnitude_b = math.sqrt(sum(val * val for val in counter_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def generate_summary(tokens: List[str], length: int = 40) -> str:
    """Generate a simple summary by taking the first N words from the résumé."""
    summary_tokens = tokens[:length]
    summary = ' '.join(summary_tokens)
    if len(tokens) > length:
        summary += '...'
    return summary


def evaluate(jd_text: str, resume_text: str, threshold: int = 50) -> Tuple[int, str, str]:
    """
    Evaluate the JD and résumé and return the score, summary, and recommendation.

    This function will prefer TF‑IDF cosine similarity via sklearn if available; otherwise it
    falls back to manual cosine similarity over token frequency.  The score is scaled to an
    integer percentage (0–100).

    :param jd_text: The job description text.
    :param resume_text: The résumé text.
    :param threshold: Minimum score required to recommend an interview.
    :return: (score, summary, recommendation)
    """
    jd_tokens = tokenize(jd_text)
    resume_tokens = tokenize(resume_text)

    # Compute similarity.  If sklearn is available, use TF‑IDF for greater nuance.
    if _HAS_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            # Provide raw text rather than tokens to the vectorizer.  It will handle tokenisation and
            # stop word removal internally.  We still generate a summary based on our tokenised
            # résumé to satisfy the project requirement.
            vectors = vectorizer.fit_transform([jd_text, resume_text])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        except Exception:
            # In case of any runtime error (e.g. from missing language resources), fall back
            similarity = manual_cosine_similarity(jd_tokens, resume_tokens)
    else:
        similarity = manual_cosine_similarity(jd_tokens, resume_tokens)

    score = int(round(similarity * 100))
    summary = generate_summary(resume_tokens)
    recommendation = 'Interview' if score >= threshold else 'Reject'
    return score, summary, recommendation


def main():
    parser = argparse.ArgumentParser(description='Evaluate a résumé against a job description.')
    parser.add_argument('--jd', required=True, help='Path to job description text file')
    parser.add_argument('--resume', required=True, help='Path to résumé text file')
    parser.add_argument('--threshold', type=int, default=50, help='Score threshold for recommending an interview')
    args = parser.parse_args()

    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    with open(args.resume, 'r', encoding='utf-8') as f:
        resume_text = f.read()

    score, summary, recommendation = evaluate(jd_text, resume_text, args.threshold)
    result = {
        'score': score,
        'summary': summary,
        'recommendation': recommendation,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()