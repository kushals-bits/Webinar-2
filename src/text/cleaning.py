"""
Text Cleaning & Normalization Module
Removes HTML tags, URLs, noise, contractions, punctuation, and stopwords.
"""

import re
import html


# Standard English Stopwords list
ENGLISH_STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", 
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "will", "with", "you", "your", "yours", "yourself", "yourselves"
])

# Common contraction mapping
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}


def expand_contractions(text: str) -> str:
    """Expands common English contractions."""
    for cont, expanded in CONTRACTIONS.items():
        text = re.sub(re.escape(cont), expanded, text, flags=re.IGNORECASE)
    return text


def strip_html_tags(text: str) -> str:
    """Removes HTML tags and decodes HTML entities."""
    clean = re.sub(r"<.*?>", " ", text)
    return html.unescape(clean)


def strip_urls_and_emails(text: str) -> str:
    """Removes URLs (http, https, www) and email addresses."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    return text


def remove_special_characters(text: str, keep_alphanumeric_only: bool = True) -> str:
    """Removes emojis, symbols, and non-alphanumeric noise."""
    if keep_alphanumeric_only:
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    expand_cont: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True
) -> str:
    """
    Comprehensive text cleaning pipeline:
    1. HTML stripping & unescaping
    2. URL and Email removal
    3. Contraction expansion
    4. Lowercasing
    5. Punctuation & Special characters stripping
    6. Stopwords removal
    7. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    if remove_html:
        text = strip_html_tags(text)

    if remove_urls:
        text = strip_urls_and_emails(text)

    if expand_cont:
        text = expand_contractions(text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = remove_special_characters(text)

    tokens = text.split()

    if remove_stopwords:
        tokens = [w for w in tokens if w not in ENGLISH_STOPWORDS and len(w) > 1]

    return " ".join(tokens)


def batch_clean_texts(texts: list, **kwargs) -> list:
    """Cleans a list or pandas Series of texts."""
    return [clean_text(t, **kwargs) for t in texts]
