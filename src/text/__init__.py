"""
Text Preprocessing Package for Webinar 2: Data Preprocessing.
Covers: Cleaning, TF-IDF Vectorization, Numerical Feature Engineering, and Imbalance Handling.
"""

from .cleaning import clean_text, batch_clean_texts, strip_html_tags, strip_urls_and_emails, expand_contractions
from .vectorization import TFIDFProcessor
from .feature_engineering import extract_numerical_text_features
from .imbalance import run_text_pipeline

__all__ = [
    "clean_text",
    "batch_clean_texts",
    "strip_html_tags",
    "strip_urls_and_emails",
    "expand_contractions",
    "TFIDFProcessor",
    "extract_numerical_text_features",
    "run_text_pipeline"
]
