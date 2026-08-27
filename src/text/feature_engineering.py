"""
Text Numerical Feature Engineering Module
Extracts statistical and linguistic numerical features from text data.
"""

import re
import numpy as np
import pandas as pd


POSITIVE_WORDS = set([
    "love", "loved", "great", "excellent", "amazing", "good", "best", "perfect", "fantastic", 
    "recommend", "super", "flawless", "durable", "solid", "smooth", "happy", "pleased", "top"
])

NEGATIVE_WORDS = set([
    "bad", "terrible", "horrible", "worst", "poor", "waste", "defective", "broken", "disappointed", 
    "damaged", "laggy", "bugs", "regret", "disaster", "returning", "unresponsive", "slow", "broke"
])


def extract_numerical_text_features(texts: list) -> pd.DataFrame:
    """
    Extracts numerical metadata & statistical features from raw text:
    - char_count: Total characters
    - word_count: Total word count
    - avg_word_length: Mean character length per word
    - uppercase_count & uppercase_ratio: Measure of shouting / strong emphasis
    - exclamation_count: Density of '!'
    - question_count: Density of '?'
    - digit_count: Frequency of numbers
    - lexical_diversity: Ratio of unique words to total words
    - pos_keyword_count: Frequency of positive lexicon terms
    - neg_keyword_count: Frequency of negative lexicon terms
    - lexicon_polarity_score: Difference (pos - neg)
    """
    features = []

    for raw in texts:
        t = str(raw) if not pd.isna(raw) else ""
        char_len = len(t)
        words = t.split()
        num_words = len(words)
        
        # Word lengths
        avg_word_len = np.mean([len(w) for w in words]) if num_words > 0 else 0.0
        
        # Uppercase statistics
        upper_chars = sum(1 for c in t if c.isupper())
        upper_ratio = (upper_chars / char_len) if char_len > 0 else 0.0
        
        # Punctuation counts
        exclamation_cnt = t.count("!")
        question_cnt = t.count("?")
        digit_cnt = sum(1 for c in t if c.isdigit())
        
        # Lexical diversity
        lower_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in words]
        unique_words = len(set(lower_words))
        lex_diversity = (unique_words / num_words) if num_words > 0 else 0.0
        
        # Lexicon sentiment heuristic
        pos_cnt = sum(1 for w in lower_words if w in POSITIVE_WORDS)
        neg_cnt = sum(1 for w in lower_words if w in NEGATIVE_WORDS)
        polarity_score = pos_cnt - neg_cnt

        features.append({
            "char_count": char_len,
            "word_count": num_words,
            "avg_word_length": round(float(avg_word_len), 2),
            "uppercase_count": upper_chars,
            "uppercase_ratio": round(float(upper_ratio), 4),
            "exclamation_count": exclamation_cnt,
            "question_count": question_cnt,
            "digit_count": digit_cnt,
            "lexical_diversity": round(float(lex_diversity), 4),
            "pos_keyword_count": pos_cnt,
            "neg_keyword_count": neg_cnt,
            "lexicon_polarity_score": polarity_score
        })

    return pd.DataFrame(features)
