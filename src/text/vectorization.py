"""
Text Vectorization Module
Implements TF-IDF vectorization with n-grams, sublinear term frequency, and vocabulary filtering.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TFIDFProcessor:
    """
    TF-IDF Vectorizer wrapper with configurable hyperparameters:
    - n_gram ranges (unigrams, bigrams)
    - max_features limit
    - sublinear_tf: log scaling 1 + log(tf)
    - min_df and max_df document frequency pruning
    """
    def __init__(self, max_features=500, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df
        )
        self.feature_names_ = []

    def fit(self, texts, y=None):
        self.vectorizer.fit(texts)
        self.feature_names_ = self.vectorizer.get_feature_names_out().tolist()
        return self

    def transform(self, texts) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts, y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    def get_top_keywords(self, texts, top_n=10) -> pd.DataFrame:
        """Computes the overall highest TF-IDF weighted terms across the corpus."""
        tfidf_matrix = self.transform(texts)
        mean_tfidf = np.mean(tfidf_matrix, axis=0)
        top_indices = np.argsort(mean_tfidf)[::-1][:top_n]
        
        return pd.DataFrame({
            "term": [self.feature_names_[i] for i in top_indices],
            "mean_tfidf_score": [round(float(mean_tfidf[i]), 4) for i in top_indices]
        })
