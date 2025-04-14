

import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack


DEFAULT_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def create_vectorizer(max_features=5000):
    return TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        dtype=np.float32
    )


def fit_vectorizer(vectorizer, texts, save_path=DEFAULT_VECTORIZER_PATH, batch_size=10000):
    """
    Fit the vectorizer in batches and save it.
    """
    print("Fitting vectorizer in batches...")
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    # Entrenamos el vectorizador completo (TfidfVectorizer no tiene partial_fit)
    vectorizer.fit(texts)  # Este es el paso pesado

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"Vectorizer saved to {save_path}")
    return vectorizer


def transform_text(vectorizer, texts, batch_size=10000):
    """
    Transform texts in batches using the fitted vectorizer.
    """
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        X_batch = vectorizer.transform(batch)
        results.append(X_batch)

    # Stack sparse matrices
    from scipy.sparse import vstack
    return vstack(results)
