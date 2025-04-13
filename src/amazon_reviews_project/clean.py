



import re
from tqdm import tqdm



# SECTION OF DATA CLEAN

def clean_review(text):
    # Lowercase
    text = text.lower()

    # Keep punctuation that may carry sentiment
    # (.,!? are typically useful in sentiment, remove only symbols that are noise)
    text = re.sub(r"[^\w\s\.,!?$:/\-']", "", text)

    # Optional: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Clean the reviews in batches for performance and better memory usage
def batch_clean_reviews(reviews, batch_size=10000):
    cleaned_reviews = []
    total = len(reviews)

    # Process in chunks
    for i in tqdm(range(0, total, batch_size), desc="Cleaning reviews"):
        batch = reviews[i:i + batch_size]
        cleaned = [clean_review(text) for text in batch]
        cleaned_reviews.extend(cleaned)
    
    return cleaned_reviews
