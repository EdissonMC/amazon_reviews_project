from amazon_reviews_project import config
from amazon_reviews_project.load_data import load_reviews_from_txt
from amazon_reviews_project.clean import batch_clean_reviews
from amazon_reviews_project.vectorize import create_vectorizer, fit_vectorizer, transform_text
from amazon_reviews_project.model import train_model, evaluate_model

def main():
    print("Loading training data...")
    df_train = load_reviews_from_txt(config.TRAIN_PATH)
    print("Training data loaded:")
    df_train.info()

    print("Loading test data...")
    df_test = load_reviews_from_txt(config.TEST_PATH)
    print("Test data loaded:")
    df_test.info()

    print("Cleaning training and test reviews...")
    df_train['clean_review'] = batch_clean_reviews(df_train['review'])
    df_test['clean_review'] = batch_clean_reviews(df_test['review'])

    # === TF-IDF Vectorization ===
    print("Creating and fitting TF-IDF vectorizer...")
    vectorizer = create_vectorizer()
    vectorizer = fit_vectorizer(vectorizer, df_train['clean_review'])

    print("Transforming training and test texts...")
    X_train = transform_text(vectorizer, df_train['clean_review'])
    X_test = transform_text(vectorizer, df_test['clean_review'])

    print("Embeddings generation complete.")
    print(f"Train embeddings shape: {X_train.shape}")
    print(f"Test embeddings shape: {X_test.shape}")

    # Optional: Train and evaluate the model
    # y_train = df_train['label']  # Ensure 'label' column exists
    # y_test = df_test['label']
    print("Training model...")
    model = train_model(X_train, df_train['label'])
    print("Evaluating model...")
    evaluate_model(model, X_test, df_test['label'])

if __name__ == "__main__":
    main()
