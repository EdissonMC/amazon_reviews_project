
from amazon_reviews_project import config
from amazon_reviews_project.load_data import load_reviews_from_txt
from amazon_reviews_project.clean import batch_clean_reviews
from amazon_reviews_project.vectorize import vectorize_data
from amazon_reviews_project.model import train_model, evaluate_model

def main():
    print("Loading data...")
    df = load_reviews_from_txt(config.TRAIN_PATH)
    df.info()
    print("Cleaning data...")
    df['review'] = batch_clean_reviews(df['review'])

    print("Vectorizando...")
    X_train, X_test, y_train, y_test, _ = vectorize_data(df)

    print("Entrenando modelo...")
    model = train_model(X_train, y_train)

    print("Evaluando...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()