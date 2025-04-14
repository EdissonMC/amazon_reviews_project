
# Amazon Reviews Sentiment Classifier

This project implements a sentiment classification system for Amazon product reviews using classical machine learning and natural language processing (NLP) techniques.

---

## 🔍 Core Features

- Efficient text preprocessing and cleaning.
- TF-IDF vectorization (lightweight and interpretable).
- Training and evaluation using classical ML models (e.g., Logistic Regression, SVM).
- Modular and maintainable codebase with Poetry environment management.

---

## 📦 Dataset

We use the [Amazon Reviews Polarity dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) from Kaggle.

### 📁 File Format

The dataset is provided in compressed `.bz2` format:

- `train.ft.txt.bz2`
- `test.ft.txt.bz2`

### 🛠️ How to Prepare the Data

1. Download the dataset files from Kaggle.
2. Decompress using:

```bash
bzip2 -d train.ft.txt.bz2
bzip2 -d test.ft.txt.bz2

By default, the project uses a TF-IDF vectorizer trained  on the dataset.

If desired, you can switch to using sentence-transformers by modifying the embedding logic in main.py.

If desired, you can switch to using sentence-transformers by modifying the vectorize.py or creating embedding logic and refactory in main.py.


###  🧠 Model
Currently using a logistic regression model for binary sentiment classification. Can be replaced with SVM or other classifiers using scikit-learn.



Project Structure

amazon_reviews_project/
├── clean.py               # Text cleaning functions
├── config.py              # Configuration constants
├── embeddings.py          # (Optional) sentence-transformers interface
├── load_data.py           # Load data from .txt files
├── model.py               # Train and evaluate model
├── vectorizacion.py       # TF-IDF vectorizer utilities
├── main.py                # Main orchestration script
data/
├── train.ft.txt
└── test.ft.txt

### 🚀 How to Run (with Poetry)

Install Poetry if you haven't:
> curl -sSL https://install.python-poetry.org | python3 -

Install dependencies:
> poetry install

Run the script inside the Poetry environment:
> poetry run python -m amazon_reviews_project.main


