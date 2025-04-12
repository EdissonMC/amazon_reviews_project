import os
import pandas as pd
import re
import time
import string
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Obtener el directorio actual y subir dos niveles de una vez
current_dir = os.path.dirname(os.path.abspath(__file__))
two_levels_up = os.path.dirname(os.path.dirname(current_dir))

# Acceder al archivo en la carpeta data
train_path = os.path.join(two_levels_up, "data", "train.txt")
test_path = os.path.join(two_levels_up, "data", "test.txt")

print("")
print(train_path)
print(test_path)

# # Leer los archivos
# train_data = pd.read_csv(train_path, delimiter=' ', header=None, names=['label', 'review'], engine='python')

# # Extraer las reseñas y las etiquetas (las etiquetas están en la primera columna, el resto es la reseña)
# train_data['label'] = train_data['label'].apply(lambda x: x.replace('__label__', ''))  # Limpiar la etiqueta

# # Mostrar las primeras filas
# print(train_data.head())


def load_reviews_from_txt(file_path):
    labels = []
    reviews = []
    current_review_lines = []
    current_label = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__label__"):
                # Guardamos la reseña anterior
                if current_label is not None:
                    reviews.append(" ".join(current_review_lines).strip())
                    labels.append(current_label)
                # Nueva reseña
                parts = line.split(" ", 1)
                current_label = parts[0].replace("__label__", "")
                current_review_lines = [parts[1]] if len(parts) == 2 else []
            else:
                current_review_lines.append(line)

        # Última reseña
        if current_label is not None:
            reviews.append(" ".join(current_review_lines).strip())
            labels.append(current_label)

    return pd.DataFrame({'label': labels, 'review': reviews})



start = time.time()
# Cargar los datos y crear el DataFrame
df_train = load_reviews_from_txt(train_path)
df_test = load_reviews_from_txt(test_path)

end = time.time()
print(f"Tiempo: {end - start:.2f} segundos")
# Mostrar las primeras filas para verificar
print(df_train.head())

# Información del DataFrame
print("\nInformación del DataFrame:")
print(f"Total de filas: {len(df_train)}")
print(df_train['label'].value_counts())

# Verificar que tenemos todas las revisiones
print("\nPrimeras 3 revisiones completas:")
for i in range(min(3, len(df_train))):
    print(f"\nLabel {df_train.iloc[i]['label']}:")
    print(df_train.iloc[i]['review'][:150] + "..." if len(df_train.iloc[i]['review']) > 150 else df_train.iloc[i]['review'])


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

# Apply cleaning with batching (this will show a progress bar in the terminal)
start = time.time()
df_train['clean_review'] = batch_clean_reviews(df_train['review'].tolist())
end = time.time()
print(f"Time clean review train: {end - start:.2f} segundos")
print(df_train.head())



start = time.time()
df_test['clean_review'] = batch_clean_reviews(df_test['review'].tolist())
end = time.time()
print(f"Time clean review test : {end - start:.2f} segundos")
print(df_train.head())

print("\nEjemplos limpios:")
for i in range(3):
    print(f"\nLabel {df_train.iloc[i]['label']} (limpia):")
    print(df_train.iloc[i]['clean_review'][:150] + "...")


#VECTORIZATION AND BASE MODEL


# Split train dataset for validation
# Using a validation set (20%) to evaluate baseline model
X_train, X_val, y_train, y_val = train_test_split(df_train['review'], df_train['label'], test_size=0.2, random_state=42)

# Convert text into TF-IDF vectors
# Limiting max features to reduce dimensionality and memory
vectorizer = TfidfVectorizer(max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train a logistic regression model as baseline
print("Training logistic regression model...")
model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_val_vec)

# Print performance metrics
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()