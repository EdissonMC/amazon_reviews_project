import os
import pandas as pd
import re
import time
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
print(df.head())

# Información del DataFrame
print("\nInformación del DataFrame:")
print(f"Total de filas: {len(df)}")
print(df['label'].value_counts())

# Verificar que tenemos todas las revisiones
print("\nPrimeras 3 revisiones completas:")
for i in range(min(3, len(df))):
    print(f"\nLabel {df.iloc[i]['label']}:")
    print(df.iloc[i]['review'][:150] + "..." if len(df.iloc[i]['review']) > 150 else df.iloc[i]['review'])