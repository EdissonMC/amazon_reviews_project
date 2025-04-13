

import pandas as pd



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
