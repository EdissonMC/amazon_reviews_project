

import os


# Obtener el directorio actual y subir dos niveles de una vez
# BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR  = os.path.dirname(os.path.dirname(current_dir))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.txt')
TEST_PATH = os.path.join(DATA_DIR, 'test.txt')