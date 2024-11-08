import os
import numpy as np

import subprocess

try:
    subprocess.check_output('nvidia-smi')
    GPU_AVAILABLE = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer as cuML_TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer as sk_TfidfVectorizer

EMBEDDINGS_FOLDER = "data/processed/embeddings"

class Vectorizer():

    def __init__(self, kb_type, use_gpu=False):
        self.kb_type = kb_type
        self.use_gpu = use_gpu and GPU_AVAILABLE

        self.embeddings_data = None

    def save(self,  embeddings_folder= EMBEDDINGS_FOLDER):
        if self.use_gpu:
            embedding_file = f"{self.kb_type}_embeddings_gpu.npy"
        else:
            embedding_file = f"{self.kb_type}_embeddings_cpu.npy"
        np.save(os.path.join(embeddings_folder, embedding_file), self.embeddings)

    def tfidf_vectorizer(self, processed_labels_data):
        if self.use_gpu:
            vectorizer = cuML_TfidfVectorizer()
            data = cudf.Series(processed_labels_data)
        else:
            vectorizer = sk_TfidfVectorizer()
            data = processed_labels_data

        embeddings = vectorizer.fit_transform(data)
        self.embeddings = embeddings
        self.save()
        return embeddings

    




