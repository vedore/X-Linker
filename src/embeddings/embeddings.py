import json
import os
import pandas as pd
import numpy as np
import pickle

from src.embeddings.clustering import HierarchicalClustering
from src.embeddings.vectorizer import Vectorizer

LABELS_FOLDER = "data/processed/index_labels"
EMBEDDINGS_FOLDER = "data/processed/embeddings"
CLUSTERING_FOLDER = "data/processed/clustering"

class Embeddings:

    kb_types = {'medic', 'chemical'}

    def __init__(self, kb_type=None, use_gpu=False):

        self.kb_type = kb_type
        self.use_gpu = use_gpu
        
        self.labels_data_dict = None
        self.processed_labels_data = None
        self.processed_labels_id= None
        self.embeddings = None
        self.clustering_df = None

    @classmethod
    def load_labels(cls, kb_type, labels_folder=LABELS_FOLDER):
        
        labels_file = os.path.join(labels_folder, f"{kb_type}_labels_processed.json")

        with open(labels_file, 'r') as json_file:
            data = json.load(json_file)

        instance = cls()
        instance.labels_data = data
        instance.kb_type = kb_type
        return instance

    def prepare_data(self):
        labels_dict = {}
        for labels_id, entries in self.labels_data.items():
            # Combine names and synonyms into a single text string for each entity
            names_and_synonyms = [entry.split(': ', 1)[1] for entry in entries]
            combined_text = ' '.join(names_and_synonyms)
            labels_dict[labels_id] = combined_text
        
        self.processed_labels_data = list(labels_dict.values())
        self.processed_labels_id = list(labels_dict.keys())

    def create_embeddings(self):
        vectorizer = Vectorizer(self.kb_type, self.use_gpu)
        self.embeddings = vectorizer.tfidf_vectorizer(self.processed_labels_data)
    
    def load_embeddings(self, embeddings_folder=EMBEDDINGS_FOLDER):
        # embeddings_cpu_file = os.path.join(embeddings_folder, f"{self.kb_type}_embeddings_cpu.npy")
        embeddings_gpu_file = os.path.join(embeddings_folder, f"{self.kb_type}_embeddings_gpu.pkl")
        print(embeddings_gpu_file)
        if os.path.exists(embeddings_gpu_file):
            with open(embeddings_gpu_file, 'rb') as pickle_file:
                self.embeddings = pickle.load(pickle_file)
            print("Loaded GPU Embeddings")
        # elif os.path.exists(embeddings_cpu_file):
        #     self.embeddings = np.load(embeddings_cpu_file, allow_pickle=True)
        #     print("Loaded CPU Embeddings")
        # else:
        #     print("File Doesn't Exist")

    def create_clustering(self):
        clustering = HierarchicalClustering(self.kb_type, self.use_gpu)
        self.clusters = clustering.hierarchical_clustering(self.embeddings, self.processed_labels_id)

    def load_clustering(self, clustering_folder=CLUSTERING_FOLDER): 
        clustering_file = os.path.join(clustering_folder, f"{self.kb_type}_clustering_gpu.parquet")
        self.clustering_df = pd.read_parquet(clustering_file)



