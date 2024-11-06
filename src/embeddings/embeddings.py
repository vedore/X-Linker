import json
import os
import pandas as pd

from src.embeddings.clustering import HierarchicalClustering
from src.embeddings.vectorizer import Vectorizer

LABELS_PROCESSED = "data/processed/index_labels"
EMBEDDINGS_PROCESSED = "data/processed/embeddings"

class Embeddings:

    kb_types = {'medic', 'chemical'}

    def __init__(self, kb_type=None, use_gpu=False):

        self.kb_type = kb_type
        self.use_gpu = use_gpu
        
        self.labels_data_dict = None
        self.processed_labels_data = None
        self.processed_labels_id= None
        self.embeddings = None
        self.clusters = None

    @classmethod
    def load_labels(cls, kb_type, labels_folder=LABELS_PROCESSED):
        
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

        print("Data Prepared")

    def create_embeddings(self):
        vectorizer = Vectorizer(self.kb_type, self.use_gpu)
        self.embeddings = vectorizer.tfidf_vectorizer(self.processed_labels_data)

        print("Embeddings Done")
    
    def load_embeddings(self):
        return 
    
    def save_embeddings(self):
        return 

    def clustering(self):
        clustering = HierarchicalClustering(self.kb_type, self.use_gpu)
        self.clusters = clustering.hierarchical_clustering(self.embeddings, self.processed_labels_id)

    def save_clustering(self):
        return


