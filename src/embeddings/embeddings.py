import json
import os
import pandas as pd

import cupy as cp
import cudf
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.cluster import AgglomerativeClustering

from scipy.sparse import save_npz, load_npz

LABELS_PROCESSED = "data/processed/index_labels"
EMBEDDINGS_PROCESSED = "data/processed/embeddings"

class Embeddings:
    """
    A class to handle embeddings data for a knowledge base (KB) type, including 
    loading, preparing, saving, and clustering embeddings.

    Attributes:
        embeddings_data_unprocessed (dict): Raw embeddings data loaded from a file.
        embeddings_data_processed (dict): Processed embeddings data ready for saving or further use.
        embeddings_matrix (csr_matrix): Matrix of embeddings created using TF-IDF.
        embeddings_ids (list): List of IDs corresponding to processed embeddings.
        kb_type (str): Type of knowledge base for this embeddings data.
    """

    kb_types = {'medic', 'chemical'}

    def __init__(self):
        """Initialize an Embeddings instance with placeholders for data and KB type."""
        self.embeddings_data_unprocessed = None
        self.embeddings_data_processed = None
        self.embeddings_matrix = None
        self.embeddings_ids = None
        self.kb_type = None

    @classmethod
    def load_unprocessed(cls, kb_type, labels_folder=LABELS_PROCESSED):
        """
        Load unprocessed embeddings data from a JSON file.

        Args:
            kb_type (str): Type of knowledge base.
            labels_folder (str): Directory where the labels JSON file is stored.

        Returns:
            Embeddings: An instance of the Embeddings class with loaded unprocessed data.
        """
        labels_file = os.path.join(labels_folder, f"{kb_type}_labels_processed.json")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"File {labels_file} not found.")

        with open(labels_file, 'r') as labels_json:
            data = json.load(labels_json)

        instance = cls()
        instance.embeddings_data_unprocessed = data
        instance.kb_type = kb_type
        return instance

    def prepare_data(self):
        """
        Process the raw embeddings data by combining names and synonyms for each entity.
        
        Populates `embeddings_data_processed` with combined text for each ID and sets `embeddings_ids`.
        """
        labels_dict = {}
        for labels_id, entries in self.embeddings_data_unprocessed.items():
            # Combine names and synonyms into a single text string for each entity
            names_and_synonyms = [entry.split(': ', 1)[1] for entry in entries]
            combined_text = ' '.join(names_and_synonyms)
            labels_dict[labels_id] = combined_text
        self.embeddings_data_processed = labels_dict
        self.embeddings_ids = list(labels_dict.keys())

    def create_embeddings(self):
        """
        Create a TF-IDF vectorized matrix from the processed embeddings data.
        
        Populates `embeddings_matrix` with the TF-IDF values.
        """
        if not self.embeddings_data_processed:
            raise ValueError("Embeddings data has not been processed. Call prepare_data() first.")
        
        vectorizer = TfidfVectorizer()
        data_df = cudf.DataFrame.from_dict(self.embeddings_data_processed, orient='index', columns=['text'])
        self.embeddings_matrix = vectorizer.fit_transform(data_df['text'].to_array())

    def save_processed(self, kb_type, embeddings_folder=EMBEDDINGS_PROCESSED):
        """
        Save the processed embeddings matrix and IDs to disk.

        Args:
            kb_type (str): Type of knowledge base.
            embeddings_folder (str): Directory to save the processed embeddings files.
        """
        os.makedirs(embeddings_folder, exist_ok=True)

        # Save embeddings matrix
        embeddings_matrix_path = os.path.join(embeddings_folder, f"{kb_type}_embeddings_matrix_processed.npz")
        save_npz(embeddings_matrix_path, self.embeddings_matrix)

        # Save embeddings IDs
        embeddings_ids_path = os.path.join(embeddings_folder, f"{kb_type}_embeddings_id_processed.json")
        with open(embeddings_ids_path, "w") as json_file:
            json.dump(self.embeddings_ids, json_file, indent=4)

    @classmethod
    def load_processed(cls, kb_type, embeddings_folder=EMBEDDINGS_PROCESSED):
        """
        Load processed embeddings matrix and IDs from disk.

        Args:
            kb_type (str): Type of knowledge base.
            embeddings_folder (str): Directory containing the processed embeddings files.

        Returns:
            Embeddings: An instance of the Embeddings class with loaded processed data.
        """
        instance = cls()
        instance.kb_type = kb_type

        embeddings_matrix_path = os.path.join(embeddings_folder, f"{kb_type}_embeddings_matrix_processed.npz")
        embeddings_ids_path = os.path.join(embeddings_folder, f"{kb_type}_embeddings_id_processed.json")

        if not os.path.exists(embeddings_matrix_path) or not os.path.exists(embeddings_ids_path):
            raise FileNotFoundError(f"Processed embeddings files for {kb_type} not found in {embeddings_folder}.")

        instance.embeddings_matrix = load_npz(embeddings_matrix_path)

        with open(embeddings_ids_path, 'r') as ids_file:
            instance.embeddings_ids = json.load(ids_file)

        return instance

    def clustering(self):
        """
        Perform hierarchical clustering on the embeddings matrix.

        Returns:
            dict: A dictionary mapping entity IDs to cluster labels.
        """
        if self.embeddings_matrix is None:
            raise ValueError("Embeddings matrix is not created. Call create_embeddings() first.")

        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric='cosine',
            linkage='average'
        )
        clustering_model.fit(self.embeddings_matrix.toarray())  # Convert sparse matrix for clustering
        clusters = dict(zip(self.embeddings_ids, clustering_model.labels_))
        return clusters
