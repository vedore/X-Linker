import json
import os

LABELS_PROCESSED = "data/processed/index_labels"
EMBEDDINGS_PROCESSED = "data/processed/embeddings"

class Embeddings:
    """
    A class to handle embeddings data for a knowledge base (KB) type.

    Attributes:
        embeddings_data_unprocessed (dict): Raw embeddings data loaded from a file.
        embeddings_data_processed (dict): Processed embeddings data ready for saving or further use.
        kb_type (str): Type of knowledge base for this embeddings data.
    """

    def __init__(self):
        """
        Initializes an Embeddings instance with placeholders for data and KB type.
        """
        self.embeddings_data_unprocessed = None
        self.embeddings_data_processed = None
        self.kb_type = None

    def save(self, kb_type, embeddings_folder=EMBEDDINGS_PROCESSED):
        """
        Save processed embeddings data to disk in JSON format.

        Args:
            kb_type (str): Type of knowledge base.
            embeddings_folder (str): Directory to save the processed embeddings file.
        """
        os.makedirs(embeddings_folder, exist_ok=True)
        embeddings_file = os.path.join(embeddings_folder, f"{kb_type}_embeddings_processed.json")
        with open(embeddings_file, "w") as json_file:
            json.dump(self.embeddings_data_processed, json_file, indent=4)

    @classmethod
    def load(cls, kb_type, labels_folder=LABELS_PROCESSED):
        """
        Load unprocessed embeddings data from a JSON file.

        Args:
            kb_type (str): Type of knowledge base.
            labels_folder (str): Directory where the labels JSON file is stored.

        Returns:
            Embeddings: An instance of the Embeddings class with loaded data.
        """
        labels_file = os.path.join(labels_folder, f"{kb_type}_labels_processed.json")
        with open(labels_file, 'r') as labels_json:
            data = json.load(labels_json)

        instance = cls()
        instance.embeddings_data_unprocessed = data
        instance.kb_type = kb_type
        return instance

    def prepare_data(self):
        """
        Process the raw embeddings data by combining names and synonyms for each disease.

        This method populates `embeddings_data_processed` with combined text for each disease,
        making it ready for saving or further processing.
        """
        disease_texts = {}
        for disease_id, entries in self.embeddings_data_unprocessed.items():
            # Combine names and synonyms into a single text string for each disease
            names_and_synonyms = [entry.split(': ', 1)[1] for entry in entries]
            combined_text = ' '.join(names_and_synonyms)
            disease_texts[disease_id] = combined_text
        self.embeddings_data_processed = disease_texts
