import os
import pandas as pd
import json

from src.utils.data_cleaner import DataCleaner
from src.utils.text_normalizer import TextNormalizer

MESH_PROCESSED = "data/processed/mesh_processed"
LABELS_PROCESSED = "data/processed/index_labels"

class Kb:
    """Knowledge Base Class for handling KB data processing."""

    kb_types = {'medic', 'chemical'}

    kb_columns = {'medic': 
                  {'id':'DiseaseID', 'name':'DiseaseName', 'synonyms':'Synonyms'},
                  'chemical':
                  {'id':'ChemicalID', 'name':'ChemicalName', 'synonyms':'Synonyms'}}
    
    def __init__(self):
        """
        Initialize a Kb instance with an empty dataframe placeholder.

        Attributes:
            dataframe (DataFrame): Placeholder for the cleaned dataframe.
        """
        self.dataframe = None
        self.kb_type = None

    def save(self, kb_type, kb_folder=MESH_PROCESSED):
        """Save cleaned dataframe from KB to disk.

        Args:
            kb_folder (str): Folder to save to.
            kb_type (str): Name of the type to save.
        """
        os.makedirs(kb_folder, exist_ok=True)
        self.dataframe.to_parquet(os.path.join(kb_folder, f"{kb_type.lower()}_mesh_processed.parquet"))
        self.kb_type = kb_type

    @classmethod
    def load(cls, kb_type, kb_folder=MESH_PROCESSED):
        """Load a saved cleaned DataFrame from disk.

        Args:
            kb_folder (str): Folder where `kb` was saved to using `kb.save`.
            kb_type (str): Type of KB to load.

        Returns:
            Kb: The loaded Kb object or a Kb object with None as DataFrame if the folder does not exist.
        """
        if os.path.exists(kb_folder):
            if kb_type in cls.kb_types:
                dataframe = pd.read_parquet(os.path.join(kb_folder, f"{kb_type.lower()}_mesh_processed.parquet"))
                instance = cls()
                instance.dataframe = dataframe
                instance.kb_type = kb_type
                return instance
        print("Doesn't Exist")
        return cls()

    @classmethod
    def clean_dataframe(cls, data_filepath, kb_type, id_column, delimiter, skip_rows, kb_folder=MESH_PROCESSED):
        """Load, clean, and normalize data, then save as a parquet file.

        Args:
            data_filepath (str): Path to the input data file.
            kb_type (str): Type of KB, e.g., 'medic'.
            skip_rows (int): Number of rows to skip in the file.
            delimiter (str): Delimiter used in the input file.
            id_column (str): Name of the column with unique identifiers.
            kb_folder (str): Folder to save the cleaned data.

        Returns:
            Kb: A Kb object with an cleaned Dataframe
        """
        column_names = cls._get_column_names_from_tsv(data_filepath, skip_rows)
        dataframe = pd.read_csv(data_filepath, sep=delimiter, header=None,
                                 names=column_names, skiprows=skip_rows)
        
        # Create an instance of the class to work with the dataframe
        instance = cls()
        instance.dataframe = dataframe
        
        # Clean and normalize the data
        instance.dataframe = instance._clean_data(instance.dataframe, kb_type, id_column)
        instance.dataframe = instance._normalize_data(instance.dataframe, kb_type)

        # Save the cleaned dataframe
        instance.save(kb_type, kb_folder)

        return instance

    def create_labels(self, labels_folder=LABELS_PROCESSED):
        """Generate labels for KB entries and save them to a JSON file.

        Args:
            labels_folder (str): Folder path to save the labels file.
        """
        if self.kb_type == 'medic':
            all_labels = []
            for _, row in self.dataframe.iterrows():
                primary_label = f"{row['DiseaseID']}: {row['DiseaseName']}"
                all_labels.append({'DiseaseID': row['DiseaseID'], 'Label': primary_label})
                
                for synonyms in row['Synonyms']:
                    synonyms_label = f"{row['DiseaseID']}: {synonyms}"
                    all_labels.append({'DiseaseID': row['DiseaseID'], 'Label': synonyms_label})

            all_labels_df = pd.DataFrame(all_labels)
            json_data = all_labels_df.groupby('DiseaseID')['Label'].apply(list).to_dict()

            os.makedirs(labels_folder, exist_ok=True)
            with open(os.path.join(labels_folder, f"{self.kb_type}_labels_processed.json"), 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

    @staticmethod
    def _clean_data(dataframe, kb_type, id_column):
        """Clean the KB data using DataCleaner.

        Args:
            dataframe (DataFrame): The KB DataFrame.
            kb_type (str): Type of KB.
            id_column (str): Name of the column with unique identifiers.

        Returns:
            DataFrame: The cleaned DataFrame.
        """
        cleaner = DataCleaner(dataframe, kb_type, id_column)
        return cleaner.clean_data()

    @staticmethod
    def _normalize_data(dataframe, kb_type):
        """Normalize KB data using TextNormalizer.

        Args:
            dataframe (DataFrame): The KB DataFrame.
            kb_type (str): Type of KB.

        Returns:
            DataFrame: The normalized DataFrame.
        """
        normalizer = TextNormalizer()
        return normalizer.normalize_dataframe(dataframe, kb_type)

    @staticmethod    
    def _get_column_names_from_tsv(filepath, skip_rows):
        """Extract column names from the KB TSV file based on the skip rows.

        Args:
            filepath (str): Path to the TSV file.
            skip_rows (int): Number of rows to skip.

        Returns:
            list: List of column names extracted from the TSV file.
        """
        with open(filepath, 'r') as fin:
            for _ in range(skip_rows - 1):
                line = fin.readline()
            col_names = [str(item).strip() for item in line.split("\t")]
            col_names[0] = col_names[0].replace('#', '').strip()
        return col_names

    @staticmethod
    def _fill_missing_values(df, percent=20):
        """Fill missing values in the DataFrame columns with 'Not Available' if the
        percentage of missing values in that column exceeds the threshold.

        Args:
            df (DataFrame): The KB DataFrame.
            percent (int): Threshold percentage to fill missing values.
        """
        for column in df.columns:
            miss_percent = df[column].isnull().mean() * 100
            if miss_percent >= percent:
                df[column] = df[column].fillna("Not Available")

