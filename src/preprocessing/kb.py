import os
import pandas as pd
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.text_normalizer import TextNormalizer

MESH_PROCESSED = "data/processed/mesh_processed"

class Kb:
    
    def __init__(self, filepath, kb_type, id_column, name_column,
                 delimiter='\t', skip_rows=29, kb_file=None):
        """
        Initialize a Kb instance with metadata and path information.

        Args:
            filepath (str): Path to the KB file.
            kb_type (str): Type of KB, e.g., 'medic'.
            id_column (str): Name of the column with unique identifiers.
            name_column (str): Name of the column with KB entry names.
            delimiter (str): Delimiter used in the file.
            skip_rows (int): Rows to skip for headers.
            kb_file (str): Output file for processed KB in parquet format.
        """
        self.filepath = filepath
        self.kb_type = kb_type
        self.id_column = id_column
        self.name_column = name_column
        self.delimiter = delimiter
        self.skip_rows = skip_rows
        self.kb_file = kb_file
        self.dataframe = None
        self.column_names = None

    def create_parquet(self):
        """Load, clean, and normalize data, then save as a parquet file."""
        self.column_names = self.get_column_names_from_tsv(self.filepath, self.skip_rows)
        self.dataframe = pd.read_csv(self.filepath, sep=self.delimiter, header=None,
                                     names=self.column_names, skiprows=self.skip_rows)
        self.dataframe = self.clean_data()
        self.dataframe = self.normalize_data()
        self.kb_file = os.path.join(MESH_PROCESSED, f"{self.kb_type}_processed.parquet")
        self.dataframe.to_parquet(self.kb_file)

    def load_parquet(self):
        """Load the KB data from an existing parquet file."""
        self.dataframe = pd.read_parquet(self.kb_file)

    def create_labels(self):
        """Generate any labels for KB entries if needed."""
        pass  # Define label creation logic if needed

    def clean_data(self):
        """Clean the KB data using DataCleaner."""
        cleaner = DataCleaner(self.dataframe, self.kb_type, self.id_column)
        return cleaner.clean_data()

    def normalize_data(self):
        """Normalize KB data using TextNormalizer."""
        normalizer = TextNormalizer()
        return normalizer.normalize_dataframe(self.dataframe, self.kb_type)

    @staticmethod    
    def get_column_names_from_tsv(filepath, skip_rows):
        """Extract column names from the KB TSV file based on the skip rows."""
        with open(filepath, 'r') as fin:
            for _ in range(skip_rows - 1):
                line = fin.readline()
            col_names = [str(item).strip() for item in line.split("\t")]
            col_names[0] = col_names[0].replace('#', '').strip()  # Remove any leading #
        return col_names

    @staticmethod
    def fill_missing_values(df, percent=20):
        """
        Fill missing values in the DataFrame columns with 'Not Available' if the
        percentage of missing values in that column exceeds the threshold.

        Args:
            df (DataFrame): The KB DataFrame.
            percent (int): Threshold percentage to fill missing values.
        """
        for column in df.columns:
            miss_percent = df[column].isnull().mean() * 100
            if miss_percent >= percent:
                df[column] = df[column].fillna("Not Available")
