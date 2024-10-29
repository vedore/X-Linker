import pandas as pd

class DataCleaner:
    """
    A class for cleaning and processing data within a DataFrame, specifically designed for knowledge base (KB) data.
    """

    def __init__(self, dataframe, kb_type, id_column):
        """
        Initializes the DataCleaner with the DataFrame, type of knowledge base, and column to identify duplicates.

        Args:
            dataframe (pd.DataFrame): The DataFrame to clean.
            kb_type (str): The type of knowledge base, such as 'medic'.
            id_column (str): The name of the column to check for duplicates.
        """
        self.dataframe = dataframe
        self.kb_type = kb_type
        self.id_column = id_column

    def drop_duplicate_rows(self):
        """
        Removes duplicate rows in the DataFrame based on the specified ID column.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        self.dataframe.drop_duplicates(subset=[self.id_column], inplace=True)
        return self.dataframe

    def fill_missing_data(self, threshold_percent=20):
        """
        Fills missing values in columns where the percentage of missing data exceeds a specified threshold.

        Args:
            threshold_percent (float): The percentage threshold above which columns will have missing
                                       values filled with "Not Available" (default is 20).

        Returns:
            pd.DataFrame: DataFrame with missing values filled as specified.
        """
        for column in self.dataframe.columns:
            miss_percent = self.dataframe[column].isnull().mean() * 100
            if miss_percent >= float(threshold_percent):
                self.dataframe[column] = self.dataframe[column].fillna("Not Available")
        return self.dataframe

    def process_knowledge_base_columns(self):
        """
        Processes specific columns based on the knowledge base type. For a 'medic' knowledge base, certain columns
        are transformed from strings with delimiter '|' to lists.

        Returns:
            pd.DataFrame: DataFrame with processed columns based on knowledge base type.
        """
        if self.kb_type == 'medic':
            columns_to_split = ['AltDiseaseIDs', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings']
            for column in columns_to_split:
                self.dataframe[column] = self.dataframe[column].apply(lambda x: x.split('|') if pd.notnull(x) else [])
        return self.dataframe

    def clean_data(self, threshold_percent=20):
        """
        Performs all data cleaning operations in sequence: removing duplicates, filling missing data, 
        processing KB-specific columns, and then returns the cleaned DataFrame.

        Args:
            threshold_percent (float): The percentage threshold above which columns will have missing
                                       values filled with "Not Available" (default is 20).

        Returns:
            pd.DataFrame: The fully cleaned DataFrame.
        """
        self.drop_duplicate_rows()
        self.fill_missing_data(threshold_percent=threshold_percent)
        self.process_knowledge_base_columns()
        return self.dataframe



    

    


    


