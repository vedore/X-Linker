import pandas as pd

class DataMop:

    def __init__(self, kb_type):
        self.kb_type = kb_type

    @staticmethod
    def mop():
        return


    def drop_duplicate_rows(self):
        self.dataframe.drop_duplicates(subset=[self.id_column], inplace=True)
        return self.dataframe

    def fill_missing_data(self, threshold_percent=20):
        for column in self.dataframe.columns:
            miss_percent = self.dataframe[column].isnull().mean() * 100
            if miss_percent >= float(threshold_percent):
                self.dataframe[column] = self.dataframe[column].fillna("Not Available")
        return self.dataframe

    def process_knowledge_base_columns(self):
        if self.kb_type == 'medic':
            columns_to_split = ['AltDiseaseIDs', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings']
            for column in columns_to_split:
                self.dataframe[column] = self.dataframe[column].apply(lambda x: x.split('|') if pd.notnull(x) else [])
        return self.dataframe




    

    


    


