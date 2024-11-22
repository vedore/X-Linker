import os
import pickle
import pandas as pd
import string
import nltk
import re
import json

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
}

class KnowledgeBase():

    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def save(self, kb_folder):
        os.makedirs(kb_folder, exist_ok=True)
        with open(os.path.join(kb_folder, 'knowledge_base.pkl'), 'wb') as fout:
            pickle.dump(self.dataframe, fout)
    
    @classmethod
    def load(cls, kb_folder):
        kb_path = os.path.join(kb_folder, 'knowledge_base.pkl')
        assert os.path.exists(kb_path), f"{kb_path} does not exist"
        with open(kb_path, 'rb') as fin:
            return cls(pickle.load(fin))
    
    @classmethod
    def mop(cls, kb_type, kb_filepath):
        assert os.path.exists(kb_filepath) is True, f"{kb_filepath} does not exist"

        defaults = {
            'sep': '\t',
            'header': None,
            'skiprows': 29,
            'names': cls.get_column_names(kb_filepath, 29)
        }
        
        df = KnowledgeBaseCleaner.clean(kb_type, pd.read_csv(kb_filepath, **defaults))
        df = KnowledgeBaseTextNormalizer.normalize_dataframe(kb_type, df)
        return cls(df)
    
 
    def extract_labels(self, labels_folder):
        KnowledgeBaseLabelsExtraction.extract_labels(self.kb_type, self.dataframe, labels_folder)

    @staticmethod
    def get_column_names(kb_filepath, skip_rows):
        with open(kb_filepath, 'r') as fin:
            for _ in range(skip_rows - 1):
                line = fin.readline()
                col_names = [str(item).strip() for item in line.split("\t")]
                col_names[0] = col_names[0].replace('#', '').strip()
        return col_names
    

class KnowledgeBaseLabelsExtraction():
    
    def __init__(self, labels_dict=None):
        self.labels_dict = labels_dict

    @classmethod
    def extract_labels(cls, kb_type, dataframe):
        if kb_type == 'medic':
            all_labels = []
            for _, row in dataframe.iterrows():
                primary_label = f"{row['DiseaseID']}: {row['DiseaseName']}"
                all_labels.append({'DiseaseID': row['DiseaseID'], 'Label': primary_label})
                
                for synonyms in row['Synonyms']:
                    synonyms_label = f"{row['DiseaseID']}: {synonyms}"
                    all_labels.append({'DiseaseID': row['DiseaseID'], 'Label': synonyms_label})

            all_labels_df = pd.DataFrame(all_labels)
            labels_dict = all_labels_df.groupby('DiseaseID')['Label'].apply(list).to_dict()
            return cls(labels_dict)

    
    def save(self, labels_folder):
        os.makedirs(labels_folder, exist_ok=True)
        with open(os.path.join(labels_folder, 'labels.json'), 'w') as fout:
            json.dump(self.labels_dict, fout, indent=4)

    @classmethod
    def load(cls, labels_folder):
        labels_filepath = os.path.join(labels_folder, 'labels.json')
        assert os.path.exists(labels_filepath), f"{labels_filepath} does not exist"
        with open(labels_filepath, 'r') as fin:
            labels = json.load(fin)
        return cls(labels)


class KnowledgeBaseCleaner():
    
    @classmethod
    def clean(cls, kb_type, dataframe):
        assert kb_type in list(kb_dict.keys()), f"{kb_type} is not supported"
        id_column = kb_dict[kb_type]
        cls.drop_duplicates(id_column, dataframe)
        cls.fill_missing_data(dataframe)
        cls.process_knowledge_base_columns(kb_type, dataframe)     
        return dataframe

    @staticmethod
    def drop_duplicates(id_column, dataframe):
        dataframe.drop_duplicates(subset=[id_column], inplace=True)

    @staticmethod
    def fill_missing_data(dataframe, threshold_percent=20):
        for column in dataframe.columns:
            miss_percent = dataframe[column].isnull().mean() * 100
            if miss_percent >= float(threshold_percent):
                dataframe[column] = dataframe[column].fillna("Not Available")

    @staticmethod
    def process_knowledge_base_columns(kb_type, dataframe):
        if kb_type == 'medic':
            columns_to_split = ['AltDiseaseIDs', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings']
            for column in columns_to_split:
                dataframe[column] = dataframe[column].apply(lambda x: x.split('|') if isinstance(x, str) else [])
        return dataframe
    
class KnowledgeBaseTextNormalizer():
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()    

    @classmethod
    def normalize_dataframe(cls, kb_type, dataframe):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer() 

        if kb_type == 'medic':
            kb_medic = ['Definition', 'Synonyms', 'SlimMappings']
                
            dataframe['DiseaseName'] = dataframe['DiseaseName'].apply(lambda text: cls.normalize_text(stop_words, lemmatizer, text))
            for col in kb_medic:
                dataframe[col] = dataframe[col].apply(lambda list: cls.normalize_list(stop_words, lemmatizer, list))
        
        return dataframe

    @classmethod
    def normalize_list(cls, stop_words, lemmatizer, text_list):
        return [cls.normalize_text(stop_words, lemmatizer, text) for text in text_list]

    @staticmethod
    def normalize_text(stop_words, lemmatizer, text):
        # Lowercase the text
        text = text.lower()
            
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Tokenize the text
        tokens = word_tokenize(text)
            
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        # Join tokens back into a single string
        normalized_text = " ".join(filtered_tokens)
            
        # Clean up extra whitespace
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            
        return normalized_text
    
