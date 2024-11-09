import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class TextNormalizer:
    """
    A class for text normalization, which performs a variety of preprocessing tasks
    such as stopword removal, lemmatization, punctuation removal, and whitespace cleaning.
    """

    def __init__(self):
        """
        Initializes the TextNormalizer with a set of English stopwords and a WordNet lemmatizer.
        """
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords_from_text(self, text):
        """
        Removes stopwords from a given text.

        Args:
            text (str): The text to process.

        Returns:
            str: Text with stopwords removed.
        """
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return " ".join(filtered_tokens)

    def lemmatize_text(self, text):
        """
        Lemmatizes each token in the text to its root form.

        Args:
            text (str): The text to lemmatize.

        Returns:
            str: Text with each word lemmatized.
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized_tokens)

    def convert_to_lowercase(self, text):
        """
        Converts all characters in the text to lowercase.

        Args:
            text (str): The text to convert.

        Returns:
            str: Text in lowercase.
        """
        return text.lower()

    # Retirar todos a pontuacao de coisas que tem de proposito?
    def remove_punctuation_from_text(self, text):
        """
        Removes punctuation from the text.

        Args:
            text (str): The text to process.

        Returns:
            str: Text without punctuation.
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_extra_whitespace(self, text):
        """
        Removes extra spaces within the text.

        Args:
            text (str): The text to process.

        Returns:
            str: Text with extra whitespace removed.
        """
        return " ".join(text.split())

    def normalize_text(self, text):
        """
        Applies a sequence of normalization steps on the text: lowercase conversion, punctuation
        removal, whitespace trimming, stopword removal, and lemmatization.

        Args:
            text (str): The text to normalize.

        Returns:
            str: Fully normalized text.
        """
        text = self.convert_to_lowercase(text)
        text = self.remove_punctuation_from_text(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_stopwords_from_text(text)
        text = self.lemmatize_text(text)
        return text

    def normalize_list(self, text_list):
        """
        Normalizes each element in a list of text entries.

        Args:
            text_list (list): A list of strings to normalize.

        Returns:
            list: A list of normalized strings.
        """
        return [self.normalize_text(text) for text in text_list]

    def normalize_dataframe(self, df, kb_type):
        """
        Normalizes specific columns of a DataFrame based on the type of knowledge base.

        Args:
            df (pd.DataFrame): The DataFrame to normalize.
            kb_type (str): The type of knowledge base (e.g., 'medic').

        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if kb_type == 'medic':
            return self._normalize_medic_data(df)

    def _normalize_medic_data(self, df):
        """
        Normalizes columns specific to the medical knowledge base, such as 'Definition', 'Synonyms', and 'SlimMappings'.

        Args:
            df (pd.DataFrame): The DataFrame to normalize.

        Returns:
            pd.DataFrame: DataFrame with specified columns normalized.
        """
        kb_medic = {'Definition', 'Synonyms', 'SlimMappings'}
        df['DiseaseName'] = df['DiseaseName'].apply(self.normalize_text)
        print("DiseaseName column normalization finished")
        for col in kb_medic:
            df[col] = df[col].apply(self.normalize_list)
            print(f"{col} column normalization finished")
        return df





    