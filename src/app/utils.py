from src.trainning.cpu.train import TrainCPU
from src.extractor.knowledge_base import KnowledgeBase, KnowledgeBaseLabelsExtraction
from src.featurization.preprocessor import Preprocessor
from src.machine_learning.cpu.ml import AgglomerativeClusteringCPU
from src.featurization.vectorizer import TfidfVectorizer
from src.machine_learning.clustering import Clustering


def iniatilize_knowledge_base(kb_type, kb_location, erase):
    print("Initializing Knowledge Base")
    kb_path = "data/processed/mesh_processed"

    if not erase:
        try:
            kb = KnowledgeBase.load(kb_path)
            print("Loaded Knowledge Base\n")
        except Exception as e:
            print(f"Could not load Knowledge Base ({e}). Creating a new one.")
            erase = True

    if erase:
        kb = KnowledgeBase.mop(kb_type, kb_location)
        kb.save(kb_path)
        print("Created New Knowledge Base\n")

    return kb.dataframe

def initialize_labels(kb_dataframe, kb_type, erase):
    print("Initializing Labels")
    labels_path = "data/processed/labels"

    if not erase:
        try:
            kb_labels = KnowledgeBaseLabelsExtraction.load(labels_path)
            print("Loaded Labels\n")
        except Exception as e:
            print(f"Could not load Labels ({e}). Creating new one's.")
            erase = True

    if erase:
        kb_labels = KnowledgeBaseLabelsExtraction.extract_labels(kb_type, kb_dataframe)
        kb_labels.save(labels_path)
        print("Created New Labels\n")

    return kb_labels.labels_dict

def processed_labels_from_preprocessor(labels_dict=None, from_file=True):
    print("Getting Processed Labels from Preprocessor")
    labels_path = "data/processed/labels"

    if from_file:
        processed_labels = Preprocessor.load_labels_from_file(labels_path)
    else:
        processed_labels = Preprocessor.load_labels_from_dict(labels_dict)

    # processed_labels_id = processed_labels[0]
    processed_labels_data = processed_labels[1]

    return processed_labels_data

def embedddings_from_preprocessor(processed_labels, erase):
    print("Getting Embeddings from Preprocessor")
    vectorizer_path = "data/processed/vectorizer"

    if not erase:
        try:
            model = Preprocessor.load(vectorizer_path)
            print(f"Loaded Vectorizer, Type: {model.vectorizer_type}\n")
        except Exception as e:
            print(f"Could not load Vectorizer ({e}). Creating a new one.")
            erase = True
        
    if erase:
        model = TfidfVectorizer.train(processed_labels)
        model.save(vectorizer_path)
        print("Saved Vectorizer\n")

    transformed_labels = model.predict(processed_labels)

    return transformed_labels

def cluster_labels_from_clustering(embeddings, erase):
    print("Getting Clustering Labels From Clustering")
    clustering_path = "data/processed/clustering"

    if not erase:
        try:
            model = Clustering.load(clustering_path)
            print(f"Loaded Clustering Model, Type: {model.model_type}\n")
        except Exception as e:
            print(f"Could not load Clustering Model ({e}). Creating a new one.")
            erase = True

    if erase:
        model = AgglomerativeClusteringCPU.train(embeddings.toarray())
        model.save(clustering_path)
        model.save_labels(clustering_path)
        print("Saved Cluster Labels")

    cluster_labels = model.load_labels(clustering_path)

    return cluster_labels

def metrics_from_trainning(embeddings, clustering_labels):
    print("Trainning")
    TrainCPU.train(embeddings, clustering_labels)