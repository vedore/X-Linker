import time
import torch
import os
import pandas as pd
from scipy.sparse import csr_matrix

from src.extractor.knowledge_base import KnowledgeBase, KnowledgeBaseLabelsExtraction
from src.featurization.preprocessor import Preprocessor
from src.machine_learning.gpu.ml import AgglomerativeClusteringGPU
from src.machine_learning.cpu.ml import AgglomerativeClusteringCPU
from src.featurization.vectorizer import DistilBertVectorizer, TfidfVectorizer
from src.machine_learning.clustering import Clustering
from src.trainning.gpu.train import TrainGPU


def clean_kb():
    try:
        kb = KnowledgeBase.load("data/processed/mesh_processed")
        print("Loaded KB")
    except:
        kb = KnowledgeBase.mop('medic', 'data/raw/mesh_data/medic/CTD_diseases.tsv')
        kb.save("data/processed/mesh_processed")
        print("Saved KB")

    # kb = KnowledgeBase.mop('medic', 'data/raw/mesh_data/medic/CTD_diseases.tsv')
    # kb.save("data/processed/mesh_processed")
    # print("Saved KB")
    
    return kb.dataframe

def create_labels(dataframe):
    try:
        kb_labels = KnowledgeBaseLabelsExtraction.load("data/processed/labels")
        print("Loaded Labels")
    except:
        kb_labels = KnowledgeBaseLabelsExtraction.extract_labels('medic', dataframe)
        kb_labels.save("data/processed/labels")
        print("Saved Labels")

    # kb_labels = KnowledgeBaseLabelsExtraction.extract_labels_version_2('medic', dataframe)
    # kb_labels.save("data/processed/labels")
    # print("Saved Labels")

def get_labels_to_preprocessor():
    processed_labels = Preprocessor.load_labels_from_file("data/processed/labels")

    processed_labels_id = processed_labels[0]
    processed_labels_data = processed_labels[1]

    return processed_labels_data

def preprocessor(processed_labels_data):
    """
    try:
        tfdif = Preprocessor.load("data/processed/vectorizer")
        print("Loaded Vectorizer")
    except: 
        tfdif = TfidfVectorizer.train(processed_labels_data)
        tfdif.save("data/processed/vectorizer")
        print("Saved Vectorizer")
    """
    
    tfdif = TfidfVectorizer.train(processed_labels_data)
    tfdif.save("data/processed/vectorizer")
    print("Saved Vectorizer")

    transformed_labels = tfdif.predict(processed_labels_data)

    return transformed_labels

    # return DistilBertVectorizer.predict(processed_labels_data)

def clustering(transformed_labels):
    """
    if torch.cuda.is_available():
        try:
            model = AgglomerativeClusteringGPU.load("data/processed/clustering")
        except:
            model = AgglomerativeClusteringGPU.train(transformed_labels)
            model.save("data/processed/clustering")
            model.save_labels("data/processed/clustering")
    else:
        try:
            model = AgglomerativeClusteringCPU.load("data/processed/clustering")
        except:
            model = AgglomerativeClusteringCPU.train(transformed_labels)
            model.save("data/processed/clustering")
            model.save_labels("data/processed/clustering")
    """
    
    # Suppose to be GPU
    # model = AgglomerativeClusteringCPU.train(transformed_labels)
    # model.save("data/processed/clustering")
    # model.save_labels("data/processed/clustering")

    model = AgglomerativeClusteringCPU.load("data/processed/clustering")
    labels = model.get_labels()

    # print(labels.to_numpy())

    clustering_df = pd.DataFrame(
        {
            'Labels': labels
        }
    )

    return clustering_df

    # return model.load_labels("data/processed/clustering")

def load_clustering_labels(cluster_labels_folder):
    return Clustering.load_labels(cluster_labels_folder)

def trainning(embeddings, clustering_labels):
    TrainGPU.train(embeddings, clustering_labels)

    
dataframe = clean_kb()

# create_labels(dataframe)

processed_labels = get_labels_to_preprocessor()[1:]

embeddings = preprocessor(processed_labels)

# tfidf
embeddings = embeddings.toarray() 

# DistilBert
# embeddings = embeddings.numpy()

# print(type(embeddings))

cluster_labels = clustering(embeddings)

# cluster_labels = load_clustering_labels("data/processed/clustering")

# print(cluster_labels)

trainning(embeddings, cluster_labels)

