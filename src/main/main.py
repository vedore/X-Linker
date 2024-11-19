import torch

from src.extractor.knowledge_base import KnowledgeBase, KnowledgeBaseLabelsExtraction
from src.featurization.preprocessor import Preprocessor
from src.machine_learning.cpu.ml import AgglomerativeClustering as CPUAC, ClusteringCPU
from src.machine_learning.gpu.ml import AgglomerativeClustering as GPUAC, ClusteringGPU


def clean_kb():
    try:
        kb = KnowledgeBase.load("data/processed/mesh_processed")
    except:
        kb = KnowledgeBase.mop('medic', 'data/raw/mesh_data/medic/CTD_diseases.tsv')
        kb.save("data/processed/mesh_processeded")
    
    return kb.dataframe

def create_labels(dataframe):
    try:
        kb_labels = KnowledgeBaseLabelsExtraction.load("data/processed/labels")
    except:
        kb_labels = KnowledgeBaseLabelsExtraction.extract_labels('medic', dataframe)
        kb_labels.save("data/processed/labels")
    
def get_labels_to_preprocessor():
    processed_labels = Preprocessor.load_data_from_file("data/processed/labels")

    processed_labels_id = processed_labels[0]
    processed_labels_data = processed_labels[1]

    return processed_labels_data

def preprocessor(processed_labels_data):
    try:
        preprocessor = Preprocessor.load("data/processed/preprocessor")
    except: 
        preprocessor = Preprocessor.train(processed_labels_data)
        preprocessor.save("data/processed/preprocessor")

    transformed_labels = preprocessor.predict(processed_labels_data)

    return transformed_labels

def clustering(transformed_labels):

    if torch.cuda.is_available():
        try:
            model = ClusteringGPU.load("data/processed/clustering")
        except:
            model = GPUAC.train(transformed_labels)
            model.save("data/processed/clustering")
    else:
        try:
            model = ClusteringCPU.load("data/processed/clustering")
        except:
            model = CPUAC.train(transformed_labels)
            model.save("data/processed/clustering")
    
    print(model.get_labels())
    

dataframe = clean_kb()

create_labels(dataframe)

processed_labels = get_labels_to_preprocessor()

transformed_labels = preprocessor(processed_labels)

print(transformed_labels)