import os
import pandas as pd

from src.featurization.preprocessor import Preprocessor
from src.extractor.knowledge_base import KnowledgeBase, KnowledgeBaseLabelsExtraction
from src.machine_learning.cpu.ml import AgglomerativeClustering as CPUAC
from src.machine_learning.gpu.ml import AgglomerativeClustering as GPUAC



"""
    Clean the Database
"""
# kb = KnowledgeBase.mop('medic', kb_filepath)

"""
    Save The Database
"""
# kb.save(processed_mesh_folder)

"""
    Load an Dataframe 
"""
kb = KnowledgeBase.load("data/processed/mesh_processed")

"""
    Print the Dataframe
"""
# print(kb.dataframe)

"""
    Create Labels
"""
# kb_labels = KnowledgeBaseLabelsExtraction.extract_labels('medic', kb.dataframe)

"""
    Save Labels
"""
# kb_labels.save(processed_labels)

"""
    Load the Labels
"""
kb_labels = KnowledgeBaseLabelsExtraction.load("data/processed/labels")

"""
    Print the Labels
"""
# print(kb_labels.labels_dict)

"""
    Get Processed Labels to fit in Preprocessor
"""
processed_labels = Preprocessor.load_data_from_file("data/processed/labels")

processed_labels_id = processed_labels[0]
processed_labels_data = processed_labels[1]

"""
    Fit Corpus in Preprocessor
"""
# preprocessor = Preprocessor.train(processed_labels_data)

"""
    Save Preprocessor
"""
# preprocessor.save("data/processed/preprocessor")

"""
    Load Preprocessor
"""
preprocessor = Preprocessor.load("data/processed/preprocessor")

"""
    Transform the Corpus to Embeddings
"""
transformed_labels = preprocessor.predict(processed_labels_data)

"""
    Print Embeddings
"""
# print(transformed_labels)

"""
    Hieararchical Clustering Using CPU
"""
# clustering_model = CPUAC.train(transformed_labels)

"""
    Hierarchical Clustering Using GPU
"""
clustering_model = GPUAC.train(transformed_labels)

"""
    Save Model
"""
clustering_model.save("data/processed/clustering")

"""
    Print The Labels
"""
print(clustering_model.get_labels())





# embeddings = Preprocessing.load_labels(kb_type, processed_index_labels)
# embeddings.use_gpu = False
# embeddings.prepare_data()
# embeddings.create_embeddings()
# embeddings.load_embeddings()
# embeddings.create_clustering()
# embeddings.load_clustering()

# emb = embeddings.embeddings
# clustering_df = embeddings.clustering_df

# tm = TrainningModel()
# tm.prepare_data(clustering_df, emb)
# tm.train_model()

# df = clustering_df.groupby('ClusterLabel')['EntityID'].apply(list)

# print(df)


# embeddings.prepare_data()
# embeddings.create_embeddings()
# embeddings.clustering()
# embeddings.save_processed(kb_type)

# embeddings = Embeddings().load_processed(kb_type, processed_embeddings)
# cluster_dict = embeddings.clustering()
# print(cluster_dict)



