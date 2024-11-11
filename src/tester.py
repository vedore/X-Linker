import os
import pandas as pd

from src.embeddings.embeddings import Embeddings 
from src.entity_linking.trainning_model import TrainningModel
from src.preprocessing.kb import Kb

kb_mappings = {
        "medic": {
            "filepath": "CTD_diseases.tsv",
            "id_column": "DiseaseID",
            "name_column": "DiseaseName",
            "skiprows": 29
        },
        "ctd_chemicals": {
            "filepath": "CTD_chemicals.tsv",
            "id_column": "ChemicalID",
            "name_column": "ChemicalName",
            "skiprows": 29
        },
        "ctd_genes": {
            "filepath": "CTD_genes.tsv",
            "id_column": "GeneID",
            "name_column": "GeneSymbol",
            "skiprows": 29
        },
        "ncbi_taxon": {
            "filepath": "ncbi_taxon.tsv",
            "id_column": "TaxonID",
            "name_column": "TaxonName",
            "skiprows": 1
        }
    }

kb_filepath = os.path.abspath("data/raw/mesh_data/medic/CTD_diseases.tsv")
kb_type = 'medic'
id_column = "DiseaseID"
delimiter = '\t'
skip_rows = 29

processed_mesh_folder = "data/processed/mesh_processed"

processed_index_labels = "data/processed/index_labels"

processed_embeddings = "data/processed/embeddings"

processed_clustering = "data/processed/clustering"


# kb = Kb().load(kb_type)
# kb = Kb().clean_dataframe(kb_filepath, kb_type, id_column, delimiter, skip_rows)
# kb.create_labels()

embeddings = Embeddings().load_labels(kb_type, processed_index_labels)
embeddings.use_gpu = False
# embeddings.prepare_data()
# embeddings.create_embeddings()
embeddings.load_embeddings()
# embeddings.create_clustering()
embeddings.load_clustering()

emb = embeddings.embeddings
clustering_df = embeddings.clustering_df

tm = TrainningModel()
tm.prepare_data(clustering_df, emb)
tm.train_model()



df = clustering_df.groupby('ClusterLabel')['EntityID'].apply(list)

print(df)


# embeddings.prepare_data()
# embeddings.create_embeddings()
# embeddings.clustering()
# embeddings.save_processed(kb_type)

# embeddings = Embeddings().load_processed(kb_type, processed_embeddings)
# cluster_dict = embeddings.clustering()
# print(cluster_dict)



