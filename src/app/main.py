from src.app.commandhelper import MainCommand
from src.app.utils import cluster_labels_from_clustering, embedddings_from_preprocessor, iniatilize_knowledge_base, initialize_labels, metrics_from_trainning, processed_labels_from_preprocessor


def main():
    args = MainCommand().run()
    kb_type = "medic"
    kb_location = "data/raw/mesh_data/medic/CTD_diseases.tsv"

    kb_dataframe = iniatilize_knowledge_base(kb_type, kb_location, erase=args.erase)

    labels_dict = initialize_labels(kb_dataframe, kb_type, erase=args.erase)

    processed_labels = processed_labels_from_preprocessor(labels_dict, from_file=False)

    processed_labels = processed_labels[1:]

    embeddings = embedddings_from_preprocessor(processed_labels, erase=args.erase)

    cluster_labels = cluster_labels_from_clustering(embeddings, erase=args.erase)

    metrics_from_trainning(embeddings, cluster_labels)

if __name__ == "__main__":
    main()

