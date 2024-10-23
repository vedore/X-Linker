import argparse
import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train XR-Transformer model")
    """Required true for run_name ?"""
    parser.add_argument("-run_name", type=str, help="Name for the current run")
    parser.add_argument("-ent_type", type=str, default="Disease", help="Entity type to be trained on")
    parser.add_argument("-kb", type=str, default="medic", help="Knowledge base to use")
    parser.add_argument(
        "-model",
        type=str,
        default="bert",
        choices=["bert", "roberta", "biobert", "scibert", "pubmedbert"],
        help="Pre-trained model to use for training"
    )
    parser.add_argument(
        "-clustering", type=str, default="pifa", choices=["pifa", "pifa_lf"], help="Clustering method to use"
    )
    parser.add_argument("-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--only_kb", action="store_true", help="Use only knowledge base data")
    parser.add_argument("--max_inst", type=int, help="Maximum instances to use")
    parser.add_argument("--batch_gen_workers", type=int, help="Number of workers for batch generation")
    return parser.parse_args()

def setup_log_folder(run_name):
    # Define the log file path
    log_file_path = f"log/TRAIN_{run_name}.log"

    # Get the directory part of the path (in this case "log")
    log_dir = Path(log_file_path).parent

    # Check if the directory exists; if not, create it
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{log_dir}' created.")

        # Blank the file by opening it in write mode
    with open(log_file_path, 'w') as log_file:
        # Simply opening it in 'w' mode clears the file content
        print(f"Log file '{log_file_path}' has been blanked.")

def setup_logging(run_name):

    setup_log_folder(run_name)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                filename=f"log/TRAIN_{run_name}.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=2,
            ),
        ],
    )
    logging.info("Logging setup complete.")

def get_training_filepath(args):

    data_dir = f"data/train"

    if args.only_kb:
        return f"{data_dir}/{args.ent_type}/labels.txt"
    if args.ent_type == "Disease":
        return f"{data_dir}/Disease/train_Disease_{args.max_inst}.txt"
    elif args.ent_type == "Chemical":
        return f"{data_dir}/Chemical/train_Chemical.txt"
    else:
        raise ValueError("Invalid entity type specified.")
    
def handle_tfidf_model(X_train, model_dir):
    tf_idf_filepath = f"{model_dir}/tfidf_model"
    
    if os.path.exists(tf_idf_filepath):
        logging.info("Loading TF-IDF model from disk")
        return Preprocessor.load(tf_idf_filepath)
    else:
        logging.info("Training TF-IDF model")
        vectorizer_config = {
            "type": "tfidf",
            "kwargs": {
                "base_vect_configs": [
                    {
                        "ngram_range": [1, 2],
                        "max_df_ratio": 0.98,
                        "analyzer": "word",
                        "buffer_size": 0,
                        "threads": 30,
                    },
                ],
            },
        }
        tfidf_model = Preprocessor.train(X_train, vectorizer_config)
        tfidf_model.save(tf_idf_filepath)
        logging.info("Saved TF-IDF model")
        return tfidf_model

def build_cluster_chain(X_train, X_train_feat, Y_train, clustering_method, model_dir):
    cluster_chain_filepath = f"{model_dir}/cluster_chain_{clustering_method}"
    Z_filepath = None
    if clustering_method == "pifa_lf":
        Z_filepath = f"data/kbs/{args.kb}/Z_{args.kb}_300_dim_20.npz"
    return get_cluster_chain(
        X=X_train,
        X_feat=X_train_feat,
        Y=Y_train,
        method=clustering_method,
        cluster_chain_filepath=cluster_chain_filepath,
        Z_filepath=Z_filepath,
    )

def build_cluster_chain(X_train, X_train_feat, Y_train, clustering_method, model_dir):
    cluster_chain_filepath = f"{model_dir}/cluster_chain_{clustering_method}"
    Z_filepath = None
    if clustering_method == "pifa_lf":
        Z_filepath = f"data/kbs/{args.kb}/Z_{args.kb}_300_dim_20.npz"
    return get_cluster_chain(
        X=X_train,
        X_feat=X_train_feat,
        Y=Y_train,
        method=clustering_method,
        cluster_chain_filepath=cluster_chain_filepath,
        Z_filepath=Z_filepath,
    )

def train_model(train_problem, R_train, clustering_chain, train_params):
    custom_xtf = XTransformer.train(
        train_problem,
        R=R_train,
        clustering=clustering_chain,
        train_params=train_params,
        verbose_level=3,
    )
    return custom_xtf

