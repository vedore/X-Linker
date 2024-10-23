"""Train the PECOS-EL Disease or Chemical model"""
import copy
import json
import os
import logging
import torch

from data_builder.shell_utils.train_pecos.train_pecos_controller import get_training_filepath, parse_arguments, setup_logging
from data_builder.shell_utils.train_pecos.get_cluster_chain import get_cluster_chain

# import wandb
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from logging.handlers import RotatingFileHandler

# wandb.login()

DATA_DIR = "data/train"

def main():

    args = parse_arguments()
    setup_logging(args.run_name)

    logging.info(f"CUDA is available:{torch.cuda.is_available()}")
    logging.info(f"CUDA Device Count:{torch.cuda.device_count()}")

    kb_dir = f"data/kbs/{args.kb}"
    run_name = args.run_name
    model_dir = f"data/models/trained/{run_name}"   
    os.makedirs(model_dir, exist_ok=True)

    # Set up training data
    train_filepath = get_training_filepath(args)
    parsed_train_data = Preprocessor.load_data_from_file(
        train_filepath, label_text_path=f"{kb_dir}/labels.txt"
    )
    
    # Extract Y_train and X_train
    Y_train = parsed_train_data["label_matrix"]
    X_train = parsed_train_data["corpus"]
    R_train = copy.deepcopy(Y_train)

    # Feature extraction
    tfidf_model = handle_tfidf_model(X_train, model_dir)
    X_train_feat = tfidf_model.predict(X_train)

    # Build cluster chain
    cluster_chain = build_cluster_chain(X_train, X_train_feat, Y_train, args.clustering, model_dir)

    # Train model
    train_problem = MLProblemWithText(X_train, Y_train, X_feat=X_train_feat)
    custom_xtf = train_model(train_problem, R_train, cluster_chain, train_params)

    logging.info("Training completed!")
    custom_xtf.save(f"{model_dir}/xtransformer")

if __name__ == "__main__":
    main()