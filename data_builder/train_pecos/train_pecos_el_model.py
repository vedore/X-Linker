"""Train the PECOS-EL Disease or Chemical model"""
import argparse
import copy
import json
import os
import logging
import torch

# import wandb
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from data_builder.train_pecos.get_cluster_chain import get_cluster_chain
from logging.handlers import RotatingFileHandler

# wandb.login()

# ------------------------------------------------------------
# Check the available GPUs
# ------------------------------------------------------------
# See https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/6
# torch.cuda.is_available()


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train XR-Transformer model")
parser.add_argument("-run_name", type=str)
parser.add_argument("-ent_type", type=str, default="Disease", help="")
parser.add_argument("-kb", type=str, default="medic", help="")
parser.add_argument(
    "-model",
    type=str,
    default="bert",
    choices=["bert", "roberta", "biobert", "scibert", "pubmedbert"],
    help="",
)
parser.add_argument(
    "-clustering", type=str, default="pifa", choices=["pifa", "pifa_lf"], help=""
)
parser.add_argument("-epochs", type=int, default=10, help="")
parser.add_argument("-batch_size", type=int, default=32, help="")
parser.add_argument("--only_kb", action="store_true", help="")
parser.add_argument("--max_inst", type=int, help="")
parser.add_argument("--batch_gen_workers", type=int, help="")
args = parser.parse_args()

# ------------------------------------------------------------
# Filepaths
# ------------------------------------------------------------
DATA_DIR = "data/train"
KB_DIR = f"data/kbs/{args.kb}"
RUN_NAME = args.run_name

model_dir = f"data/models/trained/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)

# ------------------------------------------------------------
# Configure the logger
# ------------------------------------------------------------
log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # To write to console
        RotatingFileHandler(
            filename=f"log/TRAIN_{RUN_NAME}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
        ),  # To write to a rotating file
    ],
)
logging.info("\n------------------------------------------")

logging.info(f"CUDA is available:{torch.cuda.is_available()}")
logging.info(f"CUDA DEVICE COUNT:{torch.cuda.device_count()}")

# ------------------------------------------------------------
# Import training parameters from train_params.json
# ------------------------------------------------------------
logging.info("Loading training parameters")
params_filepath = f"{model_dir}/train_params.json"

if not os.path.exists(params_filepath):
    logging.info(
        "No train_params.json file found! Copying from data/models/trained/train_params.json"
    )
    os.system(
        f"cp data/models/trained/train_params.json data/models/trained/{RUN_NAME}/train_params.json"
    )

with open(params_filepath, "r", encoding="utf-8") as params_file:
    params = json.load(params_file)
    params_file.close()

# Path to download the corresponding model from the huggingFace repository
model_maps = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "scibert": "allenai/scibert_scivocab_cased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}

model_path = model_maps[args.model]
params["train_params"]["matcher_params_chain"]["model_shortcut"] = model_path
params["train_params"]["matcher_params_chain"]["num_train_epochs"] = args.epochs
params["train_params"]["matcher_params_chain"]["batch_size"] = args.batch_size
params["train_params"]["matcher_params_chain"][
    "batch_gen_workers"
] = args.batch_gen_workers

# Import training parameters from params.json
# Parameters explanation: https://github.com/amzn/pecos/blob/cef885f2014a4e4ca2ebb28350317b0536e9ff45/pecos/xmc/xlinear/model.py#L24
# XR-Transformer: https://github.com/amzn/pecos/tree/cef885f2014a4e4ca2ebb28350317b0536e9ff45/pecos/xmc/xtransformer
train_params = XTransformer.TrainParams.from_dict(params["train_params"])

# ------------------------------------------------------------
# Parse training data
# ------------------------------------------------------------
logging.info("Parsing training data")

if args.only_kb:
    train_filepath = f"{DATA_DIR}/{args.ent_type}/labels.txt"

else:
    if args.ent_type == "Disease":
        train_filepath = f"{DATA_DIR}/Disease/train_Disease_{args.max_inst}.txt"

    elif args.ent_type == "Chemical":
        train_filepath = f"{DATA_DIR}/Chemical/train_Chemical.txt"

parsed_train_data = Preprocessor.load_data_from_file(
    train_filepath, label_text_path=f"{KB_DIR}/labels.txt"
)

logging.info(f"Parse train file: {train_filepath}")

Y_train = parsed_train_data["label_matrix"]
X_train = parsed_train_data["corpus"]

# Use training label frequency scores as costs -> build relevance matrix
R_train = copy.deepcopy(Y_train)

logging.info(
    f"Constructed training corpus len={len(X_train)}, training label matrix with shape={Y_train.shape} and nnz={Y_train.nnz}"
)

# ------------------------------------------------------------
# Feature extraction: build TF-IDF model with training corpus
# ------------------------------------------------------------
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

tf_idf_filepath = f"{model_dir}/tfidf_model"

if os.path.exists(tf_idf_filepath):
    logging.info("Loading TF-IDF model from disk")
    tfidf_model = Preprocessor.load(tf_idf_filepath)

else:
    logging.info("Training TF-IDF model")
    tfidf_model = Preprocessor.train(X_train, vectorizer_config)
    tfidf_model.save(tf_idf_filepath)
    logging.info("Saved TF-IDF model")

X_train_feat = tfidf_model.predict(X_train)
logging.info(
    f"Constructed TRAINING feature matrix with shape={X_train_feat.shape} and nnz={X_train_feat.nnz}"
)

del tfidf_model

# ------------------------------------------------------------
# Construct label hierarchy
# see https://github.com/amzn/pecos/blob/2283da828b9ce6061ca2125ff62d8a37c934550f/pecos/xmc/base.py#L1827
# -----------------------------------------------------------------
logging.info(f"Building cluster chain with method {args.clustering}")

cluster_chain_filepath = f"{model_dir}/cluster_chain_{args.clustering}"

Z_filepath = None

if args.clustering == "pifa_lf":
    Z_filepath = f"data/kbs/{args.kb}/Z_{args.kb}_300_dim_20.npz"

cluster_chain = get_cluster_chain(
    X=X_train,
    X_feat=X_train_feat,
    Y=Y_train,
    method=args.clustering,
    cluster_chain_filepath=cluster_chain_filepath,
    Z_filepath=Z_filepath,
)

# ------------------------------------------------------------
# Train XR-Transformer model
# ------------------------------------------------------------
logging.info("Training model")

# start a new wandb run to track this script
# wandb.init(
# set the wandb project where this run will be logged
#    project=f"x_linker_{args.ent_type}",
#    name=RUN_NAME,
# track hyperparameters and run metadata
#    config=train_params
# )

train_problem = MLProblemWithText(X_train, Y_train, X_feat=X_train_feat)

custom_xtf = XTransformer.train(
    train_problem,
    R=R_train,
    clustering=cluster_chain,
    train_params=train_params,
    verbose_level=3,
)

logging.info("Training completed!")
custom_xtf.save(f"{model_dir}/xtransformer")