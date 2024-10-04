import os
import numpy as np
import pandas as pd

from root.utils.calculate_topk_accuracy import calculate_topk_accuracy
from root.utils.get_dataset_abbreviations import get_dataset_abbreviations
from root.utils.prepare_input import prepare_input
from src.python.utils import parse_json

from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
from tqdm import tqdm

def sapbert_kb_info(kb):
    name_2_id = parse_json(f"data/kbs/{kb}/name_2_label.json")
    label_2_name = parse_json(f"data/kbs/{kb}/label_2_name.json")
    id_2_synonym = parse_json(f"data/kbs/{kb}/label_2_synonym.json")

    kb_pairs = []

    for node_name, node_id in tqdm(name_2_id.items()):
        kb_pairs.append((node_name.lower(), node_id))

        syns = id_2_synonym.get(node_id, [])

        for syn in syns:
            kb_pairs.append((syn.lower(), node_id))

    print("Number of KB pairs:", len(kb_pairs))
    all_names = [p[0] for p in kb_pairs]
    all_ids = [p[1] for p in kb_pairs]

    return all_names, all_ids, kb_pairs, label_2_name

def sapbert_load_setup_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    model = AutoModel.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )  # .cuda(1)

    return model, tokenizer

def sapbert_enconde_kb_labels(model, tokenizer, all_names, ent_type):

    kb_embeds_path = f"data/sapbert/all_reps_emb_{ent_type}_to_keep.npy"

    if os.path.exists(kb_embeds_path):
        all_reps_emb = np.load(kb_embeds_path)
        print("Loaded embeddings from file")

    else:
        print("Encoding KB labels...")
        bs = 128
        all_reps = []
        for i in tqdm(np.arange(0, len(all_names), bs)):
            toks = tokenizer.batch_encode_plus(
                all_names[i : i + bs],
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt",
            )
            # toks_cuda = {}
            # for k,v in toks.items():
            #    toks_cuda[k] = v.cuda(1)
            # output = model(**toks_cuda)

            output = model(**toks)
            cls_rep = output[0][:, 0, :]

            all_reps.append(cls_rep.cpu().detach().numpy())

            all_reps_emb = np.concatenate(all_reps, axis=0)
            np.save(kb_embeds_path, all_reps_emb)

    return all_reps_emb

def sapbert_dataset_abbreviations(abbrv, dataset):
    abbreviations = {}
    if abbrv:
        abbreviations = get_dataset_abbreviations(dataset)
    return abbreviations

def sapbert_load_tests(abbreviations, label_2_name, dataset, ent_type):
    with open(f"data/datasets/{dataset}/test_{ent_type}.txt", "r") as f:
        test_annots_raw = f.readlines()
    f.close()

    test_input, test_annots = prepare_input(test_annots_raw, abbreviations, label_2_name)
    return test_input, test_annots

def sapbert_apply_model_to_test_instances(model, tokenizer, test_input, test_annots, all_reps_emb, kb_pairs, top_k):

    predictions = []
    pbar = tqdm(total=len(test_input))

    for i, mention in enumerate(test_input):
        pred_labels = []

        # Encode query
        query_toks = tokenizer.batch_encode_plus(
            [mention],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )

        query_output = model(**query_toks)
        query_cls_rep = query_output[0][:, 0, :]

        # Find nearest neighbour
        dist = cdist(query_cls_rep.detach().numpy(), all_reps_emb)

        # nn_index = np.argmin(dist)
        # Find indices of top-k nearest neighbors
        k = top_k
        top_k_indices = np.argsort(dist, axis=1)[:, :k][0]

        scores = []

        for index in top_k_indices:
            pred_label = kb_pairs[index][1]
            pred_labels.append(pred_label)
            scores.append(dist[0][index])

        true_label = test_annots[i][4]
        predictions.append(
            [
                test_annots[i][0],
                test_annots[i][1],
                test_annots[i][2],
                test_annots[i][3],
                true_label,
                pred_labels,
                scores,
            ]
        )
        pbar.update(1)

    pbar.close()

    return predictions

def sapbert_evaluation(predictions):
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(
        predictions, columns=["doc_id", "start", "end", "text", "code", "codes", "scores"]
    )

    # Evaluate model performance
    topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 5, 10, 25, 50])
    print(f"Top-k accuracies: {topk_accuracies}")

