import os

import pandas as pd

from root.python_utils.calculate_topk_accuracy import calculate_topk_accuracy
from root.python_utils.get_dataset_abbreviations import get_dataset_abbreviations
from root.python_utils.prepare_input import prepare_input
from root.xlinker_utils.apply_pipeline_to_mention import apply_pipeline_to_mention
from root.xlinker_utils.load_kb_info import load_kb_info
from root.xlinker_utils.load_model import load_model
from root.xlinker_utils.process_pecos_preds import process_pecos_preds

from tqdm import tqdm


def model(model_dir, clustering):
    custom_xtf, tfidf_model, cluster_chain = load_model(model_dir, clustering)
    return custom_xtf, tfidf_model, cluster_chain

def kb_info(kb, inference):
    id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = load_kb_info(kb, inference)
    return id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms

def dataset_abbreviations(dataset, abbrv):
    abbreviations = {}
    if abbrv:
        abbreviations = get_dataset_abbreviations(dataset)
    return abbreviations

def load_tests(dataset, ent_type, unseen):
    if unseen:
        test_path = f"data/datasets/{dataset}/test_{ent_type}_unseen.txt"
    else:
        test_path = f"data/datasets/{dataset}/test_{ent_type}.txt"

    with open(test_path, "r") as f:
        test_annots_raw = f.readlines()

    return test_annots_raw

def prep_input(test_annots_raw, abbreviations, id_2_name):
    test_input, test_annots = prepare_input(test_annots_raw, abbreviations, id_2_name)
    return test_input, test_annots

def apply_model_to_test_instances(custom_xtf, test_input, tfidf_model,
                                  test_annots, kb_names, kb_synonyms,
                                  name_2_id_lower, synonym_2_id_lower, index_2_id,
                                  pipeline, top_k, fuzzy_top_k, threshold):
    x_linker_preds = custom_xtf.predict(
        test_input, X_feat=tfidf_model.predict(test_input), only_topk=top_k
    )
    print("Linking test instances...")

    output = []
    pbar = tqdm(total=len(test_annots))

    for i, annotation in enumerate(test_annots):
        mention_preds = x_linker_preds[i, :]

        if pipeline:
            # Apply pipeline to every mention in test set
            mention_output = apply_pipeline_to_mention(
                test_input[i],
                annotation,
                mention_preds,
                kb_names,
                kb_synonyms,
                name_2_id_lower,
                synonym_2_id_lower,
                index_2_id,
                top_k=top_k,
                fuzzy_top_k=fuzzy_top_k,
                threshold=threshold,
            )

        else:
            # Just consider the X-linker predictions
            mention_output = process_pecos_preds(
                annotation, mention_preds, index_2_id, top_k
            )

        output.append(mention_output)
        pbar.update(1)

    pbar.close()

    return output

def evaluation(output, dataset, ent_type, kb, fuzzy_top_k, ppr):

    predictions_df = pd.DataFrame(
        output, columns=["doc_id", "start", "end", "text", "code", "codes", "scores"]
    )

    if ppr:
        # Prepare input for PPR
        run_name = f"{dataset}_{ent_type}_{kb}"
        os.makedirs(f"data/REEL/{run_name}", exist_ok=True)
        pred_path = f"data/REEL/{run_name}/xlinker_preds.tsv"
        predictions_df.to_csv(pred_path, sep="\t", index=False)
        ppr.prepare_ppr_input(
            run_name,
            predictions_df,
            ent_type,
            fuzzy_top_k=fuzzy_top_k,
            kb=kb,
        )

        # Build the disambiguation graph, run PPR and process the results
        ppr.run(entity_type=ent_type, kb=kb, reel_dir=f"data/REEL/{run_name}")

    else:
        # Evaluate model performance
        pred_path = f"data/evaluation_{dataset}_{ent_type}.tsv"
        predictions_df.to_csv(pred_path, sep="\t", index=False)
        topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 5, 10, 15, 20, 25])
        print(f"Top-k accuracies: {topk_accuracies}")
        topK_list = [list(topk_accuracies.values())]
        df = pd.DataFrame(topK_list)
        df.to_csv("data.tsv", sep="\t", index=False)