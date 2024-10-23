from src.utils.python_utils.load_kb_info import load_kb_info
from src.utils.python_utils.load_model import load_model
from src.utils.python_utils.process_pecos_preds import process_pecos_preds

from tqdm import tqdm


def inference_load_model(model_dir, clustering):
    custom_xtf, tfidf_model, cluster_chain = load_model(model_dir, clustering)
    return custom_xtf, tfidf_model, cluster_chain

def inference_kb_info(kb, inference):
    id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = load_kb_info(kb, inference)
    return id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms

def inference_apply_model_to_input_entities(input_entities, custom_xtf, tfidf_model, index_2_id, id_2_name, top_k):
    input_entities_processed = [entity.lower() for entity in input_entities]

    x_linker_preds = custom_xtf.predict(
        input_entities_processed, X_feat=tfidf_model.predict(input_entities_processed), only_topk=top_k
    )

    pbar = tqdm(total=len(input_entities_processed))

    output = ''

    for i, entity in enumerate(input_entities_processed):
        mention_preds = x_linker_preds[i, :]
        mention_output = process_pecos_preds(
            entity, mention_preds, index_2_id, top_k, inference=True,
            label_2_name=id_2_name
        )

        output += f'Entity: {entity}\nOutput: {mention_output}\n--------\n'
        pbar.update(1)

    pbar.close()

    print(output)