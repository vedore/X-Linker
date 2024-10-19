from root.utils.python_utils.process_pecos_preds import process_pecos_preds

from src.python.xlinker.candidates import map_to_kb


"""
    Apply X-Linker pipeline to given input mention. The pipeline includes
    the string matcher and the processing of the output of PECOS-EL model.

    Parameters
    ----------
    input_text : str
        The input text containing the mention.
    annotation : list of lists
        A list containing the annotation information for the mention, including
        doc_id, annot_start, annot_end, annot_text, annot_kb_id.
    mention_preds : csr_matrix
        The predictions of the PECOS-EL model for the mention.
    kb_names : list of str
        A list of knowledge base names.
    kb_synonyms : list of str
        A list of knowledge base synonyms.
    name_2_id : dict
        A dictionary mapping entity names to labels (aka KB identifiers).
    synonym_2_id : dict
        A dictionary mapping entity synonyms to labels (aka KB identifiers).
    index_2_label : dict
        A dictionary mapping indices to labels. Each KB entity has name, 
        label (or identifier) and an index.
    top_k : int
        The number of top-k predictions to return. Default is 1.
    fuzzy_top_k : int
        The number of top-k fuzzy matches to return. Default is 1.
    threshold : float
        The threshold for the prediction score. Default is 0.15.

    Returns
    -------
    output : list
        A list containing the updated annotation information for the
        mention, including doc_id, annot_start, annot_end, annot_text, annot_kb_id,
        labels, and scores.
"""
def apply_pipeline_to_mention(
        input_text,
        annotation,
        mention_preds,
        kb_names,
        kb_synonyms,
        name_2_id,
        synonym_2_id,
        index_2_label,
        top_k=1,
        fuzzy_top_k=1,
        threshold=0.15,
):

    output = []
    annot_text = annotation[3]
    true_label = annotation[4]
    # -----------------------------------------
    #   Get exact match from KB
    # -----------------------------------------
    kb_matches = map_to_kb(
        input_text, kb_names, kb_synonyms, name_2_id, synonym_2_id, top_k=fuzzy_top_k
    )

    # -----------------------------------------------
    # Process X-Linker predictions
    # -----------------------------------------------
    pecos_output = process_pecos_preds(annotation, mention_preds, index_2_label, top_k)
    labels_to_add, scores_to_add = [], []

    if kb_matches[0]["score"] == 1.0:
        labels_to_add.append(kb_matches[0]["kb_id"])
        scores_to_add.append(kb_matches[0]["score"])

        if pecos_output[6][0] == 1.0:
            labels_to_add.append(pecos_output[5][0])
            scores_to_add.append(pecos_output[6][0])

    else:

        if pecos_output[6][0] >= threshold:
            labels_to_add.append(pecos_output[5][0])
            scores_to_add.append(pecos_output[6][0])

        else:

            for i, label in enumerate(pecos_output[5]):
                labels_to_add.append(label)
                scores_to_add.append(pecos_output[6][i])

            for i, match in enumerate(kb_matches):
                labels_to_add.append(match["kb_id"])
                scores_to_add.append(match["score"])

    output = [
        annotation[0],
        annotation[1],
        annotation[2],
        annotation[3],
        annotation[4],
        labels_to_add,
        scores_to_add,
    ]

    return output