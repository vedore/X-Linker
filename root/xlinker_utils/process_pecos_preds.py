"""
    Process the predictions of the PECOS-EL model for a given mention.

    Parameters
    ----------
    annotation : list
        A list containing the annotation information for the mention.
    mention_preds : csr_matrix
        The predictions of the PECOS-EL model for the mention.
    index_2_label : dict
        A dictionary mapping indices to labels.
    top_k : int
        The number of top-k predictions to return.
    inference : bool, optional
        A flag to indicate if the function is being used for inference.
        Default is False.
    label_2_name : dict, optional
        A dictionary mapping labels to entity names. Default is None.
"""
def process_pecos_preds(annotation, mention_preds, index_2_label, top_k, inference=False, label_2_name=None):
    # Add all X-Linker predictions
    pred_labels = []
    pred_scores = []

    for k in range(top_k):
        pred_score = float(mention_preds.data[k])
        pred_scores.append(pred_score)
        pred_index = mention_preds.indices[k]
        pred_label = index_2_label[str(pred_index)]
        pred_labels.append(pred_label)

    if inference:
        output = []
        pred_names = [label_2_name[label] for label in pred_labels]

        for name, label, score in zip(pred_names, pred_labels, pred_scores):
            output.append((name, label, score))

        return output

    else:
        return [
            annotation[0],
            annotation[1],
            annotation[2],
            annotation[3],
            annotation[4],
            pred_labels,
            pred_scores,
        ]