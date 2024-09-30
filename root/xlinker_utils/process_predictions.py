import os

"""
    Process the predictions of the PECOS-EL model into a dictionary.

    Parameters
    ----------
    output : csr_matrix
        The output of the PECOS-EL model.
    annotations : list of lists
        A list of lists containing the annotations for each document.
    entity_type : str
        The type of entity being annotated.
    labels : list of str, optional
        A list of labels for the entities. Default is None.
    label_2_name : dict of str to str, optional
        A dictionary mapping labels to entity names. Default is None.

    Returns
    -------
    doc_out : dict
        A dictionary containing the predictions for each document.
"""

def process_predictions(output, annotations, entity_type, labels=None, label_2_name=None):
    doc_out = {}

    for i in range(len(annotations)):
        pred_index = output[i, :].indices[0]  # -1
        pred_label = labels[pred_index]
        key = str(i)
        doc_out[key] = (pred_label, entity_type)

    return doc_out