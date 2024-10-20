from root.utils.python_utils.parse_json import parse_json
from root.utils.python_utils.kb import KnowledgeBase


"""
Convert the keys of a dictionary to lowercase.
"""
def lower_dict_keys(input_dict):
    return {k.lower(): v for k, v in input_dict.items()}


"""
    Convert a list of strings to lowercase.
"""
def lower_list(input_list):
    return [item.lower() for item in input_list]


"""
    Load the information of a knowledge base from disk (generated mappings 
    in JSON format) to be used in the PECOS-EL model.
"""
def load_kb_info(kb, inference=False):

    """
    Loads knowledge base (KB) information such as labels, mappings, and names from files.

    Args:
        kb (str): The name of the knowledge base to load from.
        inference (bool, optional): If True, loads additional mappings for inference. Defaults to False.

    Returns:
        If `inference` is False:
            tuple: 
                - labels (list): List of labels from the file 'labels.txt'.
                - label_2_name (dict): Dictionary mapping labels to names from 'label_2_name.json'.
                - index_2_label (dict): Dictionary mapping index to label from 'index_2_label.json'.
        
        If `inference` is True:
            tuple: 
                - label_2_name (dict): Dictionary mapping labels to names from 'label_2_name.json'.
                - index_2_label (dict): Dictionary mapping index to label from 'index_2_label.json'.
                - synonym_2_label_lower (dict): Lowercased dictionary mapping synonyms to labels from 'synonym_2_label.json'.
                - name_2_label_lower (dict): Lowercased dictionary mapping names to labels from 'name_2_label.json'.
                - kb_names (list): List of lowercased KB names.
                - kb_synonyms (list): List of lowercased KB synonyms.
    """
        
    data_dir = f"data/kbs/{kb}"

    # Read labels from the labels.txt file
    with open(f"{data_dir}/labels.txt", "r", encoding="utf-8") as fin:
        labels = [line.strip() for line in fin]

    # Load the JSON mappings
    label_2_name = parse_json(f"{data_dir}/label_2_name.json")
    index_2_label = parse_json(f"{data_dir}/index_2_label.json")

    if inference:

        name_2_label = parse_json(f"{data_dir}/name_2_label.json")
        synonym_2_label = parse_json(f"{data_dir}/synonym_2_label.json")

        # Convert dictionary keys to lowercase
        synonym_2_label_lower = {key.lower(): value for key, value in synonym_2_label.items()}
        name_2_label_lower = {key.lower(): value for key, value in name_2_label.items()}

        # Convert keys to lowercase for lists
        kb_names = [name.lower() for name in name_2_label.keys()]
        kb_synonyms = [synonym.lower() for synonym in synonym_2_label.keys()]

        return (
            label_2_name,
            index_2_label,
            synonym_2_label_lower,
            name_2_label_lower,
            kb_names,
            kb_synonyms,
        )

    else:
        return labels, label_2_name, index_2_label


"""
    Load knowledge base object.
"""
def load_kb_object(kb):
    return KnowledgeBase(kb=kb, input_format="tsv")