from src.python.utils import parse_json
from src.python.kbs import KnowledgeBase


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
    data_dir = f"data/kbs/{kb}"

    with open(f"{data_dir}/labels.txt", "r", encoding="utf-8") as fin:
        labels = [ll.strip() for ll in fin.readlines()]
        fin.close()

    # Open mappings label to name
    label_2_name = parse_json(f"{data_dir}/label_2_name.json")
    index_2_label = parse_json(f"{data_dir}/index_2_label.json")

    if inference:
        name_2_label = parse_json(f"{data_dir}/name_2_label.json")
        synonym_2_label = parse_json(f"{data_dir}/synonym_2_label.json")
        kb_names = lower_list(name_2_label.keys())
        kb_synonyms = lower_list(synonym_2_label.keys())
        name_2_label_lower = lower_dict_keys(name_2_label)
        synonym_2_label_lower = lower_dict_keys(synonym_2_label)

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