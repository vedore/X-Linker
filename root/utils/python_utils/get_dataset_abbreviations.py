import os

from src.python.abrv import run_Ab3P, parse_Ab3P_output


"""
    Get the abbreviations for a given dataset. If the abbreviations have not
    been detected yet, the function runs the Ab3P tool to detect them.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
    -------
    abbreviations : dict
        A dictionary containing the abbreviations for each document.
"""
def get_dataset_abbreviations(dataset):
    dataset_dir = f"data/datasets/{dataset}"
    abbrv_dir = f"{dataset_dir}/abbrv"
    abbrv_filepaths = []

    if os.path.isdir(abbrv_dir):
        abbrv_filepaths = os.listdir(abbrv_dir)

    else:
        os.makedirs(abbrv_dir, exist_ok=True)

    if len(abbrv_filepaths) == 0:
        print("Running Ab3P to detect abbreviations...")
        abbreviations = run_Ab3P(dataset_dir)

    elif len(abbrv_filepaths) > 0:
        print("Parsing abbreviations...")
        abbreviations = parse_Ab3P_output(f"{dataset_dir}/abbrv")

    return abbreviations