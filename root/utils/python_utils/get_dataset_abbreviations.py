import os

from abrv.abrv import run_Ab3P, parse_Ab3P_output


def get_dataset_abbreviations(dataset):
    """
    Retrieves or generates abbreviations for a dataset.

    If abbreviation files exist in the dataset directory, they are parsed. 
    Otherwise, it runs Ab3P to detect abbreviations and stores them.

    Args:
        dataset (str): The name of the dataset for which to retrieve abbreviations.

    Returns:
        list: A list of abbreviations found in the dataset.
    """

    dataset_dir = f"data/datasets/{dataset}"
    abbrv_dir = os.path.join(dataset_dir, "abbrv")

    # Check if the abbreviation directory exists and get file paths if available
    if not os.path.isdir(abbrv_dir):
        os.makedirs(abbrv_dir, exist_ok=True)
        abbrv_filepaths = []
    else:
        abbrv_filepaths = os.listdir(abbrv_dir)

     # Detect abbreviations if no files exist, otherwise parse existing files
    if not abbrv_filepaths:
        print("Running Ab3P to detect abbreviations...")
        abbreviations = run_Ab3P(dataset_dir)
    else:
        print("Parsing abbreviations...")
        abbreviations = parse_Ab3P_output(abbrv_dir)

    return abbreviations