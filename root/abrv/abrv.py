import os


def parse_Ab3P_output(abbrv_dir):
    """
    Parse the output of the Ab3P tool from a text file into a dictionary
    for later reuse.

    Parameters
    ----------
    abbrv_dir : str
        The directory containing the files that were output by Ab3P.

    Returns
    -------
    dict
        A dictionary containing the abbreviations with the format:
        {doc_id: {long_form: abbreviation}}.
    """

    abbrvs_filepaths = os.listdir(abbrv_dir)
    abbreviations = {}
    doc_id = ""

    for filepath in abbrvs_filepaths:
        doc_abbrvs = {}
        doc_id = filepath

        if ".txt" in filepath or ".ann" in filepath:
            doc_id = filepath[:-4]

        filepath_up = f"{abbrv_dir}/{doc_id}" 

        with open(filepath_up, "r") as out_file:
            data = out_file.readlines()
            out_file.close()

            for line in data:

                if line[0] == " ":
                    line_data = line.split("|")

                    if len(line_data) == 3:
                        score = float(line_data[2])

                        if score >= 0.90:
                            doc_abbrvs[line_data[0].strip(" ")] = line_data[1]

        abbreviations[doc_id] = doc_abbrvs

    return abbreviations


def run_Ab3P(dataset_dir):
    """
    Apply the abbreviation detector Ab3P to the texts located in `dataset_dir`.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory containing the texts of the documents where the 
        entities were recognized.

    Returns
    -------
    dict
        A dictionary containing the abbreviations with the format:
        {doc_id: {long_form: abbreviation}}.
    """

    txt_dir = f"../../{dataset_dir}/txt"
    abbrv_dir = f"../../{dataset_dir}/abbrv"
    os.makedirs(abbrv_dir, exist_ok=True)

    # change to Ab3P directory
    cwd = os.getcwd()
    os.chdir("abbreviation_detector/Ab3P/")
    os.makedirs("tmp/", exist_ok=True)

    # Run Ab3P for each text file
    txt_filepaths = os.listdir(txt_dir)

    for filepath in txt_filepaths:
        doc_id = filepath

        if ".txt" in filepath:
            doc_id = filepath[:-4]

        comm = (
            f"./identify_abbr {txt_dir}/{filepath} 2> /dev/null >> {abbrv_dir}/{doc_id}"
        )
        os.system(comm)

    # Return to the original dir
    os.chdir(cwd)

    return parse_Ab3P_output(abbrv_dir)
