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

    abbreviations = {}
    abbrvs_filepaths = os.listdir(abbrv_dir)

    for filepath in abbrvs_filepaths:
        # Extract document ID (remove extensions like .txt or .ann)
        doc_id = os.path.splitext(filepath)[0]
        doc_abbrvs = {}

        filepath_up = os.path.join(abbrv_dir, filepath)

        with open(filepath_up, "r") as out_file:
            lines = out_file.readlines()

            for line in lines:
                # Only process lines that start with a space (valid Ab3P lines)
                if line.startswith(" "):
                    line_data = line.split("|")
                    
                    # Ensure the line contains the expected abbreviation data
                    if len(line_data) == 3:
                        score = float(line_data[2])

                        # Only consider abbreviations with a confidence score >= 0.90
                        if score >= 0.90:
                            long_form = line_data[0].strip()
                            abbreviation = line_data[1].strip()
                            doc_abbrvs[long_form] = abbreviation

        # Add the document's abbreviations to the main dictionary
        abbreviations[doc_id] = doc_abbrvs

    return abbreviations


def run_Ab3P(dataset_dir):
    """
    Applies the Ab3P abbreviation detection tool to the text documents in the specified directory.

    It processes all text files located in the `dataset_dir` and stores the output in 
    an abbreviation directory, then parses and returns the abbreviations found.

    Args:
        dataset_dir (str): Path to the directory containing the document texts where 
                           entities were recognized.

    Returns:
        dict: A dictionary containing the abbreviations in the format:
              {doc_id: {long_form: abbreviation}}.
    """
    
    # Define paths to the text directory and abbreviation output directory
    txt_dir = os.path.abspath(os.path.join(dataset_dir, "txt"))
    abbrv_dir = os.path.abspath(os.path.join(dataset_dir, "abbrv"))
    os.makedirs(abbrv_dir, exist_ok=True)

    # Save the current working directory and change to Ab3P's directory
    cwd = os.getcwd()
    ab3p_dir = os.path.join("abbreviation_detector", "Ab3P")
    os.chdir(ab3p_dir)
    
    os.makedirs("tmp", exist_ok=True)

    # Process each text file with Ab3P
    txt_filepaths = os.listdir(txt_dir)

    for filepath in txt_filepaths:
        doc_id = os.path.splitext(filepath)[0]  # Remove file extension to get doc_id
        txt_filepath = os.path.join(txt_dir, filepath)
        abbrv_filepath = os.path.join(abbrv_dir, doc_id)

        # Run Ab3P command and suppress stderr output
        command = f"./identify_abbr {txt_filepath} 2> /dev/null >> {abbrv_filepath}"
        os.system(command)

    # Restore the original working directory
    os.chdir(cwd)

    # Parse the Ab3P output and return the abbreviation mappings
    return parse_Ab3P_output(abbrv_dir)