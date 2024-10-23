import json


def parse_json(in_filepath):
    """
    Parse a JSON file.
    """
    
    with open(in_filepath, "r", encoding="utf-8") as in_file:
        in_data = json.load(in_file)
        in_file.close()

    return in_data

def add_kb_labels_to_train(kb, out_labels_file):
    """
    Add concept names and synonyms to the training data file, generating a file.

    This function reads concept names and synonyms from the specified knowledge base (KB) 
    and appends them to the output labels file, associating each concept name and synonym 
    with its corresponding index from the KB. 

    Parameters
    ----------
    kb : str
        The identifier of the knowledge base from which to retrieve concept names and synonyms.

    out_labels_file : str
        The path to the output training data file where the concept names and synonyms will be added.

    Returns
    -------
    None
        The function appends the concept names and synonyms to the specified training data file.
    """

    data_dir = f"data/kbs/{kb}"
    name_2_label = parse_json(f"{data_dir}/name_2_label.json")
    label_2_index = parse_json(f"{data_dir}/label_2_index.json")

    output = []

    # Add names to output
    for name, kb_id in name_2_label.items():
        kb_id = str(kb_id)
        index = label_2_index.get(kb_id)
        if index is not None:
            output.append(f"{index}\t{name}")

    # Add synonyms to output
    if kb != "umls":
        synonym_2_label = parse_json(f"{data_dir}/synonym_2_label.json")
        for synonym, kb_id in synonym_2_label.items():
            index = label_2_index.get(kb_id)
            if index is not None:
                output.append(f"{index}\t{synonym}")

    # Write all output to the file at once
    with open(out_labels_file, "a") as out_file:
        out_file.write("\n".join(output) + "\n")

def correct_pub_file(in_filename, out_filename):
    """
    Remove incorrect lines from a Pubtator file and save the corrected version.

    This function reads a Pubtator file and removes lines that do not contain a valid index 
    (the first field must be an integer). The valid lines are written to a new output file, 
    ensuring that the corrected file only contains entries with valid indices.

    Parameters
    ----------
    in_filename : str
        The path to the input Pubtator file that contains incorrect lines.

    out_filename : str
        The path to the output file where the corrected content will be saved.

    Returns
    -------
    None
        The function writes the corrected content to `out_filename`.
    """

    # List to hold valid lines
    output_lines = []

    with open(in_filename, "r", encoding="ISO-8859-1") as in_file:
        for line in in_file:
            line_ = line.strip().split("\t")

            # Check if the first element can be converted to an integer
            try:
                index = int(line_[0])
                output_lines.append(line)  # Collect valid lines
            except ValueError:
                # Log or print the skipped line for debugging if needed
                print(f"Skipping invalid line: {line.strip()}")
                continue

    # Write all valid lines to the output file
    with open(out_filename, "w", encoding="ISO-8859-1") as out_file:
        out_file.writelines(output_lines)

def correct_train_data_encoding(in_path):
    """
    Correct encoding-based errors in a training file and save the corrected version.

    This function reads a training file that may contain encoding errors. It corrects these errors
    by encoding lines in UTF-8 format and ensuring that each line contains exactly two fields 
    separated by a tab. Valid lines are written to a new output file with '_utf8' appended to 
    the original filename.

    Parameters
    ----------
    in_path : str
        The path to the input file that contains encoding errors.

    Returns
    -------
    None
        The function writes the corrected content to a new file with '_utf8' appended to the filename.
    """

    output_lines = []

    with open(in_path, "r", encoding="ISO-8859-1") as in_file:
        for line in in_file:
            # Correct the encoding (non-UTF-8) of the line if necessary
            encoded = line.encode("utf-8", errors="replace")
            decoded = encoded.decode("utf-8", errors="strict").replace("\n", " ")
            splitted = decoded.split("\t")

            # Only append lines that are properly split into two parts
            if len(splitted) == 2:
                output_lines.append(decoded + "\n")

    # Write all valid lines to the output file
    output_filename = f"{in_path.replace('.txt', '_utf8.txt')}"
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.writelines(output_lines)

