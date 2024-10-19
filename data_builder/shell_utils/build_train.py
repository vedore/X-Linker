import json


def parse_json(in_filepath):
    """
    Parse a JSON file.
    """

    with open(in_filepath, "r", encoding="utf-8") as in_file:
        in_data = json.load(in_file)
        in_file.close()

    return in_data

def add_kb_labels_to_train(kb, train_filename):
    """
    Add concept names and synonyms to the training data file, generating the
    file `train_plus_labels.txt`.

    Parameters
    ----------
    kb : object
        The knowledge base object containing concept names and synonyms.
    train_filename : str
        The path to the training data file.

    Returns
    -------
    None
        Function creates a new file `train_plus_labels.txt` that includes the 
        original training data along with the added concept names and synonyms.
    """


    data_dir = f"data/kbs/{kb}"
    name_2_label = parse_json(f"{data_dir}/name_2_label.json")

    synonym_2_label = {}

    if kb != "umls":
        synonym_2_label = parse_json(f"{data_dir}/synonym_2_label.json")

    label_2_index = parse_json(f"{data_dir}/label_2_index.json")

    output = ""

    for name, kb_id in name_2_label.items():

        if isinstance(kb_id, str):
            if "MESH:" in kb_id:
                kb_id = kb_id.strip("MESH:")
        else:
            kb_id = str(kb_id)

        if kb_id in label_2_index.keys():
            index = label_2_index[kb_id]
            output += f"{index}\t{name}\n"

    for synonym, kb_id in synonym_2_label.items():

        if isinstance(kb_id, str):
            if "MESH:" in kb_id:
                kb_id = kb_id.strip("MESH:")

        if kb_id in label_2_index.keys():
            index = label_2_index[kb_id]
            output += f"{index}\t{synonym}\n"

    with open(train_filename, "a") as out_file:
        out_file.write(output)
        out_file.close()


def correct_pub_file(in_filename, out_filename):
    """
    Remove incorrect lines from a Pubtator file and save the corrected version.

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

    output = ""

    with open(in_filename, "r", encoding="ISO-8859-1") as in_file:
        data = in_file.readlines()

        for line in data:
            line_ = line.strip("\n").split("\t")

            try:
                index = int(line_[0])
                output += line

            except ValueError:
                continue

    with open(out_filename, "w") as out_file:
        out_file.write(output)
        out_file.close()


def correct_train_data_encoding(in_path):
    """
    Correct encoding-based errors in a training file and save the corrected 
    version.

    Parameters
    ----------
    in_path : str
        The path to the input file that contains encoding errors.

    Returns
    -------
    None
        The function writes the corrected content to `out_filename`.
    """

    output = ""

    with open(in_path, "r", encoding="ISO-8859-1") as in_file:
        data = in_file.readlines()

        for line in data:
            # Correct the encoding (non-UTF-8) of the lines if necessary
            encoded = line.encode("utf-8", errors="replace")
            decoded = encoded.decode("utf-8", errors="strict")
            decoded = decoded.replace("\n", " ")
            # Correct lines that include only the text and miss the labels
            splitted = decoded.split("\t")

            if len(splitted) == 2:
                output += f"{decoded}\n"

        in_file.close()

    with open(
        f"{in_path.replace('.txt', '_utf8.txt')}", "w", encoding="utf-8"
    ) as out_file:
        out_file.write(output)
        out_file.close()