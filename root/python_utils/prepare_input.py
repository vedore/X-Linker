"""
    Prepare the input for the Entity Linking model (X-Linker or SapBERT).
    The function filters out the annotations corresponding to NIL ids or IDs
    that are not present in the used version of the target KB and lowercase
    each entity text.

    Parameters
    ----------
    test_annots_raw : list
        A list containing the raw annotations from the evaluation dataset.
        Each element corresponds to an annotation in the string format.
    abbreviations : dict
        A dictionary containing the abbreviations for each document.
    id_2_name : dict
        A dictionary containing the mapping between knowledge base identifiers
        and names.

    Returns
    -------
    test_input : list
        A list containing the input text for the Entity Linking model.
    test_annots : list
        A list containing the annotations to be linked.
"""
def prepare_input(test_annots_raw, abbreviations, id_2_name):
    # Filter out the annotations corresponding to NIL ids or
    # IDs that are not present in the current version of the target KB
    removed_annotations = []
    test_input = []
    test_annots = []

    for annot in test_annots_raw:
        doc_id = annot.split("\t")[0]
        annot_start = int(annot.split("\t")[1])
        annot_end = int(annot.split("\t")[2])
        annot_text = annot.split("\t")[3]
        annot_kb_id = annot.split("\t")[5].strip("\n").replace("MESH:", "")

        if (
                (annot_kb_id == "-1" or annot_kb_id not in id_2_name)
                and "|" not in annot_kb_id
                and "," not in annot_kb_id
        ):
            removed_annotations.append(annot)

        else:
            input_text = annot_text

            if doc_id in abbreviations:

                if annot_text in abbreviations[doc_id]:
                    input_text = abbreviations[doc_id][annot_text]

            test_input.append(input_text.lower())
            test_annots.append(
                [doc_id, annot_start, annot_end, annot_text, annot_kb_id]
            )

    assert len(test_input) == len(test_annots)
    print(f"{len(test_annots)} test instances loaded!")
    print(f"{len(removed_annotations)} annotations removed!")

    return test_input, test_annots