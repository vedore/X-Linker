def prepare_input(test_annots_raw, abbreviations, id_2_name):
    """
    Prepares the input data for testing by filtering out annotations that correspond
    to NIL IDs or IDs not present in the current knowledge base (KB). It also resolves
    abbreviations using the provided dictionary.

    Args:
        test_annots_raw (list): List of raw annotations, each represented as a tab-separated string.
        abbreviations (dict): Dictionary mapping document IDs to their respective abbreviations.
                              Format: {doc_id: {long_form: abbreviation}}.
        id_2_name (dict): Dictionary mapping KB IDs to names, representing valid KB entities.

    Returns:
        tuple: A tuple containing two lists:
            - test_input (list): List of processed annotation texts (potentially with abbreviations resolved).
            - test_annots (list): List of valid annotations, where each annotation is a list 
                                  [doc_id, start, end, original_text, kb_id].
    """

    removed_annotations = []
    test_input = []
    test_annots = []

    for annot in test_annots_raw:
        # Split the annotation into its components
        components = annot.split("\t")
        doc_id, annot_start, annot_end, annot_text, annot_kb_id = (
            components[0],
            int(components[1]),
            int(components[2]),
            components[3],
            components[5].strip().replace("MESH:", "")
        )

        # Filter out NIL IDs and invalid KB IDs
        if (annot_kb_id == "-1" or annot_kb_id not in id_2_name) and "|" not in annot_kb_id and "," not in annot_kb_id:
            removed_annotations.append(annot)
        else:
            # Resolve abbreviations if available for the document
            input_text = abbreviations.get(doc_id, {}).get(annot_text, annot_text)
            test_input.append(input_text.lower())
            test_annots.append([doc_id, annot_start, annot_end, annot_text, annot_kb_id])

    # Ensure input and annotation lists have matching lengths
    assert len(test_input) == len(test_annots)
    
    # Debugging and progress information
    print(f"{len(test_annots)} test instances loaded!")
    print(f"{len(removed_annotations)} annotations removed!")

    return test_input, test_annots
