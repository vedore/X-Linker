def parse_pubtator_file(dataset, filename, ent_types, parsed_data):
    """
    Parse a Pubtator file and returns a dictionary with the text and
    annotations.
    """

    with open(filename, "r", encoding="utf-8") as pubtator_file:
        data = pubtator_file.readlines()
        pubtator_file.close()
        text = ""

        for line in data:

            if "|t|" in line:
                text += line + "\n"
                doc_id = line.split("|t|")[0]

            elif "|a|" in line:
                line_data = line.split("|a|")
                text += line + "\n"
                parsed_data[doc_id] = {"text": text, "annotations": []}
                text = ""

            else:
                line_data = line.split("\t")

                if len(line_data) == 6:
                    annot_type = line_data[4]
                    doc_id = line_data[0]

                    if annot_type in ent_types or dataset == "med_mentions":

                        if dataset == "med_mentions":
                            annot_type = "BIO"

                        annot = (
                            line_data[1],
                            line_data[2],
                            line_data[3],
                            annot_type,
                            line_data[5].strip("\n"),
                        )

                        parsed_data[doc_id]["annotations"].append(annot)

    return parsed_data

def parse_pubtator(dataset, ent_types=[], evaluate=True):
    """
    Parse a dataset in the Pubtator format and return a dictionary containing
    document texts and their annotations.

    Parameters
    ----------
    dataset : str
        Name of the dataset to parse.
    ent_types : lst, optional
        Types of entities to include in the annotations 
        (e.g. ['Disease', 'Chemical'])
    evaluate : bool, optional
        If True, includes evaluation datasets. If False, includes training and
        development datasets as well (default is True).

    Returns
    -------
    dict
        A dictionary containing the text and annotations with the format:
        {doc_id: {"text": text, "annotations": [(start, end, entity, type, id)]}}
    """

    data_dir = f"data/datasets/{dataset}"

    if dataset == "bc5cdr":
        data_dir += "/CDR.Corpus.v010516"

        filenames = ["CDR_TestSet.PubTator.txt"]

        if not evaluate:
            filenames.append("CDR_DevelopmentSet.PubTator.txt")
            filenames.append("CDR_TrainingSet.PubTator.txt")

    elif dataset == "ncbi_disease":
        filenames = ["NCBItestset_corpus.txt"]

        if not evaluate:
            filenames.append("NCBIdevelopset_corpus.txt")
            filenames.append("NCBItrainset_corpus.txt")

        ent_types = ["Modifier", "SpecificDisease", "DiseaseClass"]

    elif dataset == "biored":
        filenames = ["Test.PubTator"]

        if not evaluate:
            filenames.append("Train.PubTator")
            filenames.append("Dev.PubTator")

    elif dataset == "med_mentions":
        filenames = ["corpus_pubtator.txt"]

    parsed_data = {}

    for filename in filenames:
        parsed_data = parse_pubtator_file(
            dataset, f"{data_dir}/{filename}", ent_types, parsed_data
        )

    return parsed_data

def create_datasets_pmids_list(email=None):
    """Create a file with PMIDs to exclude from the training data since they 
    correspond to documents present in evaluation datasets. 
    The goal is to further generate unbiased training data.

    List of evaluation datasets:
    -MedMentions [x] -> UMLS
    -BC5CDR [x] -> MESH
    -NCBI Disease [x] -> MESH
    -BioRED [x] -> MESH
    -NLM-Chem [x] -> MESH
    -LINNAEUS [x] -> NCBI Taxonomy
    -BC2GN [x] -> NCBI Gene
    -CRAFT Corpus [x] -> several ontologies
    """
    datasets_dir = "data/datasets"
    pmids_list = []
    out_info = ""

    # Import MedMentions PMIDs
    # medmentions_filepath = f"{datasets_dir}/med_mentions/corpus_pubtator_pmids_all.txt"

    # with open(medmentions_filepath, "r", encoding="utf-8") as in_file:
    #    med_pmids = in_file.readlines()
    #    in_file.close()
    #    med_pmids_up = [pmid.strip("\n") for pmid in med_pmids]
    #    pmids_list.extend(med_pmids_up)
    #    out_info += f"MedMentions IDs added: {len(med_pmids_up)}\n"

    # Import NLM-Chem PMIDs
    nlmchem_filepath = f"{datasets_dir}/nlm_chem/pmcids_corpus.txt"

    with open(nlmchem_filepath, "r", encoding="utf-8") as in_file2:
        nlmchem_pmids = in_file2.readlines()
        in_file2.close()
        nlmchem_pmids_up = [pmid.strip("\n") for pmid in nlmchem_pmids]
        pmids_list.extend(nlmchem_pmids_up)
        out_info += f"NLM_Chem IDs added: {len(nlmchem_pmids_up)}\n"

    # Import CRAFT Corpus PMIDs
    # craft_filepath = f"{datasets_dir}/craft/articles/ids/craft-pmids.txt"

    # with open(craft_filepath, "r", encoding="utf-8") as in_file3:
    #    craft_pmids = in_file3.readlines()
    #    in_file3.close()

    #    craft_pmids_up = [pmid.strip("\n") for pmid in craft_pmids]
    #    pmids_list.extend(craft_pmids_up)
    #    out_info += f"CRAFT Corpus IDs added: {len(craft_pmids_up)}\n"

    # BC5CDR
    bc5cdr_annots = parse_pubtator(
        "bc5cdr", ent_types=["Disease", "Chemical"], evaluate=True
    )
    bc5cdr_pmids = list(bc5cdr_annots.keys())
    pmids_list.extend(bc5cdr_pmids)
    out_info += f"BC5CDR IDs added: {len(bc5cdr_pmids)}\n"

    # Import PMIDs from NCBI Disease
    ncbi_disease_annots = parse_pubtator(
        "ncbi_disease", ent_types=["Disease"], evaluate=True
    )
    ncbi_disease_pmids = list(ncbi_disease_annots.keys())
    pmids_list.extend(ncbi_disease_pmids)
    out_info += f"NCBI Disease IDs added: {len(ncbi_disease_pmids)}\n"

    # Import PMIDs from BioRED
    biored_annots = parse_pubtator(
        "biored", ent_types=["Disease", "Chemical"], evaluate=True
    )
    biored_pmids = list(biored_annots.keys())
    pmids_list.extend(biored_pmids)
    out_info += f"BioRED IDs added: {len(biored_pmids)}\n"

    # Import PMIDs from BC2GN
    # bc2gn_pmids = get_bc2gn_pmids(datasets_dir)
    # pmids_list.extend(bc2gn_pmids)
    # out_info += f"BC2GN IDs added: {len(bc2gn_pmids)}\n"

    # Import PMIDs from LINNAEUS
    # linnaeus_pmids_filename = f"{datasets_dir}/linnaeus/pmids_list.txt"

    # if not os.path.exists(linnaeus_pmids_filename):
    #    print("LINNAEUS PMIDs file not found. Generating it...")
    #    linnaeus_pmids = get_linnaeus_pmids(datasets_dir, email=email)
    # Create a file to store the pmids
    #    with open(linnaeus_pmids_filename, "w", encoding="utf-8") as linnaeus_file:
    #        for pmid in linnaeus_pmids:
    #            linnaeus_file.write(f"{pmid}\n")

    #        linnaeus_file.close()

    # else:
    #    with open(linnaeus_pmids_filename, "r", encoding="utf-8") as in_file4:
    #        linnaeus_pmids = in_file4.readlines()
    #        in_file4.close()
    #        linnaeus_pmids = [pmid.strip("\n") for pmid in linnaeus_pmids]

    # pmids_list.extend(linnaeus_pmids)
    # out_info += f"LINNAEUS IDs added: {len(linnaeus_pmids)}\n"

    # Remove duplicates
    pmids_list_dedup = [pmid.strip("\n").strip("_PMID") for pmid in pmids_list]
    pmids_list_dedup = list(set(pmids_list_dedup))
    out_info += f"Total number of PMIDs (deduplicated): {len(pmids_list_dedup)}"

    # Write PMIDs to file
    with open(f"{datasets_dir}/ignore_pmids.txt", "w", encoding="utf-8") as out_file:
        for pmid in pmids_list_dedup:
            out_file.write(f"{pmid}\n")

        out_file.close()

    # Output info file
    with open(
        f"{datasets_dir}/ignore_pmids_info.txt", "w", encoding="utf-8"
    ) as out_file2:
        out_file2.write(out_info)
        out_file2.close()