import os
import bconv


def convert_bioc_xml_2_pubtator(in_filepath, out_filepath):
    """
    Convert a BioC XML file to a PubTator file.
    """

    coll = bconv.load(in_filepath, fmt="bioc_xml", byte_offsets=False)
    bconv.dump(coll, out_filepath, fmt="pubtator")

def convert_nlm_chem_2_pubtator():
    """
    Convert the NLM-Chem dataset to the Pubtator format. Create a single
    file containing all annotations.
    """

    data_dir = "data/datasets/nlm_chem"

    pmids_test = open(f"{data_dir}/pmcids_test.txt", "r", encoding="utf-8").readlines()
    pmids_test = [pmid.strip("\n") for pmid in pmids_test]

    out_dir = f"{data_dir}/pubtator/"
    os.makedirs(out_dir, exist_ok=True)

    for i, pmid in enumerate(pmids_test):
        convert_bioc_xml_2_pubtator(
            f"{data_dir}/ALL/{pmid}_v1.xml", f"{out_dir}/{pmid}"
        )

    # Combine all text files into a single pubtator file
    pubtator_files = os.listdir(out_dir)
    pubtator_files = [f"{out_dir}/{f}" for f in pubtator_files]

    with open(f"{data_dir}/test_pubtator.txt", "w", encoding="utf-8") as out_file:
        for f in pubtator_files:
            with open(f, "r", encoding="utf-8") as in_file:
                out_file.write(in_file.read())
                out_file.write("\n")
