import os
import json
import pandas as pd

from pathlib import Path


def generate_kb_mappings(kb):
    """
    Generate mappings between knowledge base concept names and idenfitiers 
    (i.e. labels). The mappings are stored in JSON files.
    """

    data_dir = Path(f"data/kbs/{kb}")

    kb_mappings = {
        "medic": {
            "filepath": "CTD_diseases.tsv",
            "id_column": "DiseaseID",
            "name_column": "DiseaseName",
            "skiprows": 29
        },
        "ctd_chemicals": {
            "filepath": "CTD_chemicals.tsv",
            "id_column": "ChemicalID",
            "name_column": "ChemicalName",
            "skiprows": 29
        },
        "ctd_genes": {
            "filepath": "CTD_genes.tsv",
            "id_column": "GeneID",
            "name_column": "GeneSymbol",
            "skiprows": 29
        },
        "ncbi_taxon": {
            "filepath": "ncbi_taxon.tsv",
            "id_column": "TaxonID",
            "name_column": "TaxonName",
            "skiprows": 1
        }
    }

    config = kb_mappings[kb]

    data_filepath = data_dir / config["filepath"]
    id_column = config["id_column"]
    name_column = config["name_column"]

    if kb == "medic":
        col_names = ["DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition", "ParentIDs", "TreeNumbers",
        "ParentTreeNumbers", "Synonyms", "Slimmappings"]
    elif kb == "ctd_chemicals":
        col_names = ["ChemicalName", "ChemicalID", "CasRN", "Definition", "ParentIDs", "TreeNumbers",
        "ParentTreeNumbers", "Synonyms"]
    elif kb == "ctd_genes":
        col_names =  ["GeneSymbol", "GeneName", "GeneID", "AltGeneIDs", "Synonyms", "BioGRIDIDs",
        "PharmGKBIDs", "UniProtIDs"]
    else:
        print("Kb Doesn't exit")
        exit(0)

    data = pd.read_csv(data_filepath, names=col_names, sep="\t", skiprows=config["skiprows"])

    print(data.iloc[1])

    """
    # Clean identifiers for specific KBs
    if kb in ["medic", "ctd_chemicals"]:
        data[id_column] = data[id_column].str.replace("MESH:", "")

    # Labels are equal to ids
    labels = data[id_column].tolist()

    names = data[name_column].tolist()

    # Generate mappings
    synonym_2_id, id_2_synonym, label_2_def = {}, {}, {}

    for _, row in data.iterrows():
        synonyms = row["Synonyms"].split("|") if isinstance(row["Synonyms"], str) else []
        for synonym in synonyms:
            synonym_2_id[synonym] = row[id_column]

        id_2_synonym[row[id_column]] = synonyms

        if kb not in ["ctd_genes", "ncbi_taxon"]:
            definition = row.get("Definition")
            if isinstance(definition, str):
                label_2_def[row[id_column]] = definition

    index_2_label = {i: label for i, label in enumerate(labels)}
    label_2_index = {label: i for i, label in enumerate(labels)}

    # Write files
    def write_json(data, filename):
        with open(data_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def write_text(data, filename):
        with open(data_dir / filename, "w", encoding="utf-8") as f:
            for line in data:
                f.write(f"{line}\n")

    # Save mappings to files
    write_text(labels, "labels.txt")
    write_json(label_2_def, "label_2_def.json")
    write_json(index_2_label, "index_2_label.json")
    write_json(label_2_index, "label_2_index.json")
    write_json(dict(zip(labels, names)), "label_2_name.json")
    write_json(dict(zip(names, labels)), "name_2_label.json")
    write_json(synonym_2_id, "synonym_2_label.json")
    write_json(id_2_synonym, "label_2_synonym.json")

    """

generate_kb_mappings("medic")