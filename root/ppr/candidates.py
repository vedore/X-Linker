import src.python.utils as utils
import os
import json
from rapidfuzz import process, fuzz
from root.ppr.kb import KnowledgeBase
from tqdm import tqdm


def check_if_related(candidate1, candidate2, kb_edges):
    """
    Determine if two candidates are related based on the knowledge base edges.

    It checks whether there is a direct relationship between
    `candidate1` and `candidate2` in the provided knowledge base edges. The
    knowledge base is represented as a list of tuples where each tuple
    represents an edge (relationship) between two nodes (candidates).

    Parameters
    ----------
    candidate1 : str
        The identifier of the first candidate.
    candidate2 : str
        The identifier of the second candidate.
    kb_edges : list of tuple of str
        A list of edges in the knowledge base. Each edge is represented as a
        tuple of two strings, indicating a relationship between two candidates.

    Returns
    -------
    bool
        True if `candidate1` and `candidate2` are related, False otherwise.
    """

    related = False

    if candidate1["url"] == candidate2["url"]:
        related = True

    if (
        candidate1["url"] in kb_edges
        and candidate2["url"] in kb_edges[candidate1["url"]]
    ):
        related = True

    if (
        candidate2["url"] in kb_edges
        and candidate1["url"] in kb_edges[candidate2["url"]]
    ):
        related = True

    return related


def get_candidates_info(candidates, id_to_info):
    """
    Retrieve information about candidates and their relationships from a
    knowledge base.

    Parameters
    ----------
    candidates : list of str
        A list of candidate identifiers for which information is to be retrieved.
    id_to_info : dict of str to dict
        A dictionary mapping candidate identifiers to their respective
        information dictionaries. Each information dictionary contains
        attributes and details such as the number of direct ancestors and
        descendants of the candidate.

    Returns
    -------
    dict of str to dict
        A dictionary where each key is a candidate identifier from the 
        `candidates` list and each value is a dictionary containing the 
        candidate's information and their relationships. The relationships 
        are stored under the key 'relationships'.
    """
    candidates_info = []

    for cand in candidates:
        cand_text = cand[2]
        cand_kb_id = cand[3]
        cand_score = cand[4]

        try:
            outcount = id_to_info[cand_kb_id][0]
            incount = id_to_info[cand_kb_id][1]

        except KeyError:
            incount = 0
            outcount = 0

        cand_id = ""

        if "OMIM" in cand_kb_id:
            cand_id = cand_kb_id[len("OMIM") :]

        elif cand_kb_id.startswith("C") or cand_kb_id.startswith("D"):
            cand_id = cand_kb_id[1:]

        elif cand_kb_id.startswith("MESH_C") or cand_kb_id.startswith("MESH_D"):
            cand_id = "000"

        candidates_info.append(
            {
                "url": cand_kb_id,
                "name": cand_text,
                "outcount": outcount,
                "incount": incount,
                "id": cand_id,
                "links": [],
                "score": cand_score,
            }
        )

    return candidates_info


def output_candidates_file(
    doc_id=None, doc_preds=None, kb_obj=None, ent_type=None, out_dir=None
):
    """
    Output candidate entities and their information to a file.

    Parameters
    ----------
    doc_id : str, optional
        The identifier of the document for which candidate entities are being
        processed.
    doc_preds : dict, optional
        Predicted entities for the document.
    kb_obj : object, optional
        A knowledge base object that provides access to entity information and
        relationships.
    ent_type : str, optional
        The type of entity to be filtered and included in the output file.
    out_dir : str, optional
        The directory where the output file will be saved.

    Returns
    -------
    .txt file in the directory specified by `out_dir`
    """

    out_pred = {}

    for search_key in doc_preds:
        out_pred[search_key] = get_candidates_info(
            doc_preds[search_key],
            kb_obj.id_2_info,
        )

    # Check for relations between candidates for diferent mentions
    for i, pred1 in enumerate(out_pred.keys()):
        candidates1 = out_pred[pred1]

        for j, pred2 in enumerate(out_pred.keys()):

            if pred1 != pred2:
                candidates2 = out_pred[pred2]

                for g, candidate_a in enumerate(candidates1):

                    for k, candidate_b in enumerate(candidates2):
                        related = check_if_related(
                            candidate_a, candidate_b, kb_obj.node_2_node
                        )

                        if related:
                            candidate_a["links"].append(candidate_b["id"])
                            candidate_b["links"].append(candidate_a["id"])

                            # Replace the info in candidates_info
                            candidates1[g] = candidate_a
                            out_pred[pred1] = candidates1
                            candidates2[k] = candidate_b
                            out_pred[pred2] = candidates2

    output = ""

    for search_key in out_pred.keys():
        ent_kb_id = search_key.split("_")[-1]
        output += f"ENTITY\ttext:{search_key}\tnormalName:{search_key}\tpredictedType:{ent_type}\tq:true\tqid:none\tdocId:{doc_id}\ttorigText:{search_key}\turl:{ent_kb_id}\n"

        for cand in out_pred[search_key]:
            links = ";".join(set(cand["links"]))
            output += f"CANDIDATE\tid:{cand['id']}\tinCount:{cand['incount']}\toutCount:{cand['outcount']}\tlinks:{links}\turl:{cand['url']}\tname:{cand['name']}\tnormalName:{cand['name'].lower()}\tnormalWikiTitle:{cand['name'].lower()}\tpredictedType:{ent_type}\n"

    with open(f"{out_dir}{doc_id}", "w", encoding="utf-8") as fout:
        fout.write(output)
        fout.close()


def map_to_kb(entity_text, names, syonyms, name_2_id, synonym_2_id, top_k=1):
    """
    Map an entity text to the most relevant entries in the knowledge base
    based on lexical similarity (Levensthein distance).


    Parameters
    ----------
    entity_text : str
        The text of the entity to be mapped to the knowledge base.
    names : list of str
        A list of names available in the knowledge base.
    synonyms : list of str
        A list of synonyms in the knowledge base.
    name_2_id : dict of str to str
        A dictionary mapping names to their respective identifiers in the
        knowledge base.
    synonym_2_id : dict of str to str
        A dictionary mapping synonyms to their respective identifiers in the
        knowledge base.
    top_k : int, optional
        The number of top matches to return. Default is 1.

    Returns
    -------
    list
        A list of dictionaries containing the top `top_k` matches. Each
        dictionary consists of the identifier, the matched name or synonym
        from the knowledge base and the matching score.
    """

    if entity_text in names:
        # There is an exact match for this entity
        top_concepts = [(entity_text, 100, 1.0)]

    elif entity_text in syonyms:
        # There is an exact match for this entity
        top_concepts = [(entity_text, 100, 1.0, "syn")]

    else:
        # Get first ten KB candidates according to lexical similarity
        # with entity_text
        top_concepts = process.extract(
            entity_text, names, limit=top_k, scorer=fuzz.token_sort_ratio
        )

        if top_concepts[0][1] == 100:
            # There is an exact match for this entity
            top_concepts = [top_concepts[0]]

        elif top_concepts[0][1] < 100:
            # Check for synonyms to this entity
            top_synonyms = process.extract(
                entity_text, syonyms, limit=10, scorer=fuzz.token_sort_ratio
            )

            for synonym in top_synonyms:

                if synonym[1] == 100:
                    synoynm_up = (synonym[0], synonym[1], synonym[2], "syn")
                    top_concepts = [synoynm_up]

                else:

                    if synonym[1] >= top_concepts[-1][1]:
                        synoynm_up = (synonym[0], synonym[1], synonym[2], "syn")
                        top_concepts.append(synoynm_up)

    # Build the candidates list with match id, name and matching score
    # with entity_text
    matches = []

    for concept in top_concepts:
        term_name = concept[0]
        term_id = ""

        if len(concept) == 4 and concept[3] == "syn":
            term_id = synonym_2_id[term_name]

        elif len(concept) == 3:
            term_id = name_2_id[term_name]

        else:
            term_id = "NIL"

        match = {"kb_id": term_id, "name": term_name, "score": concept[1] / 100}

        matches.append(match)

    return matches
