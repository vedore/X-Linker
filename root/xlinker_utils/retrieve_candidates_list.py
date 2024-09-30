from src.python.xlinker.candidates import map_to_kb


"""
    Process the predictions of the PECOS-EL model to generate a list of
    the top K candidates for each annotation in the input.
"""
def retrieve_candidates_list(predictions=None, doc_annotations=None, index_2_label=None, label_2_name=None, kb_obj=None, top_k=None, threshold=0.10):
    doc_preds = {}

    for i, annot in enumerate(doc_annotations):
        start = annot[0]
        end = annot[1]
        ent_text = annot[2]
        doc_preds[str(i)] = []

        pred_indexes = predictions[i, :].indices.tolist()

        top_cand_score = predictions[i, :].data[0]
        top_cand_label = index_2_label[f"{pred_indexes[0]}"]
        top_cand_text = label_2_name[top_cand_label]
        search_key = str(i)
        doc_preds[search_key].append(
            (start, end, top_cand_text, top_cand_label, top_cand_score)
        )

        if top_cand_score < threshold:
            del doc_preds[search_key][0]

            # complete candidates list with candidates retrived by string matching
            matches = map_to_kb(ent_text, kb_obj.name_2_id, kb_obj.synonym_2_id, 1)

            for match in matches:
                doc_preds[search_key].append(
                    (start, end, match["name"], match["kb_id"], match["score"])
                )

    return doc_preds