from argparse import ArgumentParser, BooleanOptionalAction

from root.main.controllers.controller import model, kb_info, dataset_abbreviations, load_tests, prep_input, \
    apply_model_to_test_instances, evaluation

# Parse arguments
parser = ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-ent_type", type=str, required=True)
parser.add_argument("-kb", type=str, required=True)
parser.add_argument("-model_dir", type=str, required=True)
parser.add_argument("-top_k", type=int, default=5)
parser.add_argument("-clustering", type=str, default="pifa")
parser.add_argument("--abbrv", default=False, action=BooleanOptionalAction)
parser.add_argument("--pipeline", default=False, action=BooleanOptionalAction)
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--ppr", default=False, action=BooleanOptionalAction)
parser.add_argument("--fuzzy_top_k", type=int, default=1)
parser.add_argument("--unseen", default=False, action=BooleanOptionalAction)
args = parser.parse_args()

# load model
custom_xtf, tfidf_model, cluster_chain = model(args.model_dir, args.clustering)

# efficiently loads a knowledge base's data, creating useful mappings and lists based on whether it is called in inference mode or not.
id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = kb_info(args.kb, inference=True)

# manage and retrieve abbreviations for a specified dataset
abbreviations = dataset_abbreviations(args.dataset, args.abbrv)

# get test annotations
test_annots_raw = load_tests(args.dataset, args.ent_type, args.unseen)

test_input, test_annots = prep_input(test_annots_raw, abbreviations, id_2_name)

output = apply_model_to_test_instances(custom_xtf, test_input, tfidf_model,
                              test_annots, kb_names, kb_synonyms,
                              name_2_id_lower, synonym_2_id_lower, index_2_id,
                              args.pipeline, args.top_k, args.fuzzy_top_k, args.threshold)

evaluation(output, args.dataset, args.ent_type, args.kb, args.fuzzy_top_k, args.ppr)




