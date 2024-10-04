from argparse import ArgumentParser, BooleanOptionalAction

from root.main.controllers.xlinker_controller import (xlinker_load_model, xlinker_kb_info, xlinker_dataset_abbreviations,
                                                      xlinker_load_tests, xlinker_prep_input, xlinker_apply_model_to_test_instances, xlinker_evaluation)

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

# Load and setup model to apply
custom_xtf, tfidf_model, cluster_chain = xlinker_load_model(args.model_dir, args.clustering)

# Load KB info
id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = xlinker_kb_info(args.kb, inference=True)

# Get abbreviations in dataset
abbreviations = xlinker_dataset_abbreviations(args.dataset, args.abbrv)

# Import test instances
test_annots_raw = xlinker_load_tests(args.dataset, args.ent_type, args.unseen)
test_input, test_annots = xlinker_prep_input(test_annots_raw, abbreviations, id_2_name)

# Apply model to test instances
output = xlinker_apply_model_to_test_instances(custom_xtf, test_input, tfidf_model,
                              test_annots, kb_names, kb_synonyms,
                              name_2_id_lower, synonym_2_id_lower, index_2_id,
                              args.pipeline, args.top_k, args.fuzzy_top_k, args.threshold)

# Evaluation
xlinker_evaluation(output, args.dataset, args.ent_type, args.kb, args.fuzzy_top_k, args.ppr)




