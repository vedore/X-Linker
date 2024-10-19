from argparse import ArgumentParser, BooleanOptionalAction

from root.controllers.sapbert_controller import sapbert_kb_info, sapbert_load_setup_model, \
    sapbert_enconde_kb_labels, sapbert_dataset_abbreviations, sapbert_load_tests, sapbert_apply_model_to_test_instances, \
    sapbert_evaluation

# Parse arguments
parser = ArgumentParser()
parser.add_argument("-ent_type", type=str, required=True)
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-kb", type=str, required=True)
parser.add_argument("-top_k", type=int, default=5)
parser.add_argument("--abbrv", default=False, action=BooleanOptionalAction)
args = parser.parse_args()

# Load KB info
all_names, all_ids, kb_pairs, label_2_name = sapbert_kb_info(args.kb)

# Load and setup model to apply
model, tokenizer = sapbert_load_setup_model()

# Encode KB labels
all_reps_emb = sapbert_enconde_kb_labels(model, tokenizer, all_names, args.ent_type)

# Abbreviations
abbreviations = sapbert_dataset_abbreviations(args.dataset)

# Import test instances
test_input, test_annots = sapbert_load_tests(abbreviations, label_2_name, args.dataset, args.ent_type)

# Apply model to test instances
predictions = sapbert_apply_model_to_test_instances(model, tokenizer, test_input, test_annots, all_reps_emb, kb_pairs, args.top_k)

# Evaluation
sapbert_evaluation(predictions)