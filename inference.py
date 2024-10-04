from root.main.controllers.inference_controller import inference_load_model, inference_kb_info, \
    inference_apply_model_to_input_entities

kb = "medic" # or "ctd_chemicals"
# check other models in the directory 'data/models/trained'
model_dir = "data/models/trained/disease_200_1ep"

#Label Representation via Positive Instance Feature Aggregation (PIFA).
#Check Page 6 of the article: https://arxiv.org/pdf/2010.05878
clustering = "pifa"

top_k = 5 # Number of top-k predictions to return

# Example of input entities
input_entities = ["Hypertension", "Diabetes", "Cancer"]

# Load and setup model to apply
custom_xtf, tfidf_model, cluster_chain = inference_load_model(model_dir, clustering)
print("Got the model")

# Load KB info
id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = inference_kb_info(kb, inference=True)
print("Got the KB")

# Apply model to input entities
inference_apply_model_to_input_entities(input_entities, custom_xtf, tfidf_model, index_2_id, id_2_name, top_k)