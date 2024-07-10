import json
import os
import pandas as pd
from math import log
from src.python.kbs import KnowledgeBase
from src.python.utils import calculate_topk_accuracy
from src.python.xlinker.candidates import map_to_kb, output_candidates_file


def prepare_ppr_input(
        run_name,
        pred_dataframe,
        ent_type,
        threshold=0.75,
        fuzzy_top_k=1, 
        kb=None,
        sapbert=False):
    """Prepare the input for the PPR model by generating the candidates file
    containign information to build the disambiguation graph"""

    out_dir = f"data/REEL/{run_name}/candidates/"
    os.makedirs(out_dir, exist_ok=True)
    
    kb_obj = KnowledgeBase(kb=kb, input_format="tsv")

    #lowercase all names and synonyms
    all_preds = {}

    for i, pred in enumerate(pred_dataframe.iterrows()):
        doc_id = pred[1]['doc_id']

        if doc_id not in all_preds:
            doc_preds = {}
        
        code = pred[1]['code']

        if 'OMIM' in code:
            code = code.replace('OMIM:', 'OMIM')

        pred_text = pred[1]['text']

        if ':' in pred_text:
            pred_text = pred_text.replace(':', '_')
            
        search_key = f"{pred_text}_{pred[1]['start']}_{code}".replace(' ', '_')
        doc_preds[search_key] = []
  
        for i, label in enumerate(pred[1]['codes']):
            name = kb_obj.id_2_name[label]

            if 'OMIM' in label:
                label = label.replace('OMIM:', 'OMIM')
            
            doc_preds[search_key].append(
                            (
                            pred[1]['start'], 
                            pred[1]['end'], 
                            name, 
                            label, 
                            pred[1]['scores'][i]
                            )
            )
        
        all_preds[doc_id] = doc_preds 

    for doc_id in all_preds:
        output_candidates_file(
            doc_id=doc_id,
            doc_preds=all_preds[doc_id],
            kb_obj=kb_obj, 
            ent_type=ent_type, 
            out_dir=out_dir
    )
    
    print(f"PPR candidates files generated in {out_dir}")


def add_predictions_to_tsv(predictions, orig_annotations):
    """Add the predictions from the PPR model to the original .tsv 
    predictions file
    """

    print("Adding predictions to TSV...")
    
    for row in orig_annotations.iterrows():
        doc_id = str(row[1]['doc_id'])
        #print(predictions[doc_id])
    
        if doc_id in predictions.keys():
            ent_start = row[1]['start']
            ent_text = row[1]['text'].replace(' ', '_')
            ent_kb_id = row[1]['code']

            if 'OMIM' in ent_kb_id:
                ent_kb_id = ent_kb_id.replace('OMIM:', 'OMIM')

            if ':' in ent_text:
                ent_text = ent_text.replace(':', '_')
                
            search_key = f"{ent_text}_{ent_start}_{ent_kb_id}"

            if search_key in predictions[doc_id].keys():
                orig_annotations.at[row[0], 'codes'] = \
                    [predictions[doc_id][search_key][0]]
                orig_annotations.at[row[0], 'scores'] = ["None"]
            else:
                print('search key not found:', search_key)

    return orig_annotations

#------------------------------------------------------------------------------
#                      REEL: INFORMATION CONTENT
#------------------------------------------------------------------------------

def build_term_counts(candidates_dir):
    """Build a dict containing the frequency of each candidate entity that 
    appears in the candidates files generated during the pre-processing stage.
    """

    term_counts = {}

    candidates_files = os.listdir(candidates_dir)
    
    # Get the term frequency in the corpus
    for filename in candidates_files: 

        lines = open(candidates_dir + filename, 'r').readlines()
        
        for line in lines:

            if line[:9] == 'CANDIDATE':
                url = line.split('\t')[5].split('url:')[1]

                if url not in term_counts.keys():
                    term_counts[url] = 1
                
                else:
                    term_counts[url] += 1

    return term_counts


def build_information_content_dict(candidates_dir, id_to_info, mode=None):
    """Generate dictionary with the information content for each candidate 
    term. For more info about the definition of information content see
    https://www.sciencedirect.com/science/article/pii/B9780128096338204019?via%3Dihub""" 

    term_counts = build_term_counts(candidates_dir)
    
    ic = {}
    total_terms = 0

    if mode == 'intrinsic':
        total_terms = len(id_to_info.keys())

    for term_id in term_counts:        
        
        term_probability = float()

        if mode == 'extrinsic':
            # Frequency of the most frequent term in dataset
            max_freq = max(term_counts.values()) 
            term_frequency = term_counts[term_id] 
            term_probability = (term_frequency + 1)/(max_freq + 1)
        
        elif mode == 'intrinsic':
            
            try:
                num_descendants = id_to_info[term_id][2]
                term_probability = (num_descendants + 1) / total_terms
                
            except:
                term_probability = 0.000001

        else:
            raise ValueError('Invalid mode!')
        
        information_content = -log(term_probability) + 1
        ic[term_id] = information_content + 1
    
    return ic


def generate_ic_file(reel_dir, candidates_dir, kb, model):#run_id, candidates_dir, id_to_info):
    """Generate file with information content of all entities present in the 
    candidates files."""

    
    kb_obj = KnowledgeBase(kb=kb, input_format="tsv")
  
    ic = build_information_content_dict(
        candidates_dir, kb_obj.id_2_info, mode='intrinsic') 

    # Build output string
    out_string = ""

    for term in ic.keys():
        out_string += term +'\t' + str(ic[term]) + '\n'

    # Create file ontology_pop with information content for all entities 
    # in candidates file

    with open(f"{reel_dir}/ic", 'w') as ic_file:
        ic_file.write(out_string)
        ic_file.close()

#------------------------------------------------------------------------------
#                      REEL: POST-PROCESSING
#------------------------------------------------------------------------------
def process_ppr_results(entity_type, reel_dir):
    """Process the results after the application of the PPR-IC model and
    output a JSON file in the directory 'tmp/REEL/results/<run_id>/."""

    
    ppr_filepath = f"{reel_dir}/ppr_scores"

    # Import PPR output
    with open(ppr_filepath, 'r') as results:
        data = results.readlines()
        results.close

    linked_entities = {}
    doc_id = ''
    entity_count = 0

    for line in data:
        
        if line != '\n':

            if line[0] == '=':
                doc_id = line.strip('\n').split(' ')[1]
                
            else:
                entity = line.split('\t')[1].split('=')[1]
               
                entity_count+=1           
                answer = line.split('\t')[3].split('ANS=')[1].strip('\n').\
                    replace('_', ':')

                if 'OMIM' in answer:
                    answer = answer.replace('OMIM', 'OMIM:')
                
                if doc_id in linked_entities.keys():
                    linked_entities[doc_id][entity] = (answer, entity_type)
                
                else:
                    linked_entities[doc_id] = {entity: (answer, entity_type)}
    
    return linked_entities


def run(
        entity_type=None, 
        kb=None, 
        reel_dir=None):
    """Run the PPR-IC model over the candidates files and update the .tsv
    predictions file.
    """

    # Output file with information content for each candidate
    candidates_dir = f"{reel_dir}/candidates/"
    generate_ic_file(reel_dir, candidates_dir, kb, "REEL")

    #------------------------------------------------------------------------#
    #                          REEL: PPR                                     
    #         Builds a disambiguation graph from each candidates file:            
    #         the nodes are the candidates and relations are added                
    #         according to candidate link_mode. After the disambiguation          
    #         graph is built, it runs the PPR algorithm over the graph            
    #         and ranks each candidate.                                           
    #------------------------------------------------------------------------#
    ppr_dir = reel_dir.strip('data/REEL')
    comm = f"java -classpath :src/java/ ppr_for_ned_all {ppr_dir} REEL"
    os.system(comm)

    #------------------------------------------------------------------------#
    #                         REEL: Post-processing                                                                     
    #------------------------------------------------------------------------# 
    predictions = process_ppr_results(entity_type, reel_dir)
    orig_annotations = pd.read_csv(f"{reel_dir}/xlinker_preds.tsv", sep="\t", keep_default_na=False)
    out_df=add_predictions_to_tsv(predictions, orig_annotations)
    out_df.to_csv(f"{reel_dir}/xlinker_preds_ppr.tsv", sep='\t', index=False)
    top1_accuracy = calculate_topk_accuracy(out_df, [1, 5]) # 5 is just a placeholder
    print(f"Top-1 Accuracy: {top1_accuracy[1]}")