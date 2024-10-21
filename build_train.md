# Build train data and models from scratch


## Get data
Two options: download the already generated training files or build them from scratch using the original Pubtator annotations.

Download Pubtator annotations files:

```
./src/bash/get_pubtator.sh
```

Get the already generated training files:

```
./src/bash/get_training_data.sh
```


Download KB data:

```
./src/bash/get_data_to_build.sh
```

If u want to generate kbs run the script
```
generate_kbs.sh


NOTE: If you generate a new training dataset using recent versions of the KB, you will need to train new models. The models that are provided were trained using older versions of the KBs.


## Generate Training data files
If you download the Pubtator annotations and intend to generate the training files from scratch (if not skip this step):

```
./src/bash/generate_train.sh <type> <kb> <pubtator>
```

Arguments:
- type: Entity type to tag. "Chemical", "Disease", "Species", "Gene"
- kb: target Knowledge Base. "medic", "ctd_chemicals", "ncbi_taxon", "ncbi_gene"
- pubtator: include pubtator annotations or not

Example:

```
./src/bash/generate_train.sh Disease medic pubtator
```

Or:

```
./src/bash/generate_train.sh Chemical ctd_chem pubtator

```


## Train PECOS-EL models

To train a PECOS-EL models in the previously referred training files:

```
python src/python/xlinker/train_pecos.py -ent_type <> -kb <> -model <> -dataset <> -ninstances <> -clustering <>
```

Arguments: 

* -ent_type: Target entity type; options: 'Disease', 'Chemical', 'Bio', 'Species', 'Gene'; type:str
* -kb: Target knowledge base; options: 'medic', 'ctd_chemical', 'umls', 'ncbi_taxon', 'ctd_gene';type:`str`
* -model: Deep text vectorizer; options: 'bert', 'biobert', 'scibert', 'pubmedbert'; type: `str`
* -ninstances: Number of instances by concept in the training data; options: 10, 50, 100, 500; type: `int`
* -clustering: Label representation method for clustering; options: 'pifa', 'pifa_lf'; type: `str`

Example:

```
python src/python/xlinker/train.py -ent_type Disease -kb medic -model biobert -clustering pifa
```