# Hybrid X-Linker: Automated Data Generation and Extreme Multi-label Ranking for Biomedical Entity Linking


This repository includes the necessary code to reproduce all the experiments described [here](https://arxiv.org/abs/2407.06292).

Citation:

```
@misc{ruas2024,
      title={Hybrid X-Linker: Automated Data Generation and Extreme Multi-label Ranking for Biomedical Entity Linking}, 
      author={Pedro Ruas and Fernando Gallego and Francisco J. Veredas and Francisco M. Couto},
      year={2024},
      eprint={2407.06292},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.06292}, 
}
```


## Environment setup

## Maneira que dá

- Pull the image of ubuntu 20.04 ou 22.04

...
docker pull ubuntu:22.04
...

- Run the docker inside the folder of the x_linker

...
docker run -v $(pwd):/x_linker --name x_linker --ipc="host" -it <image ID> bash
...

- If u want to re-run the docker

...
docker start <container-name>

docker exec -it <container-name> bash
...

- if u want to close just type

...
exit
...

- Update system and install python 

...
apt-get update && apt-get install -y build-essential git python3 python3-distutils python3-venv
...

- Create a venv to run python inside the folder

...
python3 -m venv venv

source venv/bin/activate
...

- afterwards, inside the venv bash, Install all the dependicies inside the requirements.txt

...
pip install -r "req.txt"
...

- to exit just type

...
deactivate
...

- To test u can just use 

...
python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir data/models/trained/disease_200_1ep -top_k 2 --abbrv --pipeline --threshold 0.15 --ppr
...

The experiments were done in an environment based on Python 3.9.1.

Minimum storage: 11 GB free disk space.

To use the official [Docker image for Python](https://hub.docker.com/layers/library/python/3.9-slim-bullseye/images/sha256-d4ea18d0da466f8e47eb9ead289da29c0ea87573370d7818e1669e9fa1f19377?context=explore), pull the image to your local environment:

```
docker pull python:3.9-slim-bullseye

3.10
```

Run a container in the project directory:

```
nvidia-docker run -v $(pwd):/x_linker --name x_linker --ipc="host" --gpus '"device=0"' -it <image ID> bash


"Dentro da pasta do x_linker" 

docker run -v $(pwd):/x_linker --name x_linker --ipc="host" -it <image ID> bash
```

Replace the Docker image ID by the respective value and the number of GPU devices.


Inside the container, install the requirements:


Feito para correr em containers 
```
./setup.sh
```

Run in the root directory of the project:

```
export PYTHONPATH="${PYTHONPATH}:"
```

## Data preparation

Data is stored in https://zenodo.org/records/12704543.

Get knowledge base files, datasets/corpora (and PubTator annotations if building training data from scratch):

```
./src/bash/get_eval_data.sh
```

If you want to generate all the training data from scratch and train PECOS-EL models follow the instructions in the file 'build_train.md'. To just perform evaluation or use X-Linker for inference follow the next steps in this file.


## Evaluation
 
### X-Linker

To evaluate X-Linker in datasets:

```
python src/python/xlinker/evaluate.py -dataset <> -ent_type <> -kb <> -model_dir <> -top_k <> -clustering <> --abbrv --pipeline --ppr --fuzzy_top_k
```

Arguments:
* -dataset: evaluation dataset, 'bc5cdr', 'biored', 'nlm_chem' or 'ncbi:disease'; type=str, required=True
* -ent_type: entity type, 'Disease' or 'Chemical' type=str, required=True
* -kb: target knowledge organization system, 'medic' or 'ctd_chemicals; type=str, required=True
* -model_dir: directory 'data/models/trained' followed by the specific model directory: 'chemical_all_1ep', 'chemical_only_kb_1ep', 'disease_100_1ep', 'disease_200_1ep', 'disease_300_1ep', 'disease_400_1ep' 'disease_only_kb_1ep'; type=str, required=True
* -top_k: top-k candidates to output; type=int, default=5
* -clustering: type=str, default="pifa"
* --abbrv: use abbreviation detector or not; default=False, action=BooleanOptionalAction
* --pipeline: use the X-Linker pipeline or not default=False, action=BooleanOptionalAction
* --threshold: threshold to filter candidates; type=float, default=0.1
* --ppr: use the Personalized PageRank or not default=False, action=BooleanOptionalAction
* --fuzzy_top_k: number of candidates to be retrieved by the string matcher; type=int, default=1


Example for BC5CDR-Disease dataset:

```
python src/python/xlinker/evaluate.py -dataset bc5cdr -ent_type Disease -kb medic -model_dir data/models/trained/disease_200_1ep -top_k 2 --abbrv --pipeline --threshold 0.15 --ppr
```


Example for BioRED-Chemical dataset:

```
python src/python/xlinker/evaluate.py -dataset biored -ent_type Chemical -kb ctd_chemicals -model_dir data/models/trained/chemical_1ep_all -top_k 2 --abbrv --pipeline --threshold 0.1 --ppr
```


### SapBERT:

```
python src/python/sapbert.py -ent_type <ent_type> -dataset <disease> -kb <kb> -top_k <top_k> --abbrv
```

Arguments:
- ent_type: entity type, 'Disease' or 'Chemical' type=str, required=True
- dataset: evaluation dataset, 'bc5cdr', 'biored', 'nlm_chem' or 'ncbi:disease'; type=str, required=True
- kb: target knowledge organization system, 'medic' or 'ctd_chemicals; type=str, required=True
- top_k: top-k candidates to output; type=int, default=5
-- abbrv: use abbreviation detector or not; default=False, action=BooleanOptionalAction


Example:

```
python3.9 src/python/sapbert.py -dataset bc5cdr -ent_type Disease -kb medic -top_k 1
```

## Inference
To use any of the PECOS-EL models for inference in a list of input entities run the script 'src/python/xlinker/inference_pecos.py'. Change the variable 'input_entities' manually. 

Example using the input ["Hypertension", "Diabetes", "Cancer"]:

```
python src/python/xlinker/inference_pecos.py
```

Output:

```
model loaded!
KB info loaded!
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 2219.99it/s]
Entity: hypertension
Output: [('Hypertension', 'D006973', 1.0), ('Essential Hypertension', 'D000075222', 0.20821528136730194), ('Hypertension, Pregnancy-Induced', 'D046110', 0.18146659433841705), ('Hypertension, Malignant', 'D006974', 0.17562662065029144), ('Hypertension, Renovascular', 'D006978', 0.11986447870731354)]
--------
Entity: diabetes
Output: [('Diabetes Mellitus', 'D003920', 0.899384617805481), ('Diabetes Mellitus, Insulin-Dependent, 19', 'C565715', 0.587672233581543), ('Diabetes Mellitus, Insulin-Dependent, 7', 'C563957', 0.5324499607086182), ('Diabetes Mellitus, Experimental', 'D003921', 0.5279378890991211), ('Diabetes Mellitus, Transient Neonatal, 1', 'C563322', 0.47543802857398987)]
--------
Entity: cancer
Output: [('Pharyngeal Neoplasms', 'D010610', 0.7916738986968994), ('Tongue Neoplasms', 'D014062', 0.7916738986968994), ('Breast Neoplasms', 'D001943', 0.7492217421531677), ('Laryngeal Neoplasms', 'D007822', 0.7154863476753235), ('Gallbladder Neoplasms', 'D005706', 0.5262947082519531)]

```