#!/usr/bin/env bash

#MEDIC
python3 -c "from data_builder.shell_utils.generate_kb_mappings import generate_kb_mappings;generate_kb_mappings('medic')"

#CTD-Chemicals
python3 -c "from data_builder.shell_utils.generate_kb_mappings import generate_kb_mappings;generate_kb_mappings('ctd_chemicals')"

#CTD-Gene
python3 -c "from data_builder.shell_utils.generate_kb_mappings import generate_kb_mappings;generate_kb_mappings('ctd_genes')"

#NCBI-Taxon
#Generate ncbi_taxon.csv
#awk -F'|' '{ gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3 == "species") { print $1 } }' data/kbs/ncbi_taxon/taxdump/nodes.dmp > data/kbs/ncbi_taxon/species_ids #store the taxon ids relative to species
#awk -F'\t' 'NR==FNR{species_ids[$1]; next} $1 in species_ids {printf ("%s,%s\n", $1,$3)}' data/kbs/ncbi_taxon/species_ids data/kbs/ncbi_taxon/taxdump/names.dmp > data/kbs/ncbi_taxon/tmp.csv #only store the names of the species
#python3.9 -c "from src.python.utils import generate_ncbi_taxon;generate_ncbi_taxon()"
#python3.9 -c "from src.python.utils import generate_kb_mappings;generate_kb_mappings('ncbi_taxon')"
