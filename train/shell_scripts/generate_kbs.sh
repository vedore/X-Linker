#!/usr/bin/env bash

#MEDIC
python3 -c "from data_builder.shell_utils.generate_kb_mappings import generate_kb_mappings;generate_kb_mappings('medic')"

#CTD-Chemicals
python3 -c "from data_builder.shell_utils.generate_kb_mappings import generate_kb_mappings;generate_kb_mappings('ctd_chemicals')"

