#!/usr/bin/env bash

# Create create a version of the PubTator annotations file containing only 
# the relevant annotations (Disease, Chemical)

target_ent_type=$1

pattern1=$'\t'"Disease"$'\t'
pattern2=$'\t'"Chemical"$'\t'
pattern3=$'\t'"Species"$'\t'
pattern4=$'\t'"Gene"$'\t'
pattern5=$'\t'"ProteinMutation"$'\t'
pattern6=$'\t'"DNAMutation"$'\t'
pattern7=$'\t'"CellLine"$'\t'


patterns=(
    $'\t'"Disease"$'\t'
    $'\t'"Chemical"$'\t'
    $'\t'"Species"$'\t'
    $'\t'"Gene"$'\t'
    $'\t'"ProteinMutation"$'\t'
    $'\t'"DNAMutation"$'\t'
    $'\t'"CellLine"$'\t'
)

for pattern in "${patterns[@]}"; do
    target_pattern=$'\t'"$target_ent_type"$'\t'
    
    if [ "$pattern" == "$target_pattern" ]; then
        #remove target pattern from patterns
        patterns=("${patterns[@]/$target_pattern}")
    fi
done

#Convert patterns into a string delimited by "|"
patterns=$(IFS="|"; echo "${patterns[*]}" | tr -d '\t' | sed 's/||/|/g')

file_path="data/pubtator/bioconcepts2pubtatorcentral.offset"
output_file="data/pubtator/pubtator_${target_ent_type}"

#grep -v -P "$pattern1|$pattern2|$pattern3|$pattern4|$pattern5|$pattern6" "$file_path" > "$output_file"
echo "$patterns"
grep -v -P "$patterns" "$file_path" > "$output_file"