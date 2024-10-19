#!/usr/bin/env bash

#Arguments
type=$1
kb=$2
pubtator=$3

labels_filename="data/kbs/${kb}/labels.txt"

echo "Generating training data for ${type}, KB: ${kb}"

mkdir -p data/train
out_dir="data/train/${type}"
mkdir -p $out_dir

#----------------------------------------------------------------------------                                                                  
# ADD LABEL NAMES AND SYNONYMS FROM THE KB TO THE TRAINING DATA
#----------------------------------------------------------------------------                                                                  
echo "ADD LABEL NAMES AND SYNONYMS FROM THE KB TO THE TRAINING DATA"

out_labels_file="${out_dir}/labels.txt"
rm $out_labels_file
touch $out_labels_file

python3 -c "from data_builder.shell_utils.build_train import add_kb_labels_to_train;add_kb_labels_to_train('$kb', '$out_labels_file')"

if [ "$kb" = "medic" ] || [ "$kb" = "ctd_chemicals" ]; then
    
    cat $out_labels_file | tr '[:upper:]' '[:lower:]' | \
        # uppercase the first letter to keep the KB identifiers
        awk '{print toupper(substr($0,1,1)) substr($0,2)}' | \
        #sort the lines
        sort | \
        #remove duplicates and store th result in a new file
        uniq -u > ${out_labels_file}_dedup

elif [ "$kb" = "ctd_genes" ] || [ "$kb" = "ncbi_taxon" ];then
    cat $out_labels_file | tr '[:upper:]' '[:lower:]' | \
        #sort the lines
        sort | \
        #remove duplicates and store th result in a new file
        uniq -u > ${out_labels_file}_dedup
fi

rm $out_labels_file
mv ${out_labels_file}_dedup $out_labels_file

#create index labels file
awk -F'\t' -v out_dir="${out_dir}" '{ print $1 > out_dir"/index_labels" }' "$out_labels_file"

#----------------------------------------------------------------------------                                                                  
# Iterate over Pubtator files and add content to train file
#----------------------------------------------------------------------------                                                                  
echo "ADD PUBTATOR ANNOTATIONS TO THE TRAINING DATA"

if [ "$pubtator" = "pubtator" ]; then
    pubtator_dir="data/pubtator"
    out_pub_file="${pubtator_dir}/${type}_pub"   
    
    #if [ ! -f $out_pub_file ];then
        # Iterates over the file with Pubtator annotations
        # select only those with the desired type
        # lowercase the lines
        # uppercase the first letter to keep the KB identifiers
        #sort the lines
        #remove duplicates and store th result in a new file
        echo "Processing Pubtator file..."
        original_pub_file="${type,}2pubtator3"

        if [ "$kb" = "medic" ]; then
            grep -v -w -F -f data/datasets/ignore_pmids.txt data/pubtator/${original_pub_file} | \
                cut -d$'\t' -f3,4 | \
                awk -F'\t' '{ split($2, parts, "|"); for (i=1; i<=length(parts); i++) print $1 "\t" parts[i] }' | \
                sed 's/MESH://g' | \
                tr '[:upper:]' '[:lower:]' | \
                sort -u | \
                sed 's/omim/OMIM/' | \
                awk '{print toupper(substr($0,1,1)) substr($0,2)}' > $out_pub_file

        elif [ "$kb" = "ctd_chemicals" ]; then
            echo 'bzz'
            #grep -v -w -F -f data/datasets/ignore_pmids.txt data/pubtator/chemical2pubtator3 | \
            #    cut -d$'\t' -f3,4 | \
            #    awk -F'\t' '{ split($2, parts, "|"); for (i=1; i<=length(parts); i++) print $1 "\t" parts[i] }' | \
            #    grep -v '^-.*' | tr '[:upper:]' '[:lower:]' | \
            #    sort -u | \
            #    sed 's/mesh://g' | \
            #    awk '{print toupper(substr($0,1,1)) substr($0,2)}' | \
            #    tr -s ' ' > $out_pub_file

        elif [ "$kb" = "ctd_genes" ] || [ "$kb" = "ncbi_taxon" ];then
            grep -v -w -F -f data/datasets/ignore_pmids.txt data/pubtator/${original_pub_file} | \
                    cut -d$'\t' -f3,4 | \
                    awk -F'\t' '{ split($2, parts, "|"); for (i=1; i<=length(parts); i++) print $1 "\t" parts[i] }' | \
                    tr '[:upper:]' '[:lower:]' | \
                    sort -u > $out_pub_file

        fi

        #convert KB identifiers in the file $out_pub_file into indexes
        echo "Converting KB identifiers into indexes..."
        while IFS=$'\t' read -r label text; do

            index=$(grep -n "$label" $labels_filename | head -n 1 | cut -d ':' -f 1)
            
            if [ "$index" != "" ];then
                index=$((index - 1))
                echo -e "${index}\t${text}" >> ${out_pub_file}_index_tmp
            else
                echo $label >> data/pubtator/${type}_labels_not_in_kb
            fi
        
        done < $out_pub_file

        #Remove the lines with incorrect format, (for example, without index)
        python3 -c "from data_builder.shell_utils.build_train import correct_pub_file;correct_pub_file('${out_pub_file}_index_tmp', '${out_pub_file}_index')"
    #fi
fi 

rm "${out_pub_file}_index_tmp"
out_pub_file="${out_pub_file}_index"
awk -F'\t' -v out_dir="${out_dir}" '{ print $1 > out_dir"/index_pub" }' "$out_pub_file"

#----------------------------------------------------------------------------
# Join the different files and create the final training file
#----------------------------------------------------------------------------
echo "JOINING FILES AND CREATING FINAL TRAINING FILE..."
final_file_tmp="${out_dir}/dataset_${type}_tmp.txt"
rm $final_file_tmp

if [ "$pubtator" = "pubtator" ]; then   
    echo Joining files $out_labels_file and $out_pub_file
    cat $out_labels_file $out_pub_file > $final_file_tmp

else
    cat $out_labels_file > $final_file_tmp
fi

final_file_unc="${out_dir}/dataset_${type}_uncoded.txt"
strings $final_file_tmp > $final_file_unc

rm $final_file_tmp

#----------------------------------------------------------------------------                                                                  
# Ensure that that the generated files have utf-8 encoding
#----------------------------------------------------------------------------    
echo "ENSURING UTF-8 ENCODING"
python3.9 -c "from data_builder.shell_utils.build_train import correct_train_data_encoding;correct_train_data_encoding('${final_file_unc}')"

final_file_utf="${final_file_unc//.txt/_utf8.txt}"
rm $final_file_unc

final_file="${final_file_utf//_uncoded_utf8.txt/.txt}"
python3.9 -c "from data_builder.shell_utils.build_train import correct_pub_file;correct_pub_file('${final_file_utf}', '${final_file}')"
rm ${final_file_utf}
echo "Final file created: ${final_file}"