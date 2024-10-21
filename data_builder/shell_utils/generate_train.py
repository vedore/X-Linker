import argparse
import shutil
from pathlib import Path
from data_builder.shell_utils.build_train import add_kb_labels_to_train, correct_pub_file, correct_train_data_encoding


def create_directories(out_dir):
    """Create directories if they don't exist."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def read_file_lines(file_path):
    """Read lines from a file and return them as a list."""
    with open(file_path, 'r') as f:
        return f.readlines()


def write_file_lines(file_path, lines):
    """Write a list of lines to a file."""
    with open(file_path, 'w') as f:
        f.writelines(lines)


def process_kb_labels(kb, out_labels_file):
    """Add KB labels to the training data and process them based on the KB type."""
    add_kb_labels_to_train(kb, out_labels_file)

    with open(out_labels_file, 'r') as infile:
        labels = infile.readlines()

    # Convert all labels to lowercase, keeping the first letter uppercase
    labels = [label.lower() for label in labels]
    labels = [label.capitalize() for label in labels]

    # Remove duplicates and sort labels
    labels = sorted(set(labels))

    # Write deduplicated and sorted labels back to the file
    write_file_lines(f"{out_labels_file}_dedup", labels)

    # Replace the original file with deduplicated version
    Path(out_labels_file).unlink(missing_ok=True)
    shutil.move(f"{out_labels_file}_dedup", out_labels_file)


def create_index_file(labels_file, out_dir):
    """Create an index file based on labels."""
    index_labels_file = Path(out_dir) / "index_labels"
    with open(labels_file, 'r') as lf, open(index_labels_file, 'w') as ilf:
        for line in lf:
            ilf.write(line.split('\t')[0] + '\n')


def process_pubtator_data(kb, pubtator_dir, type_arg, labels_filename, out_pub_file):
    """Process Pubtator annotations and convert KB identifiers into indexes."""
    original_pub_file = f"{type_arg}2pubtator3"
    original_pub_path = Path(pubtator_dir) / original_pub_file

    # Read the original pubtator file
    with open(original_pub_path, 'r') as infile:
        pubtator_lines = infile.readlines()

    # Remove lines with PMIDs listed in the ignore file
    ignore_pmids_file = Path("data/datasets/ignore_pmids.txt")
    ignore_pmids = set(read_file_lines(ignore_pmids_file))

    filtered_lines = []
    for line in pubtator_lines:
        if line.split('\t')[0] not in ignore_pmids:
            filtered_lines.append(line)

    # Extract fields and process each line
    processed_lines = []
    for line in filtered_lines:
        fields = line.split('\t')
        entity = fields[2].split('|')
        for part in entity:
            processed_lines.append(f"{fields[0]}\t{part}\n")

    # Remove MESH prefixes, convert to lowercase, and capitalize first letter
    processed_lines = [line.replace('MESH:', '').lower() for line in processed_lines]
    processed_lines = [line.capitalize() for line in processed_lines]

    # Deduplicate and sort
    processed_lines = sorted(set(processed_lines))

    # Write processed Pubtator data to file
    write_file_lines(out_pub_file, processed_lines)

    # Convert KB identifiers in the pubtator file to indexes using the labels file
    convert_kb_identifiers_to_indexes(out_pub_file, labels_filename)


def convert_kb_identifiers_to_indexes(out_pub_file, labels_filename):
    """Convert KB identifiers into indexes using labels."""
    labels = read_file_lines(labels_filename)
    label_to_index = {label.strip(): idx for idx, label in enumerate(labels)}

    with open(out_pub_file, 'r') as pub_file, open(f"{out_pub_file}_index_tmp", 'w') as pub_index_tmp:
        for line in pub_file:
            label, text = line.strip().split('\t')
            index = label_to_index.get(label)
            if index is not None:
                pub_index_tmp.write(f"{index}\t{text}\n")
            else:
                # Log labels not found in the KB
                with open(f"data/pubtator/{type_arg}_labels_not_in_kb", 'a') as not_in_kb_file:
                    not_in_kb_file.write(f"{label}\n")

    correct_pub_file(f"{out_pub_file}_index_tmp", f"{out_pub_file}_index")
    Path(f"{out_pub_file}_index_tmp").unlink()


def join_files(out_labels_file, out_pub_file, final_file_tmp, pubtator):
    """Join label and Pubtator files into one final training file."""
    final_lines = read_file_lines(out_labels_file)
    if pubtator == "pubtator":
        final_lines += read_file_lines(out_pub_file)
    
    write_file_lines(final_file_tmp, final_lines)


def ensure_utf8_encoding(final_file_tmp, final_file_unc):
    """Ensure UTF-8 encoding for the final file."""
    with open(final_file_tmp, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(final_file_unc, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate training data.")
    parser.add_argument('type', help="The type of data to generate ('Chemical', 'Disease', 'Species', 'Gene')")
    parser.add_argument('kb', help="The knowledge base to use ('medic', 'ctd_chemicals', 'ncbi_taxon', 'ncbi_gene')")
    parser.add_argument('pubtator', help="Flag for Pubtator data processing 'pubtator'")
    args = parser.parse_args()

    labels_filename = Path(f"data/kbs/{args.kb}/labels.txt")
    out_dir = Path(f"data/train/{args.type}")

    print(f"Generating training data for {args.type}, KB: {args.kb}")
    create_directories(out_dir)

    out_labels_file = out_dir / "labels.txt"

    if out_labels_file.exists():
        out_labels_file.unlink()
    out_labels_file.touch()

    # Process KB labels
    print("Processing KB labels...")
    process_kb_labels(args.kb, str(out_labels_file))

    # Create index file
    create_index_file(out_labels_file, out_dir)

    # Process Pubtator data
    if args.pubtator == "pubtator":
        print("Processing Pubtator data...")
        pubtator_dir = "data/pubtator"
        out_pub_file = Path(pubtator_dir) / f"{args.type}_pub"
        process_pubtator_data(args.kb, pubtator_dir, args.type, labels_filename, out_pub_file)

        # Create index pub file
        index_pub_file = out_dir / "index_pub"
        with open(out_pub_file, 'r') as pub_file, open(index_pub_file, 'w') as ipf:
            for line in pub_file:
                ipf.write(line.split('\t')[0] + '\n')

    # Join files and create final dataset
    print("Joining files and creating final training file...")
    final_file_tmp = out_dir / f"dataset_{args.type}_tmp.txt"
    final_file_unc = out_dir / f"dataset_{args.type}_uncoded.txt"
    join_files(out_labels_file, out_pub_file, final_file_tmp, args.pubtator)

    # Ensure UTF-8 encoding
    ensure_utf8_encoding(final_file_tmp, final_file_unc)

    # Correct file encoding
    print("Ensuring UTF-8 encoding...")
    correct_train_data_encoding(final_file_unc)

    final_file_utf = final_file_unc.with_name(final_file_unc.stem + "_utf8.txt")
    final_file = final_file_utf.with_name(final_file_utf.stem.replace("_uncoded_utf8", "") + ".txt")
    correct_pub_file(final_file_utf, final_file)
    final_file_utf.unlink()

    print(f"Final file created: {final_file}")


if __name__ == "__main__":
    main()
