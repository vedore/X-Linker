import os
import pandas as pd

RAW_DATA_PATH = os.path.abspath("data/raw/mesh_data")

kb_file = "medic/CTD_diseases.tsv"
skip_rows = 29
tsv_file = os.path.join(RAW_DATA_PATH, kb_file)
data_filepath = os.path.join(RAW_DATA_PATH, kb_file)

col_names = get_headers_from_tsv_file(tsv_file, skip_rows)

print(col_names)

mesh_data = pd.read_csv(data_filepath, sep='\t', header=None, names=col_names, skiprows=skip_rows)

print(mesh_data.index)

# drop duplicates
mesh_data.drop_duplicates(subset=mesh_data.columns[1], inplace=True)

# Handle missing values
missing_percentages = mesh_data.isnull().mean() * 100
print(missing_percentages)