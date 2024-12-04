from src.featurization.preprocessor import Preprocessor

train_filepath = "data/raw/mesh_data/medic/train_Disease_500.txt"
labels_filepath = "data/raw/mesh_data/medic/labels.txt"

data = Preprocessor.load_data_from_file(train_filepath, labels_filepath)

print(data['corpus'])