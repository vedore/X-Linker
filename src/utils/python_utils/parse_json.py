import json


def parse_json(in_filepath):
    """
    Parse a JSON file.
    """
    try:
        with open(in_filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
            raise FileNotFoundError(f"File {in_filepath} not found.")
    except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file {in_filepath}.")

