import json
import pathlib

def write_json(artifacts, save_file_path = "artifacts.json"):
    save_file_path = pathlib.Path(save_file_path)
    with open(save_file_path, "w") as f:
       json.dump(artifacts, f, indent=4)

def load_json(save_file_path = "artifacts.json"):
    with open(save_file_path) as f:
        return json.loads(f.read())