import yaml


def load_config(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.load(f)
    return config
