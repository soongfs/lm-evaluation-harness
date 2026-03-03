from pathlib import Path

import datasets


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILES = {
    "train": "dataset/train.json",
    "validation": "dataset/val.json",
}


def _resolve_data_files(data_files):
    resolved = {}
    for split, path in data_files.items():
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path
        resolved[split] = str(file_path)
    return resolved


def load_local_dataset(**kwargs):
    data_files = kwargs.get("data_files", DEFAULT_DATA_FILES)
    dataset = datasets.load_dataset("json", data_files=_resolve_data_files(data_files))
    return {split: dataset[split] for split in dataset}


def process_docs(dataset):
    def _process(doc):
        choices = doc["choices"]
        answer = doc.get("answer", "")
        label = choices.index(answer) if answer in choices else -1
        return {
            "question": doc["question"],
            "choices": choices,
            "label": label,
        }

    return dataset.map(_process)
