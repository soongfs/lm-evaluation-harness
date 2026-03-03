import re
from pathlib import Path

import datasets


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILES = {
    "train": "dataset/train.json",
    "validation": "dataset/val.json",
}
_PUNCT_RE = re.compile(r"[^\w\s]")


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


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _get_text(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def contains_answer(references, predictions) -> float:
    answer = _get_text(references)
    response = _get_text(predictions)

    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(response, str):
        response = str(response)

    answer = _normalize(answer)
    response = _normalize(response)
    if not answer or not response:
        return 0.0

    pattern = rf"(?<!\w){re.escape(answer)}(?!\w)"
    return 1.0 if re.search(pattern, response) else 0.0
