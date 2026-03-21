import os
import re
from functools import lru_cache
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


def _get_text(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_scorer():
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise ImportError(
            "kqa_pro_qa_bertscore requires the `bert-score` package. "
            "Install it before running this task."
        ) from exc

    model_type = os.environ.get("KQA_BERTSCORE_MODEL", "roberta-large")
    device = os.environ.get("KQA_BERTSCORE_DEVICE")
    num_layers = os.environ.get("KQA_BERTSCORE_NUM_LAYERS")

    scorer_kwargs = {
        "model_type": model_type,
        "lang": "en",
        "rescale_with_baseline": True,
    }
    if device:
        scorer_kwargs["device"] = device
    if num_layers:
        scorer_kwargs["num_layers"] = int(num_layers)
    return BERTScorer(**scorer_kwargs)


def bert_score_f1(references, predictions) -> float:
    reference = _get_text(references)
    prediction = _get_text(predictions)

    if not isinstance(reference, str):
        reference = str(reference)
    if not isinstance(prediction, str):
        prediction = str(prediction)

    reference = _normalize(reference)
    prediction = _normalize(prediction)

    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
        return 0.0

    scorer = _get_scorer()
    _, _, f1 = scorer.score([prediction], [reference])
    return float(f1[0].item())
