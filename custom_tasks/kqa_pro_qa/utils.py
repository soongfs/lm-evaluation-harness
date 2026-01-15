import re
from collections import Counter


_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    text = _normalize(text)
    return text.split() if text else []


def _get_text(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def _first_segment(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text.split("\n\n", 1)[0]
    text = text.split("\n", 1)[0]
    return text.strip()


def _token_f1_score(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(references, predictions) -> float:
    gold = _get_text(references)
    pred = _first_segment(_get_text(predictions))
    return _token_f1_score(pred, gold)


def fuzzy_acc(references, predictions, threshold: float = 0.5) -> float:
    gold = _get_text(references)
    pred = _first_segment(_get_text(predictions))

    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred)
    if gold_norm == pred_norm:
        return 1.0

    return 1.0 if _token_f1_score(pred, gold) >= threshold else 0.0


def exact_match_first_segment(references, predictions) -> float:
    gold = _get_text(references).strip()
    pred = _first_segment(_get_text(predictions))
    return 1.0 if pred == gold else 0.0
