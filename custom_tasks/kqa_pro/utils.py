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
