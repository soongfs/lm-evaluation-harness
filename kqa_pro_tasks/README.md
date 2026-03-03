# KQA-Pro Tasks

Root directory for local KQA-Pro task definitions.

Layout:

- `kqa_pro_mcq/`: base multiple-choice task
- `kqa_pro_mcq_instruct/`: instruct-style generative multiple-choice task
- `kqa_pro_qa/`: free-form QA task

Use them with:

```bash
--include_path ./kqa_pro_tasks
```

Each task directory is self-contained and loads local data from `dataset/train.json` and `dataset/val.json`.
