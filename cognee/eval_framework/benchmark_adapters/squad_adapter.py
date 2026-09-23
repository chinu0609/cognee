"""SQuAD v2 benchmark adapter.

Downloads directly from Stanford NLP (no extra dependencies, English only).
Follows the same adapter contract as HotpotQA / TwoWikiMultiHop:
  _get_raw_corpus  → one item per answerable QA pair (flattened from article→paragraph→qa)
  _get_corpus_entries(item) → the paragraph context for that QA pair
  _get_question_answer_pair(item) → the question + answer

limit in load_corpus samples N items → N passages + N questions, keeping corpus and
questions in sync without ingesting the full 1204-passage dev set.

Dataset: https://rajpurkar.github.io/SQuAD-explorer/
"""

import json
import os
import random
from typing import Any

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter


class SQuADAdapter(BaseBenchmarkAdapter):
    dataset_info = {
        "filename": "squad_dev_v2.json",
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    }

    def _get_raw_corpus(self) -> list[dict[str, Any]]:
        """Return one flat item per answerable QA pair across all articles."""
        filename = self.dataset_info["filename"]

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            import requests

            response = requests.get(self.dataset_info["url"], timeout=30)
            response.raise_for_status()
            raw = response.json()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)

        items = []
        for article in raw["data"]:
            for para in article.get("paragraphs", []):
                for qa in para["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    answers = qa.get("answers", [])
                    if not answers:
                        continue
                    items.append({
                        "id": qa["id"],
                        "question": qa["question"],
                        "answer": answers[0]["text"],
                        "context": para["context"],
                    })
        return items

    def _get_corpus_entries(self, item: dict[str, Any]) -> list[str]:
        return [item["context"]]

    def _get_question_answer_pair(
        self,
        item: dict[str, Any],
        load_golden_context: bool = False,
    ) -> dict[str, Any]:
        pair: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"].lower(),
        }
        if load_golden_context:
            pair["golden_context"] = item["context"]
        return pair

    def load_corpus(
        self,
        limit: int | None = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: str | list[str] | list[int] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        raw_corpus = self._get_raw_corpus()

        if instance_filter is not None:
            raw_corpus = self._filter_instances(raw_corpus, instance_filter, id_key="id")

        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = []
        question_answer_pairs = []
        for item in raw_corpus:
            corpus_list.extend(self._get_corpus_entries(item))
            question_answer_pairs.append(self._get_question_answer_pair(item, load_golden_context))

        return corpus_list, question_answer_pairs
