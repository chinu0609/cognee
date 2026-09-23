"""Eval adapter that uses TypeSafe AI's Jev model as a judge.

Jev is a System One model: unstructured text in, typed probabilistic decisions out.
It returns calibrated confidence scores within a predefined type schema — cannot
hallucinate outside the schema.

API: POST https://api.typesafe.ai/v1/systemone
  state    — the content to evaluate (string)
  model    — "jev-latest"
  questions — map of typed Question objects (noul = yes/no, score = rating)

Noul question returns a probability 0–1 (> 0.5 = yes).

This adapter feeds Jev the full LLM input cognee assembled (captured via
``only_context=True``) so it judges correctness against what cognee actually
retrieved, not just the question and golden answer.

Required env vars:
    JEV_API_KEY      — TypeSafe AI API key
    JEV_API_ENDPOINT — defaults to https://api.typesafe.ai/v1/systemone
    JEV_MODEL        — defaults to "jev-latest"
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger("JevEvalAdapter")

_STATE_TEMPLATE_WITH_CONTEXT = """\
EXPECTED ANSWER:
{golden_answer}

SYSTEM ANSWER:
{answer}

RETRIEVAL CONTEXT (what the system retrieved before answering, including the question):
{retrieval_context}

TASK FRAMING (instructions given to the system):
{task_context}
"""

_STATE_TEMPLATE_NO_CONTEXT = """\
QUESTION:
{question}

EXPECTED ANSWER:
{golden_answer}

SYSTEM ANSWER:
{answer}
"""

_QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Is the system answer factually correct given the question and expected answer? "
            "Consider the retrieval context to judge whether the answer is grounded."
        ),
    },
    "grounded": {
        "type": "noul",
        "instructions": (
            "Is the system answer grounded in the retrieval context, "
            "or does it introduce facts not present in the context?"
        ),
    },
}


class JevEvalAdapter(BaseEvalAdapter):
    def __init__(self):
        config = EvalConfig()
        if not config.jev_api_key:
            raise ValueError(
                "JEV_API_KEY is not set. Add it to your .env before using the Jev engine."
            )
        self.api_key: str = config.jev_api_key
        self.endpoint: str = config.jev_api_endpoint
        self.model: str = config.jev_model

    async def _call_jev_with_questions(
        self, state: str, questions: dict, *, max_retries: int = 3
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        import asyncio

        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.endpoint, json=payload, headers=headers)
                if response.status_code in (429, 529):
                    wait = 2 ** attempt
                    logger.warning(
                        "Jev rate limit (attempt %d/%d), retrying in %ds",
                        attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
        response.raise_for_status()  # raise after exhausting retries
        return response.json()  # unreachable, keeps type checker happy

    async def evaluate_correctness(
        self,
        question: str,
        answer: str,
        golden_answer: str,
        jev_user_prompt: str | None = None,
        jev_system_prompt: str | None = None,
    ) -> dict[str, Any]:
        has_context = bool(jev_user_prompt)

        if has_context:
            state = _STATE_TEMPLATE_WITH_CONTEXT.format(
                question=question,
                golden_answer=golden_answer,
                answer=answer,
                retrieval_context=jev_user_prompt,
                task_context=jev_system_prompt or "(not available)",
            )
            questions = _QUESTIONS
        else:
            state = _STATE_TEMPLATE_NO_CONTEXT.format(
                question=question,
                golden_answer=golden_answer,
                answer=answer,
            )
            # No retrieval context — grounding question is meaningless, skip it
            questions = {"correct": _QUESTIONS["correct"]}

        try:
            result = await self._call_jev_with_questions(state, questions)
        except Exception as exc:
            logger.warning("Jev API call failed: %s", exc)
            return {"score": None, "correct": None, "grounded": None, "reason": str(exc)}

        answers = result.get("answers", {})

        # noul returns probability 0–1 = P(yes); use it directly as score
        correct_prob: float = answers.get("correct", {}).get("noul", 0.5)
        grounded_prob: float | None = (
            answers.get("grounded", {}).get("noul") if has_context else None
        )

        return {
            "score": round(correct_prob, 4),  # P(correct) ∈ [0, 1]
            "correct": correct_prob > 0.5,
            "correct_prob": round(correct_prob, 4),
            "grounded": (grounded_prob > 0.5) if grounded_prob is not None else None,
            "grounded_prob": round(grounded_prob, 4) if grounded_prob is not None else None,
            "reason": (
                f"correct_prob={correct_prob:.3f}"
                + (f", grounded_prob={grounded_prob:.3f}" if grounded_prob is not None else "")
            ),
        }

    async def evaluate_answers(
        self, answers: list[dict[str, Any]], evaluator_metrics: list[str]
    ) -> list[dict[str, Any]]:
        results = []
        for answer in answers:
            metric_result = await self.evaluate_correctness(
                question=answer["question"],
                answer=answer["answer"],
                golden_answer=answer["golden_answer"],
                jev_user_prompt=answer.get("jev_user_prompt"),
                jev_system_prompt=answer.get("jev_system_prompt"),
            )
            results.append({**answer, "metrics": {"jev_correctness": metric_result}})
        return results
