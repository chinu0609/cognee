from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalConfig(BaseSettings):
    # Corpus builder params
    building_corpus_from_scratch: bool = True
    number_of_samples_in_corpus: int = 1
    benchmark: str = "Dummy"  # Options: 'HotPotQA', 'Dummy', 'TwoWikiMultiHop', 'BEAM'
    task_getter_type: str = (
        "Default"  # Options: 'Default', 'CascadeGraph', 'NoSummaries', 'JustChunks'
    )
    chunks_per_batch: int | None = None  # Override chunks_per_batch for the cognify pipeline

    # Question answering params
    answering_questions: bool = True
    qa_engine: str = (
        "cognee_graph_completion"  # See answer_generation.registry for supported strategies.
    )

    # Evaluation params
    evaluating_answers: bool = True
    evaluating_contexts: bool = True
    evaluation_engine: str = "DeepEval"  # Options: 'DeepEval', 'BeamEval', 'DirectLLM'
    evaluation_metrics: list[str] = [
        "correctness",
        "EM",
        "f1",
    ]  # Use only 'correctness' for DirectLLM
    deepeval_model: str = "gpt-4o-mini"

    # Metrics params
    calculate_metrics: bool = True

    # Visualization
    dashboard: bool = True

    # Reproducibility
    seed: int = 42  # Seed for deterministic corpus sampling across runs

    # Optional directory for run artifacts. When set, the runner namespaces
    # artifacts under "<results_dir>/<benchmark>_<engine>/" so successive runs
    # are comparable instead of overwriting each other. When unset, artifacts are
    # written to the current working directory (legacy behavior).
    results_dir: str | None = None

    # file paths
    questions_path: str = "questions_output.json"
    answers_path: str = "answers_output.json"
    metrics_path: str = "metrics_output.json"
    aggregate_metrics_path: str = "aggregate_metrics.json"
    dashboard_path: str = "dashboard.html"
    direct_llm_system_prompt: str = "direct_llm_eval_system.txt"
    direct_llm_eval_prompt: str = "direct_llm_eval_prompt.txt"
    instance_filter: list[str] | None = None

    # Cognee LLM + embedding used for corpus cognify and answer generation.
    # These mirror the main cognee env vars so the eval run's config.json records
    # exactly which models were in use — set them in .env as usual.
    llm_model: str = "openai/gpt-4o-mini"
    llm_provider: str = "openai"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Jev (TypeSafe AI) eval engine config
    jev_api_key: str | None = None
    jev_api_endpoint: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "building_corpus_from_scratch": self.building_corpus_from_scratch,
            "number_of_samples_in_corpus": self.number_of_samples_in_corpus,
            "benchmark": self.benchmark,
            "answering_questions": self.answering_questions,
            "qa_engine": self.qa_engine,
            "evaluating_answers": self.evaluating_answers,
            "evaluating_contexts": self.evaluating_contexts,  # Controls whether context evaluation should be performed
            "evaluation_engine": self.evaluation_engine,
            "evaluation_metrics": self.evaluation_metrics,
            "calculate_metrics": self.calculate_metrics,
            "dashboard": self.dashboard,
            "seed": self.seed,
            "results_dir": self.results_dir,
            "questions_path": self.questions_path,
            "answers_path": self.answers_path,
            "metrics_path": self.metrics_path,
            "aggregate_metrics_path": self.aggregate_metrics_path,
            "dashboard_path": self.dashboard_path,
            "deepeval_model": self.deepeval_model,
            "task_getter_type": self.task_getter_type,
            "chunks_per_batch": self.chunks_per_batch,
            "direct_llm_system_prompt": self.direct_llm_system_prompt,
            "direct_llm_eval_prompt": self.direct_llm_eval_prompt,
            "instance_filter": self.instance_filter,
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "jev_api_key": self.jev_api_key,
            "jev_api_endpoint": self.jev_api_endpoint,
            "jev_model": self.jev_model,
        }


@lru_cache
def get_llm_config():
    return EvalConfig()
