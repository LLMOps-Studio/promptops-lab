import json
import time
import mlflow
from pathlib import Path
from typing import Dict, Optional

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

# Importing from the shared platform SDK
from llmops_common.eval.evaluator import LLMEvaluator
from llmops_common.logging.mlflow_logger import MLflowLogger
from llmops_common.client.factory import get_llm_client
from llmops_common.stats.proportions import wilson_score_interval

from promptops_lab.agents.support_agent import SupportAgent
from promptops_lab.evaluation.latency import compute_latency_stats

class CustomDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, custom_client):
        self.custom_client = custom_client

    def load_model(self):
        return self.custom_client

    def generate(self, prompt: str) -> str:
        return self.custom_client.generate(prompt=prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.custom_client.generate(prompt=prompt)

    def get_model_name(self):
        return "llmops-common-local-model"

class PromptEvaluatorPipeline:
    def __init__(self, dataset_path: Path = None):
        if dataset_path is None:
            self.dataset_path = Path(__file__).parent.parent.parent.parent / "datasets" / "golden" / "golden_dataset.json"
        else:
            self.dataset_path = Path(dataset_path)
            
        self.agent = SupportAgent()
        
        llm_client = get_llm_client()
        self.llm_evaluator = LLMEvaluator(client=llm_client)
        self.mlflow_logger = MLflowLogger(experiment_name="promptops")
        
        deep_eval_model = CustomDeepEvalModel(custom_client=llm_client)
        
        self.hallucination_metric = HallucinationMetric(
            threshold=0.5,
            model=deep_eval_model
        )

    def load_dataset(self) -> list:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _check_determinism(
        self,
        prompt_version: str,
        context: str,
        question: str,
        custom_template: Optional[str],
        n: int = 5,
    ) -> Dict[str, float]:
        """
        Re-runs the same prompt/context/question `n` times and measures how
        often the response comes back identical -- a check on how
        deterministic the model's output actually is in practice.

        Run once per evaluate_version() call, on one representative item,
        not once per dataset item -- an n-way re-run for every item would
        multiply the whole run's LLM-call cost by n.
        """
        responses = [
            self.agent.execute(
                version=prompt_version,
                context=context,
                question=question,
                custom_template=custom_template,
            )
            for _ in range(n)
        ]

        baseline = responses[0]
        matches = sum(1 for response in responses if response == baseline)
        ci = wilson_score_interval(matches, n)

        return {
            "determinism_matches": matches,
            "determinism_n": n,
            "determinism_rate": matches / n,
            "determinism_rate_ci_lower": ci.lower,
            "determinism_rate_ci_upper": ci.upper,
        }

    def evaluate_version(self, prompt_version: str, custom_template: Optional[str] = None) -> Dict[str, float]:
        dataset = self.load_dataset()
        
        self.mlflow_logger.start_trace(run_name=f"eval_{prompt_version}")
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("is_custom_prompt", custom_template is not None)
        
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_hallucination = 0.0
        latencies_seconds = []

        for idx, item in enumerate(dataset):
            context = item["context"]
            question = item["question"]

            start = time.perf_counter()
            response = self.agent.execute(
                version=prompt_version,
                context=context,
                question=question,
                custom_template=custom_template,
            )
            latencies_seconds.append(time.perf_counter() - start)

            # --- DÜZELTME BURADA: Ayrı ayrı fonksiyonları çağırıyoruz ---
            f_score = self.llm_evaluator.evaluate_faithfulness(
                context=context, 
                response=response
            )
            r_score = self.llm_evaluator.evaluate_relevance(
                query=question, 
                response=response
            )
            
            test_case = LLMTestCase(
                input=question,
                actual_output=response,
                context=[context]
            )
            self.hallucination_metric.measure(test_case)
            h_score = self.hallucination_metric.score
            
            total_faithfulness += f_score
            total_relevance += r_score
            total_hallucination += h_score
            
            self.mlflow_logger.log_llm_call(
                model_name="local-llm", 
                prompt=f"[{prompt_version}] {question}",
                response=response
            )
            
            mlflow.log_metrics({
                "faithfulness": f_score,
                "relevance": r_score,
                "hallucination": h_score
            }, step=idx)
            
        num_cases = len(dataset)
        latency_stats = compute_latency_stats(latencies_seconds)
        determinism = self._check_determinism(
            prompt_version,
            dataset[0]["context"],
            dataset[0]["question"],
            custom_template,
        )

        avg_metrics = {
            "avg_faithfulness": total_faithfulness / num_cases,
            "avg_relevance": total_relevance / num_cases,
            "avg_hallucination": total_hallucination / num_cases,
            "latency_p50_seconds": latency_stats.p50,
            "latency_p95_seconds": latency_stats.p95,
            "latency_p99_seconds": latency_stats.p99,
            **determinism,
            # FIX (Faz 0.2): previously omitted -> PromptComparisonNode always
            # received empty context/response, same contract bug as the RAG
            # benchmark node. We surface the last dataset item's context and
            # generated response so downstream nodes (e.g. the LLM-judge
            # scorer) have real content to evaluate.
            "context": context,
            "response": response
        }

        mlflow.log_metrics(avg_metrics)
        self.mlflow_logger.end_trace()
        
        return avg_metrics