import json
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

from promptops_lab.agents.support_agent import SupportAgent

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

    def evaluate_version(self, prompt_version: str, custom_template: Optional[str] = None) -> Dict[str, float]:
        dataset = self.load_dataset()
        
        self.mlflow_logger.start_trace(run_name=f"eval_{prompt_version}")
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("is_custom_prompt", custom_template is not None)
        
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_hallucination = 0.0
        
        for idx, item in enumerate(dataset):
            context = item["context"]
            question = item["question"]
            
            response = self.agent.execute(
                version=prompt_version,
                context=context,
                question=question,
                custom_template=custom_template,
            )
            
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
        avg_metrics = {
            "avg_faithfulness": total_faithfulness / num_cases,
            "avg_relevance": total_relevance / num_cases,
            "avg_hallucination": total_hallucination / num_cases,
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