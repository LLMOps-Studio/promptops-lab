import json
import pytest
from pathlib import Path
from unittest.mock import patch, PropertyMock
from promptops_lab.evaluation.evaluator_pipeline import PromptEvaluatorPipeline

@pytest.fixture
def mock_responses():
    """Loads cached LLM responses for CI testing without GPU."""
    fixture_path = Path(__file__).parent / "fixtures" / "cached_responses.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)

@patch("promptops_lab.evaluation.evaluator_pipeline.mlflow")
@patch("promptops_lab.evaluation.evaluator_pipeline.MLflowLogger")
@patch("promptops_lab.evaluation.evaluator_pipeline.SupportAgent.execute")
@patch("promptops_lab.evaluation.evaluator_pipeline.LLMEvaluator.evaluate_faithfulness")
@patch("promptops_lab.evaluation.evaluator_pipeline.LLMEvaluator.evaluate_relevance")
@patch("promptops_lab.evaluation.evaluator_pipeline.HallucinationMetric.measure")
@patch("promptops_lab.evaluation.evaluator_pipeline.HallucinationMetric.score", new_callable=PropertyMock)
def test_prompt_regression(
    mock_score, mock_measure, mock_relevance, mock_faithfulness, 
    mock_execute, mock_mlflow_logger, mock_mlflow, mock_responses
):
    """
    Ensures that the new prompt version (v2) does not degrade performance 
    compared to the baseline (v1).
    """
    pipeline = PromptEvaluatorPipeline()
    
    mock_execute.side_effect = lambda version, context, question, custom_template=None: mock_responses["support_agent_v1"]["case_01"]
    mock_faithfulness.return_value = 0.90
    mock_relevance.return_value = 0.85
    mock_score.return_value = 0.15 
    
    baseline_metrics = pipeline.evaluate_version("v1")
    
    mock_execute.side_effect = lambda version, context, question, custom_template=None: mock_responses["support_agent_v2"]["case_01"]
    mock_faithfulness.return_value = 0.95
    mock_relevance.return_value = 0.95
    mock_score.return_value = 0.05 
    
    new_metrics = pipeline.evaluate_version("v2")
    
    assert new_metrics["avg_faithfulness"] >= (baseline_metrics["avg_faithfulness"] * 0.9), \
        "Regression detected: Faithfulness dropped significantly in v2!"
        
    assert new_metrics["avg_hallucination"] <= baseline_metrics["avg_hallucination"], \
        "Regression detected: Hallucination rate increased in v2!"