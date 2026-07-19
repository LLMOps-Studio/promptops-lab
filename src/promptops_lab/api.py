import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Assuming this is the core pipeline service based on standard lab structure
from promptops_lab.evaluation.evaluator_pipeline import PromptEvaluatorPipeline
from promptops_lab.agents.support_agent import SupportAgent

app = FastAPI(
    title="PromptOps Lab API",
    description="LLMOps laboratory for prompt versioning and evaluation.",
    version="0.1.0"
)

# Enable CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    prompt_version: str
    # Optional -- when omitted, behaves exactly as before (loads
    # prompts/{version}/support_agent.yaml from disk). When provided, this
    # raw template is used instead, under the given prompt_version label,
    # without needing a matching on-disk YAML -- this is what lets the UI
    # let someone hand-edit and try a prompt as "v3" (or any label) without
    # a deploy step.
    custom_template: Optional[str] = None

@app.get("/health")
def health_check():
    """Confirms the laboratory API is up and running."""
    return {"status": "healthy", "service": "promptops-lab"}

@app.get("/prompts/{version}", summary="Fetch an existing prompt template for editing")
def get_prompt_template(version: str) -> Dict[str, Any]:
    """Returns the on-disk template for a version (e.g. v1, v2) so the UI
    can load it into an editable textarea before running a tweaked variant
    through /evaluate as custom_template."""
    agent = SupportAgent()
    try:
        config = agent.load_prompt_config(version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No prompt template found for version '{version}'.")
    return {
        "version": version,
        "name": config.get("name", version),
        "description": config.get("description", ""),
        "template": config["template"],
    }

@app.post("/evaluate", summary="Evaluate a specific prompt version")
def evaluate_prompt(request: EvaluationRequest) -> Dict[str, Any]:
    """Runs the prompt evaluation pipeline for a given prompt version."""
    try:
        pipeline = PromptEvaluatorPipeline()
        # This will trigger the LLM calls and return the aggregated metrics
        results = pipeline.evaluate_version(
            prompt_version=request.prompt_version,
            custom_template=request.custom_template,
        )
        
        return {
            "status": "success",
            "prompt_version": request.prompt_version,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")