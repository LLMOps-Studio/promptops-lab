import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Assuming this is the core pipeline service based on standard lab structure
from promptops_lab.evaluation.evaluator_pipeline import PromptEvaluatorPipeline 

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

@app.get("/health")
def health_check():
    """Confirms the laboratory API is up and running."""
    return {"status": "healthy", "service": "promptops-lab"}

@app.post("/evaluate", summary="Evaluate a specific prompt version")
def evaluate_prompt(request: EvaluationRequest) -> Dict[str, Any]:
    """Runs the prompt evaluation pipeline for a given prompt version."""
    try:
        pipeline = PromptEvaluatorPipeline()
        # This will trigger the LLM calls and return the aggregated metrics
        results = pipeline.evaluate_version(prompt_version=request.prompt_version)
        
        return {
            "status": "success",
            "prompt_version": request.prompt_version,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")