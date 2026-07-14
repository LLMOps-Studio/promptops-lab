import argparse
from promptops_lab.evaluation.evaluator_pipeline import PromptEvaluatorPipeline

def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation for a specific prompt version.")
    parser.add_argument("--version", type=str, required=True, help="Prompt version to evaluate (e.g., v1, v2)")
    args = parser.parse_args()

    print(f"🚀 Starting evaluation for prompt version: {args.version}")
    pipeline = PromptEvaluatorPipeline()
    
    try:
        results = pipeline.evaluate_version(prompt_version=args.version)
        print("\n✅ Evaluation Completed. Aggregate Metrics:")
        for metric, score in results.items():
            print(f"  - {metric}: {score:.3f}")
        print(f"\n📊 Detailed results are logged to MLflow under the 'promptops' experiment.")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()