import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from llmops_common.client.factory import get_llm_client

class SupportAgent:
    """
    SupportAgent manages loading prompt templates from specific versions (v1, v2, etc.)
    and invoking the underlying LLM provider from llmops-common.
    """
    def __init__(self, prompts_dir: Path = None):
        if prompts_dir is None:
            # Resolves to src/promptops_lab/prompts
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)

    def load_prompt_config(self, version: str) -> Dict[str, Any]:
        """
        Loads prompt config and template string from the given version directory.
        """
        prompt_file = self.prompts_dir / version / "support_agent.yaml"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt configuration file not found at: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def execute(
        self,
        version: str,
        context: str,
        question: str,
        custom_template: Optional[str] = None,
    ) -> str:
        """
        Loads the specified prompt version, binds variables, and executes inference
        using the shared llmops-common LLM client.

        If custom_template is provided, it's used directly instead of loading
        version/support_agent.yaml from disk -- this is what lets the UI try
        an arbitrary, hand-edited prompt under any label (v1, v2, v3,
        "experiment-a", ...) without first creating a YAML file for it.
        custom_template must still contain the {context} and {question}
        placeholders; a missing placeholder raises KeyError from .format(),
        same as it always has for on-disk templates.
        """
        template_str = custom_template if custom_template is not None else self.load_prompt_config(version)["template"]

        # Validate and format input variables
        formatted_prompt = template_str.format(context=context, question=question)

        # Retrieve the central LLM client from the shared common platform SDK
        llm_client = get_llm_client()

        # Invoke the model execution
        response = llm_client.generate(prompt=formatted_prompt)
        return response