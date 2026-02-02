from pathlib import Path

from backend.configs.enums import PromptType, ModelRoleType
from backend.utils.common_funcs import read_file


class PromptEngine:
    """Prompt engineering helper class."""

    def __init__(self, prompt_dir: Path = Path("./prompts")):
        self.prompt_dir = prompt_dir

    def render(
            self,
            prompt_type: PromptType,
            model_role: ModelRoleType,
            version: str,
            model_function: str = ""
    ) -> tuple[str, dict]:
        """Performs prompt rendering.

        Args:
            prompt_type: The prompt type = feature to render (e.g. "learner_workflow").
            model_role: The model role = agent specialization.
            version: The version of the prompt (e.g. "v1").
            model_function: A specific function that is defined under the given role (e.g. Orchestrator can participate
                in the Routing mechanism and in Graph Index constructing process).

        Returns:
            tuple[str, dict, dict]: The formatted prompt string, system settings and model settings.
        """
        data = read_file(
            self.prompt_dir / str(prompt_type.value) / model_role.name / model_function / f"{version}.yaml"
        )
        user_prompt = data.get("user", "")
        system_prompt = data.get("system", {})
        return user_prompt, system_prompt
