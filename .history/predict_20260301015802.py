"""Prediction interface for Cog / Replicate."""

from cog import BasePredictor, Input
from megatron.utils import setup_for_inference_or_eval
from megatron.text_generation_utils import generate_samples_from_prompt


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory."""
        # ── CONFIGURE THESE ──────────────────────────────────────────────
        # Point `load` to your checkpoint directory and list the YAML
        # config files that describe the model architecture + tokenizer.
        #
        # Example:
        #   config_files = ["configs/125M.yml", "configs/local_setup.yml"]
        #   checkpoint_path = "/path/to/your/checkpoint"
        # ─────────────────────────────────────────────────────────────────

        config_files = [
            "configs/125M.yml",       # <-- replace with your model config
            "configs/local_setup.yml",
        ]
        checkpoint_path = "/src/checkpoints"  # <-- replace with your checkpoint path

        cli_args = []
        for cfg in config_files:
            cli_args.extend(["-c", cfg])

        self.model, self.neox_args = setup_for_inference_or_eval(
            use_cache=True,
            overwrite_values={"load": checkpoint_path},
            input_args=cli_args,
        )

    def predict(
        self,
        prompt: str = Input(description="Input text prompt", default="Once upon a time"),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=128,
            ge=1,
            le=2048,
        ),
        temperature: float = Input(
            description="Sampling temperature (higher = more random)",
            default=0.8,
            ge=0.01,
            le=2.0,
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
        top_k: int = Input(
            description="Top-k sampling (0 = disabled)",
            default=0,
            ge=0,
            le=1000,
        ),
    ) -> str:
        """Run a single prediction on the model."""
        output = generate_samples_from_prompt(
            neox_args=self.neox_args,
            model=self.model,
            text=prompt,
            recompute=False,
            temperature=temperature,
            maximum_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
        )

        # generate_samples_from_prompt returns a list of dicts with "text" keys
        if output and isinstance(output, list):
            return output[0].get("text", "")
        return str(output)
