# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cog predictor for GPT-NeoX text generation (https://replicate.com)."""

import os

from cog import BasePredictor, Input

from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.text_generation_utils import generate_samples_from_prompt


# Default paths expected inside the container / mounted volume.
_DEFAULT_CONFIG = os.environ.get("NEOX_CONFIG", "configs/125M.yml")
_DEFAULT_CHECKPOINT = os.environ.get("NEOX_CHECKPOINT", "checkpoints")


class Predictor(BasePredictor):
    """Runs GPT-NeoX text generation via Cog / Replicate."""

    def setup(self) -> None:
        """Load the model into memory so that multiple predict() calls are fast."""
        overwrite_values = {
            "checkpoint_activations": False,
            "partition_activations": False,
            "no_load_optim": True,
            "optimizer": None,
            "zero_optimization": None,
            "load": _DEFAULT_CHECKPOINT,
            "text_gen_type": "input-file",
        }

        neox_args = NeoXArgs.from_ymls(
            paths_to_yml_files=[_DEFAULT_CONFIG],
            overwrite_values=overwrite_values,
        )
        neox_args.configure_distributed_args()
        neox_args.build_tokenizer()

        initialize_megatron(neox_args)

        model, _, _, _ = setup_model_and_optimizer(
            neox_args=neox_args,
            use_cache=True,
            iteration=neox_args.iteration,
        )
        model.module.inference_mode(use_cache=True)

        self.model = model
        self.neox_args = neox_args

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model."),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate.",
            default=128,
            ge=1,
            le=2048,
        ),
        temperature: float = Input(
            description="Sampling temperature. 0 means greedy decoding.",
            default=0.9,
            ge=0.0,
            le=2.0,
        ),
        top_k: int = Input(
            description="Top-k sampling: keep only the k most likely next tokens. 0 disables.",
            default=0,
            ge=0,
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling probability mass. 0 disables.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
    ) -> str:
        """Run a single text-generation request and return the completion."""
        results = generate_samples_from_prompt(
            neox_args=self.neox_args,
            model=self.model,
            text=prompt,
            maximum_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if not results:
            return ""

        return results[0].get("text", "")
