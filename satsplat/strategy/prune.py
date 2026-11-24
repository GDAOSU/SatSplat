from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import remove


@dataclass
class PruneStrategy(Strategy):
    """Strategy that prunes Gaussian splats based on their opacity."""

    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    prune_opa: float = 0.005
    verbose: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        return {}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        if step < self.refine_stop_iter and step > self.refine_start_iter and step % self.refine_every == 0:
            transparent_mask = torch.sigmoid(params["opacities"]) < self.prune_opa
            if transparent_mask.any():
                remove(params, optimizers, state, transparent_mask)
                if self.verbose:
                    print(
                        f"Step {step}: {transparent_mask.sum().item()} GSs pruned. "
                        f"Now having {len(params['means'])} GSs."
                    )
