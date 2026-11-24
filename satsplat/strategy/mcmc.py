from typing import Any, Dict, Union

import torch

import gsplat.strategy


class MCMCStrategy(gsplat.strategy.MCMCStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        lr = optimizers["means"].param_groups[0]["lr"]
        super().step_post_backward(params, optimizers, state, step, info, lr=lr)
