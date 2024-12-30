from dataclasses import dataclass
from typing import Dict, Union

import torch


@dataclass
class Strategy:
    """Base class for the GS densification strategy.

    This class is the base class that defines the interface for the GS
    densification strategy.
    """

    def check_sanity(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Union[Dict[str, torch.optim.Optimizer], torch.optim.Optimizer],
            fused: bool = False,
    ):
        """Sanity check for the parameters and optimizers."""
        trainable_params = set(
            [name for name, param in params.items() if param.requires_grad]
        )

        if fused:
            # For fused optimizer (e.g. fused_adam), there is a single optimizer managing all parameters
            if isinstance(optimizers, dict) and "fused" in optimizers:
                fused_optimizer = optimizers["fused"]
            elif isinstance(optimizers, torch.optim.Optimizer):
                fused_optimizer = optimizers
            else:
                raise ValueError(
                    "For fused_adam, optimizers must be a single optimizer instance "
                    "or a dictionary containing a 'fused' optimizer."
                )

            fused_params = set(
                id(p) for group in fused_optimizer.param_groups for p in group["params"]
            )
            missing_params = [name for name, param in params.items() if id(param) not in fused_params]
            assert not missing_params, (
                "Fused optimizer is missing parameters: "
                f"{missing_params}"
            )
        else:
            assert trainable_params == set(optimizers.keys()), (
                "Trainable parameters and optimizers must have the same keys, "
                f"but got {trainable_params} and {optimizers.keys()}"
            )

            for optimizer in optimizers.values():
                assert len(optimizer.param_groups) == 1, (
                    "Each optimizer must have exactly one param_group, "
                    "that corresponds to each parameter, "
                    f"but got {len(optimizer.param_groups)}"
                )

    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def step_post_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        pass
