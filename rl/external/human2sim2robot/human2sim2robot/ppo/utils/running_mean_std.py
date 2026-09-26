from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from human2sim2robot.ppo.utils import torch_ext

"""
Updates statistics from full data.
"""


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        insize: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-05,
        per_channel: bool = False,
        norm_only: bool = False,
    ) -> None:
        super(RunningMeanStd, self).__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(
        self,
        input: torch.Tensor,
        denorm: bool = False,
        mask: Optional[torch.Tensor] = None,
        update_stats: Optional[bool] = None,
    ) -> torch.Tensor:
        should_update = self.training if update_stats is None else bool(update_stats)
        if should_update:
            self.update(input, mask=mask)

        # Change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(
                    input
                )
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(
                    input
                )
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # Get output

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

    @torch.no_grad()
    def update(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Update statistics explicitly without coupling mutation to inference."""
        if mask is not None:
            mean, var = torch_ext.get_mean_std_with_masks(input, mask)
        else:
            mean = input.mean(self.axis)
            var = input.var(self.axis)
        self.running_mean, self.running_var, self.count = (
            self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )
        )


class RunningMeanStdObs(nn.Module):
    def __init__(
        self,
        insize: Dict[str, Union[int, Tuple[int, ...]]],
        epsilon: float = 1e-05,
        per_channel: bool = False,
        norm_only: bool = False,
    ) -> None:
        assert isinstance(insize, dict)
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict(
            {
                k: RunningMeanStd(v, epsilon, per_channel, norm_only)
                for k, v in insize.items()
            }
        )

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        denorm: bool = False,
        update_stats: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        res = {
            k: self.running_mean_std[k](
                v, denorm, update_stats=update_stats
            )
            for k, v in input.items()
        }
        return res

    @torch.no_grad()
    def update(self, input: Dict[str, torch.Tensor]) -> None:
        for key, value in input.items():
            self.running_mean_std[key].update(value)
