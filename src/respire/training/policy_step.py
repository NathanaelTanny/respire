"""One constrained policy step.

Three terms share one optimiser: the reweighted flow-matching objective, the critic's return at the
policy's own samples, and the multiplier-weighted cost excess. After the step the estimated KL
against the previous iterate is compared with the radius and the step is backtracked if it does not
fit, so the trust region is a constraint on the update rather than a number in a log.

Ref: Eq. (4) (maximise the certified value, minus beta KL, subject to costs and KL <= B).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.losses.flow_matching import flow_matching_loss
from respire.losses.trust_region import backtrack, estimate_kl, snapshot
from respire.models.critic import TwinCritic
from respire.models.deterministic_head import DeterministicHead, deterministic_regulariser
from respire.models.flow import FlowPolicy
from respire.settings import RoundConfig, TheatreConfig
from respire.utils.autograd import backward
from respire.utils.logging_setup import get_logger

LOGGER = get_logger("training.policy")


@dataclass(frozen=True)
class PolicyStepReport:
    loss: float
    flow_loss: float
    return_term: float
    cost_term: float
    kl: float
    backtracks: int
    mean_propofol: float
    mean_remifentanil: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "flow_loss": self.flow_loss,
            "return_term": self.return_term,
            "cost_term": self.cost_term,
            "kl": self.kl,
            "backtracks": self.backtracks,
            "mean_propofol": self.mean_propofol,
            "mean_remifentanil": self.mean_remifentanil,
        }


def constrained_step(
    flow: FlowPolicy | DeterministicHead,
    reference: FlowPolicy | DeterministicHead,
    critic: TwinCritic,
    condition: Tensor,
    target_action: Tensor,
    weights: Tensor,
    multipliers: Tensor,
    budgets: Tensor,
    cfg: RoundConfig,
    theatre: TheatreConfig,
    seed: int,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    actor_weight: float = 1.0,
) -> PolicyStepReport:
    """Take the operator's update step and enforce the radius."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    previous_state = snapshot(flow)
    optimiser = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    last_flow = 0.0
    last_return = 0.0
    last_cost = 0.0
    last_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(condition.shape[0], generator=generator)
        for start in range(0, condition.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            chunk_condition = condition[index]
            chunk_action = target_action[index]
            chunk_weights = weights[index]
            optimiser.zero_grad()
            if isinstance(flow, FlowPolicy):
                matching = flow_matching_loss(
                    flow, chunk_condition, chunk_action, chunk_weights, generator
                )
            else:
                matching = deterministic_regulariser(
                    flow.mean_action(chunk_condition), chunk_action
                )
            sampled = flow.sample_pathwise(chunk_condition)
            clipped = torch.stack(
                [
                    sampled[:, 0].clamp(0.0, theatre.propofol_max),
                    sampled[:, 1].clamp(0.0, theatre.remifentanil_max),
                ],
                dim=-1,
            )
            estimates = critic.conservative(torch.cat([chunk_condition, clipped], dim=-1))
            raw = critic.unscale_targets(estimates)
            return_term = -estimates[:, 0].mean()
            cost_excess = raw[:, 1:] / budgets.clamp(min=1e-6).unsqueeze(0) - 1.0
            cost_term = (multipliers[:3].unsqueeze(0) * cost_excess).sum(dim=-1).mean()
            loss = matching + actor_weight * (return_term + cost_term)
            backward(loss)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimiser.step()
            last_flow = float(matching)
            last_return = float(return_term)
            last_cost = float(cost_term)
            last_loss = float(loss)
    if isinstance(flow, FlowPolicy) and isinstance(reference, FlowPolicy):
        kl, used = backtrack(
            flow,
            reference,
            previous_state,
            condition,
            cfg.trust_region,
            min(256, condition.shape[0]),
            cfg.backtrack_steps,
            generator,
        )
    else:
        kl, used = 0.0, 0
    LOGGER.info("policy step loss %.4f kl %.4f backtracks %d", last_loss, kl, used)
    with torch.no_grad():
        probe = flow.sample_pathwise(condition[: min(256, condition.shape[0])])
    return PolicyStepReport(
        loss=last_loss,
        flow_loss=last_flow,
        return_term=last_return,
        cost_term=last_cost,
        kl=kl,
        backtracks=used,
        mean_propofol=float(probe[:, 0].mean()),
        mean_remifentanil=float(probe[:, 1].mean()),
    )


def reference_copy(head: FlowPolicy | DeterministicHead) -> FlowPolicy | DeterministicHead:
    """Frozen copy of the current policy, the reference of the next trust region."""
    if isinstance(head, DeterministicHead):
        twin = DeterministicHead(head.cfg, head.cond_dim)
        twin.load_state_dict(head.state_dict())
        twin.eval()
        return twin
    clone = FlowPolicy(head.cfg, head.cond_dim)
    clone.load_state_dict(head.state_dict())
    clone.eval()
    return clone


def kl_report(updated: FlowPolicy, reference: FlowPolicy, condition: Tensor) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return estimate_kl(updated, reference, condition, min(128, condition.shape[0]), generator)
