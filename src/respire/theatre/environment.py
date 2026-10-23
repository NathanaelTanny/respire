"""Vectorised operating theatre.

Every episode is one patient on one surgical schedule. The batch runs in lockstep, so a policy
sees the same decision cadence it will see at the bedside: one action per five seconds, twelve
channels per step, and no access to the effect-site concentrations or to the latent axes.

Ref: Sec. 3.1 (5-s decisions, 12 channels, cost definitions), Eq. (1)-(2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from respire.settings import PharmacodynamicConfig, SurgicalConfig, TheatreConfig
from respire.theatre import pharmacodynamics as pd
from respire.theatre import pharmacokinetics as pk
from respire.theatre.patient import Patient
from respire.theatre.surgical import Schedule, build_schedule

CHANNELS: tuple[str, ...] = (
    "bis",
    "heart_rate",
    "map",
    "systolic",
    "diastolic",
    "spo2",
    "etco2",
    "temperature",
    "train_of_four",
    "dopamine",
    "propranolol",
    "pneumo_pressure",
)

NOISE_SD: tuple[float, ...] = (2.1, 2.0, 1.5, 2.0, 2.0, 0.5, 0.6, 0.03, 0.08, 0.0, 0.0, 0.2)


@dataclass(frozen=True)
class Observation:
    """The controller's view: a 60-s window, the matching action history and the current step."""

    window: Tensor
    actions: Tensor
    current: Tensor
    bis: Tensor
    map_mmHg: Tensor

    @property
    def batch(self) -> int:
        return int(self.window.shape[0])


@dataclass(frozen=True)
class Transition:
    """One decision interval of the batch."""

    observation: Observation
    action: Tensor
    reward: Tensor
    cost_hypotension: Tensor
    cost_suppression: Tensor
    cost_variability: Tensor
    done: bool
    propofol_ce: Tensor
    remifentanil_ce: Tensor
    true_map: Tensor
    true_bis: Tensor


@dataclass(frozen=True)
class ProbeStep:
    """Outcome of a counterfactual one-step look-ahead."""

    observed_bis: Tensor
    observed_map: Tensor
    true_bis: Tensor
    true_map: Tensor


@dataclass
class EnvState:
    """Mutable theatre state; copied on reset so a driver can replay a branch."""

    propofol: Tensor
    remifentanil: Tensor
    vitals: pd.Vitals
    step: int
    previous_action: Tensor
    cost_hypotension: Tensor
    cost_suppression: Tensor
    cost_variability: Tensor
    episode_return: Tensor


class AnaesthesiaEnv:
    """A batch of robot-assisted cases sharing one decision clock."""

    def __init__(
        self,
        patient: Patient,
        theatre: TheatreConfig,
        pd_cfg: PharmacodynamicConfig,
        schedule: Schedule | None = None,
        seed: int = 0,
    ) -> None:
        self.patient = patient
        self.theatre = theatre
        self.pd_cfg = pd_cfg
        self.schedule = (
            schedule
            if schedule is not None
            else build_schedule(len(patient), theatre.n_steps, theatre.dt_s, SurgicalConfig(), seed)
        )
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self._noise = torch.tensor(NOISE_SD, dtype=torch.float32)
        self._dt_min = theatre.dt_s / 60.0
        self._gamma_powers = torch.pow(
            torch.tensor(theatre.gamma, dtype=torch.float32), torch.arange(theatre.n_steps + 1)
        )
        self.state = self._fresh_state()
        self._action_window = torch.zeros(
            len(patient), self._window_steps(), 2, dtype=torch.float32
        )
        self._window = self._observe().unsqueeze(1).repeat(1, self._window_steps(), 1)

    @property
    def batch(self) -> int:
        return len(self.patient)

    @property
    def steps(self) -> int:
        return self.theatre.n_steps

    def _window_steps(self) -> int:
        return int(round(60.0 / self.theatre.dt_s))

    def _fresh_state(self) -> EnvState:
        batch = self.batch
        return EnvState(
            propofol=pk.initial_state(batch),
            remifentanil=pk.initial_state(batch),
            vitals=pd.initial_vitals(
                self.patient.covariates, self.patient.haemodynamic, self.pd_cfg
            ),
            step=0,
            previous_action=torch.zeros(batch, 2, dtype=torch.float32),
            cost_hypotension=torch.zeros(batch, dtype=torch.float32),
            cost_suppression=torch.zeros(batch, dtype=torch.float32),
            cost_variability=torch.zeros(batch, dtype=torch.float32),
            episode_return=torch.zeros(batch, dtype=torch.float32),
        )

    def reset(self) -> Observation:
        """Return to the pre-induction state and return the first observation."""
        self.state = self._fresh_state()
        self._action_window = torch.zeros(self.batch, self._window_steps(), 2, dtype=torch.float32)
        observed = self._observe()
        self._window = observed.unsqueeze(1).repeat(1, self._window_steps(), 1)
        return self._observation(observed)

    def _measure(self, truth: Tensor, channel: int) -> Tensor:
        sd = float(self._noise[channel])
        if sd == 0.0:
            return truth
        noise = torch.randn(truth.shape, generator=self._gen) * sd
        return truth + noise

    def _observe(self) -> Tensor:
        vitals = self.state.vitals
        effects = self._effects()
        true_bis = pd.depth_index(effects.combined, self.pd_cfg)
        bis = self._measure(true_bis, 0).clamp(0.0, 100.0)
        columns = [
            bis,
            self._measure(vitals.heart_rate, 1).clamp(20.0, 180.0),
            self._measure(vitals.map_mmHg, 2).clamp(20.0, 160.0),
            self._measure(vitals.systolic, 3).clamp(30.0, 220.0),
            self._measure(vitals.diastolic, 4).clamp(10.0, 140.0),
            self._measure(vitals.spo2, 5).clamp(50.0, 100.0),
            self._measure(vitals.etco2, 6).clamp(10.0, 90.0),
            self._measure(vitals.temperature, 7).clamp(33.0, 40.0),
            self._measure(vitals.train_of_four, 8).clamp(0.0, 4.0),
            vitals.dopamine,
            vitals.propranolol,
            self._measure(vitals.pneumo_pressure, 11).clamp(0.0, 25.0),
        ]
        return torch.stack(columns, dim=-1)

    def _effects(self) -> pd.Effects:
        state = self.state
        _, propofol_ce = pk.concentrations(state.propofol, self.patient.propofol)
        _, remifentanil_ce = pk.concentrations(state.remifentanil, self.patient.remifentanil)
        occupancy_p = pd.hill_occupancy(
            propofol_ce, self.patient.propofol_c50, self.pd_cfg.propofol_hill
        )
        occupancy_r = pd.hill_occupancy(
            remifentanil_ce, self.patient.remifentanil_c50, self.pd_cfg.remifentanil_hill
        )
        combined = pd.combined_effect(occupancy_p, occupancy_r, self.pd_cfg.opioid_synergy)
        return pd.Effects(
            propofol_ce=propofol_ce,
            propofol_occupancy=occupancy_p,
            remifentanil_occupancy=occupancy_r,
            combined=combined,
        )

    def _observation(self, current: Tensor) -> Observation:
        """One observation from a reading already drawn, so the window and the current agree."""
        return Observation(
            window=self._window.clone(),
            actions=self._action_window.clone(),
            current=current,
            bis=current[:, CHANNELS.index("bis")],
            map_mmHg=current[:, CHANNELS.index("map")],
        )

    def step(self, action: Tensor) -> Transition:
        """Advance one decision interval with a per-episode two-column action."""
        if action.shape != (self.batch, 2):
            raise ValueError(f"action must be {(self.batch, 2)}, got {tuple(action.shape)}")
        clipped = torch.stack(
            [
                action[:, 0].clamp(0.0, self.theatre.propofol_max),
                action[:, 1].clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )
        state = self.state
        propofol_mass = pk.mass_rate(clipped[:, 0], self.patient.covariates.weight_kg, 1.0 / 60.0)
        remifentanil_mass = pk.mass_rate(clipped[:, 1], self.patient.covariates.weight_kg, 1.0)
        dt_min = self._dt_min
        step_index = min(state.step, self.schedule.steps - 1)
        context = self.schedule.index(step_index)
        drive = pd.SurgicalDrive(
            insufflation_mmHg=context["insufflation_mmHg"],
            trendelenburg=context["trendelenburg"],
            stimulation=context["stimulation"],
            blood_loss_ml=context["blood_loss_ml"],
            rocuronium_block=context["rocuronium_block"],
        )

        effects = self._effects()
        vitals = pd.step_vitals(
            state.vitals,
            drive,
            effects,
            self.patient.haemodynamic,
            self.pd_cfg,
            dt_min,
        )
        propofol_state = pk.advance(
            state.propofol, propofol_mass, self.patient.propofol, dt_min, self.theatre.substeps
        )
        remifentanil_state = pk.advance(
            state.remifentanil,
            remifentanil_mass,
            self.patient.remifentanil,
            dt_min,
            self.theatre.substeps,
        )

        observed = self._observe()
        observed_bis = observed[:, CHANNELS.index("bis")]
        observed_map = observed[:, CHANNELS.index("map")]
        depth_error = (observed_bis - self.theatre.bis_target).abs()
        drug_cost = self.theatre.drug_cost_weight * (clipped[:, 0] + clipped[:, 1]).abs()
        reward = -depth_error - drug_cost

        hypotension = (observed_map < self.theatre.map_floor).to(torch.float32) * dt_min
        suppression = (observed_bis < self.theatre.bis_suppression).to(torch.float32) * dt_min
        variability = (clipped[:, 0] - state.previous_action[:, 0]).abs() + (
            clipped[:, 1] - state.previous_action[:, 1]
        ).abs()
        step_return = self._gamma_powers[min(state.step, self.steps)] * reward

        self.state = EnvState(
            propofol=propofol_state,
            remifentanil=remifentanil_state,
            vitals=vitals,
            step=state.step + 1,
            previous_action=clipped,
            cost_hypotension=state.cost_hypotension + hypotension,
            cost_suppression=state.cost_suppression + suppression,
            cost_variability=state.cost_variability + variability,
            episode_return=state.episode_return + step_return,
        )

        self._window = torch.cat([self._window[:, 1:, :], observed.unsqueeze(1)], dim=1)
        self._action_window = torch.cat(
            [self._action_window[:, 1:, :], clipped.unsqueeze(1)], dim=1
        )
        true_bis = pd.depth_index(effects.combined, self.pd_cfg)
        return Transition(
            observation=self._observation(observed),
            action=clipped,
            reward=reward,
            cost_hypotension=hypotension,
            cost_suppression=suppression,
            cost_variability=variability,
            done=self.state.step >= self.steps,
            propofol_ce=effects.propofol_ce,
            remifentanil_ce=pk.concentrations(self.state.remifentanil, self.patient.remifentanil)[
                1
            ],
            true_map=vitals.map_mmHg,
            true_bis=true_bis,
        )

    def true_depth(self) -> Tensor:
        """Noiseless depth index implied by the current effect-site state."""
        return pd.depth_index(self._effects().combined, self.pd_cfg)

    def peek(self, action: Tensor) -> ProbeStep:
        """Next-step depth and pressure for a candidate action, leaving the state untouched."""
        saved_state = self.state
        saved_window = self._window.clone()
        saved_actions = self._action_window.clone()
        saved_noise = self._gen.get_state()
        transition = self.step(action)
        result = ProbeStep(
            observed_bis=transition.observation.bis.clone(),
            observed_map=transition.observation.map_mmHg.clone(),
            true_bis=self.true_depth(),
            true_map=self.state.vitals.map_mmHg.clone(),
        )
        self.state = saved_state
        self._window = saved_window
        self._action_window = saved_actions
        self._gen.set_state(saved_noise)
        return result

    def effects(self) -> pd.Effects:
        """Current effect-site read, for diagnostics only."""
        return self._effects()

    def clone_state(self) -> EnvState:
        return replace(self.state)


def rollout_step_count(theatre: TheatreConfig) -> int:
    """Number of decisions in one episode."""
    return int(math.ceil(theatre.horizon_min * 60.0 / theatre.dt_s))
