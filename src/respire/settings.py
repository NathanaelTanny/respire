"""Typed configuration objects for the controller, the operating theatre and the rounds.

Every numeric literal carries the provenance class it belongs to: a value printed in the
article, a value implied by an equation, or an engineering default used where the article is
silent. The third class is enumerated in the verification report, so a reader can tell which
constants are ours.

Ref: Sec. 3.1 (decision cadence, return), Sec. 3.2-3.3 (operator, certificate), Sec. 5.1 (cohort).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a YAML document does not match the schema of its section."""


def _num(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer, got {type(value).__name__}")
    return value


def _str(raw: Mapping[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean, got {type(value).__name__}")
    return value


def _floats(raw: Mapping[str, Any], key: str, default: Sequence[float]) -> tuple[float, ...]:
    value = raw.get(key, list(default))
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ConfigError(f"{key} must be a sequence of numbers")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{key} must contain only numbers")
        out.append(float(item))
    return tuple(out)


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _optional_section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


@dataclass(frozen=True)
class TheatreConfig:
    """Timing, reward and measurement constants of the operating theatre."""

    dt_s: float = 5.0
    horizon_min: float = 120.0
    bis_target: float = 50.0
    bis_noise_sd: float = 2.1
    map_floor: float = 65.0
    bis_suppression: float = 20.0
    gamma: float = 0.995
    drug_cost_weight: float = 0.02
    propofol_scale: float = 10.0
    remifentanil_scale: float = 0.5
    propofol_max: float = 12.0
    remifentanil_max: float = 0.5
    substeps: int = 4

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TheatreConfig:
        base = cls()
        return cls(
            dt_s=_num(raw, "dt_s", base.dt_s),
            horizon_min=_num(raw, "horizon_min", base.horizon_min),
            bis_target=_num(raw, "bis_target", base.bis_target),
            bis_noise_sd=_num(raw, "bis_noise_sd", base.bis_noise_sd),
            map_floor=_num(raw, "map_floor", base.map_floor),
            bis_suppression=_num(raw, "bis_suppression", base.bis_suppression),
            gamma=_num(raw, "gamma", base.gamma),
            drug_cost_weight=_num(raw, "drug_cost_weight", base.drug_cost_weight),
            propofol_scale=_num(raw, "propofol_scale", base.propofol_scale),
            remifentanil_scale=_num(raw, "remifentanil_scale", base.remifentanil_scale),
            propofol_max=_num(raw, "propofol_max", base.propofol_max),
            remifentanil_max=_num(raw, "remifentanil_max", base.remifentanil_max),
            substeps=_int(raw, "substeps", base.substeps),
        )

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_min * 60.0 / self.dt_s))


@dataclass(frozen=True)
class PharmacokineticConfig:
    """Mammillary transport constants and covariate scalings for the two drugs.

    The typical-value set is the adult reference set of the two published models; the covariate
    exponents are allometric defaults because the article cites the models without reprinting
    their covariate equations.
    """

    propofol_v1: float = 6.28
    propofol_v2: float = 25.5
    propofol_v3: float = 273.0
    propofol_cl: float = 1.79
    propofol_q2: float = 1.75
    propofol_q3: float = 1.11
    propofol_ke0: float = 0.146
    remifentanil_v1: float = 5.1
    remifentanil_v2: float = 9.82
    remifentanil_v3: float = 5.42
    remifentanil_cl: float = 2.6
    remifentanil_q2: float = 2.05
    remifentanil_q3: float = 0.61
    remifentanil_ke0: float = 0.595
    weight_exponent: float = 0.75
    age_exponent: float = -0.25
    effect_site_exponent: float = -0.25

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PharmacokineticConfig:
        base = cls()
        fields = {
            "propofol_v1": base.propofol_v1,
            "propofol_v2": base.propofol_v2,
            "propofol_v3": base.propofol_v3,
            "propofol_cl": base.propofol_cl,
            "propofol_q2": base.propofol_q2,
            "propofol_q3": base.propofol_q3,
            "propofol_ke0": base.propofol_ke0,
            "remifentanil_v1": base.remifentanil_v1,
            "remifentanil_v2": base.remifentanil_v2,
            "remifentanil_v3": base.remifentanil_v3,
            "remifentanil_cl": base.remifentanil_cl,
            "remifentanil_q2": base.remifentanil_q2,
            "remifentanil_q3": base.remifentanil_q3,
            "remifentanil_ke0": base.remifentanil_ke0,
            "weight_exponent": base.weight_exponent,
            "age_exponent": base.age_exponent,
            "effect_site_exponent": base.effect_site_exponent,
        }
        return cls(**{name: _num(raw, name, default) for name, default in fields.items()})


@dataclass(frozen=True)
class PharmacodynamicConfig:
    """Depth index and haemodynamic response surfaces."""

    bis_awake: float = 93.5
    bis_floor: float = 6.5
    propofol_c50: float = 3.0
    propofol_hill: float = 1.5
    remifentanil_c50: float = 10.0
    remifentanil_hill: float = 2.0
    opioid_synergy: float = 0.35
    map_baseline: float = 88.0
    map_propofol_slope: float = 7.5
    map_remifentanil_slope: float = 10.0
    map_insufflation_slope: float = 0.55
    map_trendelenburg_gain: float = 4.5
    map_stimulation_gain: float = 6.0
    map_tau_s: float = 45.0
    hr_baseline: float = 70.0
    hr_baroreflex: float = 0.65
    hr_remifentanil_slope: float = 15.0
    hr_tau_s: float = 30.0
    spo2_baseline: float = 98.5
    spo2_depth_gain: float = 0.35
    etco2_baseline: float = 34.0
    etco2_depth_gain: float = 0.45

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PharmacodynamicConfig:
        base = cls()
        names = (
            "bis_awake",
            "bis_floor",
            "propofol_c50",
            "propofol_hill",
            "remifentanil_c50",
            "remifentanil_hill",
            "opioid_synergy",
            "map_baseline",
            "map_propofol_slope",
            "map_remifentanil_slope",
            "map_insufflation_slope",
            "map_trendelenburg_gain",
            "map_stimulation_gain",
            "map_tau_s",
            "hr_baseline",
            "hr_baroreflex",
            "hr_remifentanil_slope",
            "hr_tau_s",
            "spo2_baseline",
            "spo2_depth_gain",
            "etco2_baseline",
            "etco2_depth_gain",
        )
        return cls(**{name: _num(raw, name, getattr(base, name)) for name in names})


@dataclass(frozen=True)
class SurgicalConfig:
    """Insufflation, positioning and stimulation schedule of a robot-assisted case."""

    pneumo_pressure_mmHg: float = 12.0
    insufflation_onset_min: float = 8.0
    insufflation_ramp_min: float = 3.0
    trendelenburg_onset_min: float = 10.0
    trendelenburg_fraction: float = 0.6
    stimulation_rate_per_h: float = 6.0
    stimulation_duration_s: float = 60.0
    stimulation_gain_sd: float = 0.3
    blood_loss_ml_per_min: float = 1.4
    blood_loss_onset_min: float = 20.0
    rocuronium_bolus_min: tuple[float, ...] = (5.0, 35.0, 75.0)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SurgicalConfig:
        base = cls()
        return cls(
            pneumo_pressure_mmHg=_num(raw, "pneumo_pressure_mmHg", base.pneumo_pressure_mmHg),
            insufflation_onset_min=_num(raw, "insufflation_onset_min", base.insufflation_onset_min),
            insufflation_ramp_min=_num(raw, "insufflation_ramp_min", base.insufflation_ramp_min),
            trendelenburg_onset_min=_num(
                raw, "trendelenburg_onset_min", base.trendelenburg_onset_min
            ),
            trendelenburg_fraction=_num(raw, "trendelenburg_fraction", base.trendelenburg_fraction),
            stimulation_rate_per_h=_num(raw, "stimulation_rate_per_h", base.stimulation_rate_per_h),
            stimulation_duration_s=_num(raw, "stimulation_duration_s", base.stimulation_duration_s),
            stimulation_gain_sd=_num(raw, "stimulation_gain_sd", base.stimulation_gain_sd),
            blood_loss_ml_per_min=_num(raw, "blood_loss_ml_per_min", base.blood_loss_ml_per_min),
            blood_loss_onset_min=_num(raw, "blood_loss_onset_min", base.blood_loss_onset_min),
            rocuronium_bolus_min=_floats(raw, "rocuronium_bolus_min", base.rocuronium_bolus_min),
        )


@dataclass(frozen=True)
class CohortConfig:
    """Cohort size and covariate spread (Sec. 5.1)."""

    n_patients: int = 2000
    train_episodes: int = 20000
    eval_episodes: int = 2000
    encoder_cases: int = 192
    variability: float = 1.25
    weight_mean: float = 74.0
    weight_sd: float = 13.0
    height_mean: float = 168.0
    height_sd: float = 9.0
    age_mean: float = 52.0
    age_sd: float = 15.0
    female_fraction: float = 0.46
    sensitivity_sd: float = 1.0
    seed: int = 7

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CohortConfig:
        base = cls()
        return cls(
            n_patients=_int(raw, "n_patients", base.n_patients),
            train_episodes=_int(raw, "train_episodes", base.train_episodes),
            eval_episodes=_int(raw, "eval_episodes", base.eval_episodes),
            encoder_cases=_int(raw, "encoder_cases", base.encoder_cases),
            variability=_num(raw, "variability", base.variability),
            weight_mean=_num(raw, "weight_mean", base.weight_mean),
            weight_sd=_num(raw, "weight_sd", base.weight_sd),
            height_mean=_num(raw, "height_mean", base.height_mean),
            height_sd=_num(raw, "height_sd", base.height_sd),
            age_mean=_num(raw, "age_mean", base.age_mean),
            age_sd=_num(raw, "age_sd", base.age_sd),
            female_fraction=_num(raw, "female_fraction", base.female_fraction),
            sensitivity_sd=_num(raw, "sensitivity_sd", base.sensitivity_sd),
            seed=_int(raw, "seed", base.seed),
        )


@dataclass(frozen=True)
class EncoderConfig:
    """Shape of the latent physiological pattern encoder (Fig. 2, Sec. 4.2)."""

    n_channels: int = 12
    window_steps: int = 12
    window_s: float = 60.0
    n_blocks: int = 6
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 4
    latent_dim: int = 16
    axis_dim: int = 3
    hidden_dim: int = 32
    sparse_topk: int = 4
    action_channels: int = 2
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 12
    interventional_weight: float = 0.5
    one_step_horizon: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EncoderConfig:
        base = cls()
        ints = (
            "n_channels",
            "window_steps",
            "n_blocks",
            "d_model",
            "d_state",
            "d_conv",
            "expand",
            "n_heads",
            "latent_dim",
            "axis_dim",
            "hidden_dim",
            "sparse_topk",
            "action_channels",
            "batch_size",
            "epochs",
            "one_step_horizon",
        )
        nums = (
            "window_s",
            "dropout",
            "lr",
            "weight_decay",
            "interventional_weight",
        )
        values: dict[str, Any] = {name: _int(raw, name, getattr(base, name)) for name in ints}
        values.update({name: _num(raw, name, getattr(base, name)) for name in nums})
        return cls(**values)


@dataclass(frozen=True)
class VerifierConfig:
    """Return verifier with a heteroscedastic head (Fig. 2, Algorithm 1 line 3)."""

    hidden: tuple[int, int] = (640, 320)
    feature_dim: int = 256
    target_params: float = 400000.0
    param_tolerance: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 8
    ensemble: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VerifierConfig:
        base = cls()
        return cls(
            hidden=(
                int(_floats(raw, "hidden", base.hidden)[0]),
                int(_floats(raw, "hidden", base.hidden)[1]),
            ),
            feature_dim=_int(raw, "feature_dim", base.feature_dim),
            target_params=_num(raw, "target_params", base.target_params),
            param_tolerance=_num(raw, "param_tolerance", base.param_tolerance),
            lr=_num(raw, "lr", base.lr),
            weight_decay=_num(raw, "weight_decay", base.weight_decay),
            batch_size=_int(raw, "batch_size", base.batch_size),
            epochs=_int(raw, "epochs", base.epochs),
            ensemble=_int(raw, "ensemble", base.ensemble),
        )


@dataclass(frozen=True)
class PolicyConfig:
    """Conditional flow-matching policy over the two infusion rates (Fig. 2, Eq. (4))."""

    action_dim: int = 2
    hidden: int = 128
    depth: int = 3
    time_features: int = 16
    euler_steps: int = 4
    latent_dim: int = 16
    hidden_state_dim: int = 32
    axis_dim: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 6
    bc_epochs: int = 12
    grad_clip: float = 1.0
    sigma_min: float = 1e-4

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PolicyConfig:
        base = cls()
        ints = (
            "action_dim",
            "hidden",
            "depth",
            "time_features",
            "euler_steps",
            "latent_dim",
            "hidden_state_dim",
            "axis_dim",
            "batch_size",
            "epochs",
        )
        nums = ("lr", "weight_decay", "grad_clip", "sigma_min")
        values: dict[str, Any] = {name: _int(raw, name, getattr(base, name)) for name in ints}
        values.update({name: _num(raw, name, getattr(base, name)) for name in nums})
        return cls(**values)


@dataclass(frozen=True)
class ShieldConfig:
    """Analytic one-step shield and the pressure model it is built on (Sec. 4, Fig. 6)."""

    map_floor: float = 65.0
    safety_margin: float = 3.0
    lipschitz: float = 1.0
    noise_eta: float = 2.0
    grid_points: int = 25

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ShieldConfig:
        base = cls()
        return cls(
            map_floor=_num(raw, "map_floor", base.map_floor),
            safety_margin=_num(raw, "safety_margin", base.safety_margin),
            lipschitz=_num(raw, "lipschitz", base.lipschitz),
            noise_eta=_num(raw, "noise_eta", base.noise_eta),
            grid_points=_int(raw, "grid_points", base.grid_points),
        )


@dataclass(frozen=True)
class RoundConfig:
    """Recursion schedule and admission hyper-parameters (Algorithm 1, Eq. (3)-(4))."""

    rounds: int = 5
    alpha: float = 0.1
    delta: float = 0.05
    epsilon: float = 0.01
    admitted_fraction: float = 0.5
    trust_region: float = 0.08
    kl_weight: float = 1.0
    dual_lr: float = 0.05
    dual_init: float = 0.1
    cost_budget_hypotension_min: float = 10.0
    cost_budget_suppression_min: float = 1.0
    cost_budget_variability_quantile: float = 0.9
    coverage_se_multiple: float = 2.0
    pool_episodes: int = 512
    reference_episodes: int = 512
    calibration_fraction: float = 0.5
    replay_fraction: float = 0.3
    perturb_fraction: float = 0.05
    backtrack_steps: int = 6

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RoundConfig:
        base = cls()
        ints = ("rounds", "pool_episodes", "reference_episodes", "backtrack_steps")
        nums = (
            "alpha",
            "delta",
            "epsilon",
            "admitted_fraction",
            "trust_region",
            "kl_weight",
            "dual_lr",
            "dual_init",
            "cost_budget_hypotension_min",
            "cost_budget_suppression_min",
            "cost_budget_variability_quantile",
            "coverage_se_multiple",
            "calibration_fraction",
            "replay_fraction",
            "perturb_fraction",
        )
        values: dict[str, Any] = {name: _int(raw, name, getattr(base, name)) for name in ints}
        values.update({name: _num(raw, name, getattr(base, name)) for name in nums})
        return cls(**values)


@dataclass(frozen=True)
class EvalConfig:
    """Replay protocol and reporting knobs."""

    heldout_cases: int = 226
    robot_assisted_cases: int = 1132
    seeds: int = 5
    bootstrap_samples: int = 2000
    bootstrap_alpha: float = 0.05
    permutation_samples: int = 5000
    quartiles: int = 4
    depth_noise_floor: float = 2.1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalConfig:
        base = cls()
        ints = (
            "heldout_cases",
            "robot_assisted_cases",
            "seeds",
            "bootstrap_samples",
            "permutation_samples",
            "quartiles",
        )
        nums = ("bootstrap_alpha", "depth_noise_floor")
        values: dict[str, Any] = {name: _int(raw, name, getattr(base, name)) for name in ints}
        values.update({name: _num(raw, name, getattr(base, name)) for name in nums})
        return cls(**values)


@dataclass(frozen=True)
class ExperimentConfig:
    """One resolved experiment document."""

    name: str = "primary"
    seed: int = 11
    output_dir: str = "runs"
    theatre: TheatreConfig = field(default_factory=TheatreConfig)
    pharmacokinetics: PharmacokineticConfig = field(default_factory=PharmacokineticConfig)
    pharmacodynamics: PharmacodynamicConfig = field(default_factory=PharmacodynamicConfig)
    surgical: SurgicalConfig = field(default_factory=SurgicalConfig)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    rounds: RoundConfig = field(default_factory=RoundConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    variants: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        variants = raw.get("variants", [])
        if not isinstance(variants, Sequence) or isinstance(variants, str):
            raise ConfigError("variants must be a sequence of names")
        names: list[str] = []
        for item in variants:
            if not isinstance(item, str):
                raise ConfigError("variants must contain strings")
            names.append(item)
        return cls(
            name=_str(raw, "name", "primary"),
            seed=_int(raw, "seed", 11),
            output_dir=_str(raw, "output_dir", "runs"),
            theatre=TheatreConfig.from_dict(_section(raw, "theatre")),
            pharmacokinetics=PharmacokineticConfig.from_dict(_section(raw, "pharmacokinetics")),
            pharmacodynamics=PharmacodynamicConfig.from_dict(_section(raw, "pharmacodynamics")),
            surgical=SurgicalConfig.from_dict(_section(raw, "surgical")),
            cohort=CohortConfig.from_dict(_section(raw, "cohort")),
            encoder=EncoderConfig.from_dict(_section(raw, "encoder")),
            verifier=VerifierConfig.from_dict(_section(raw, "verifier")),
            policy=PolicyConfig.from_dict(_section(raw, "policy")),
            shield=ShieldConfig.from_dict(_section(raw, "shield")),
            rounds=RoundConfig.from_dict(_section(raw, "rounds")),
            evaluation=EvalConfig.from_dict(_section(raw, "evaluation")),
            variants=tuple(names),
        )


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides with YAML scalar parsing."""
    out = dict(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        key, _, literal = item.partition("=")
        parsed: Any = yaml.safe_load(literal)
        parts = key.split(".")
        cursor = out
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = parsed
    return out


def load_experiment(
    path: str | Path,
    overrides: Sequence[str] = (),
    base_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load an experiment document, optionally inheriting a base document."""
    raw = read_yaml(path)
    if base_path is not None:
        merged = read_yaml(base_path)
        merged.update(raw)
        raw = merged
    return ExperimentConfig.from_dict(apply_overrides(raw, overrides))
