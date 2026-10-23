"""Empirical study of the admission certificate.

Two arms. In the exchangeable arm the pool is drawn from the same distribution as the calibration
residuals, which is the condition of Proposition 1, and the measured coverage should reach the
target. In the shifted arm the pool is drawn from a policy that ran deeper, which is exactly the
split the article warns is not exchangeable, and coverage falls below the target. The gap between
the two arms is why the calibration split is regenerated every round.

Ref: Proposition 1 (Eq. (5)), Sec. 3.2 (the exchangeability caveat), Algorithm 1 lines 6-9.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from respire.cli.common import base_parser, emit, output_directory, resolve, write_json
from respire.control.certificate import ConformalCertificate
from respire.pipeline import prepare_data
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import episode_patients, make_environment
from respire.theatre.rollout import rollout
from respire.utils.logging_setup import configure


def coverage_study(
    calibration: Tensor,
    exchangeable: Tensor,
    shifted: Tensor,
    alpha: float,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    """Coverage of the lower bound built from a constant predictor and the calibration residual."""

    def measure(
        pool: Tensor, generator: torch.Generator, centre: float, quantile: float
    ) -> tuple[float, float]:
        sample = pool[
            torch.randint(0, pool.numel(), (min(256, pool.numel()),), generator=generator)
        ]
        covered = (sample >= centre - quantile).to(torch.float32)
        mean = float(covered.mean())
        return mean, math.sqrt(max(mean * (1.0 - mean), 0.0) / covered.numel())

    certificate = ConformalCertificate(alpha=alpha)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    centre = float(calibration.mean())
    quantile = certificate.quantile(calibration - centre)
    exchange_total = 0.0
    shifted_total = 0.0
    exchange_se = 0.0
    shifted_se = 0.0
    for _ in range(repetitions):
        value, error = measure(exchangeable, generator, centre, quantile)
        exchange_total += value
        exchange_se += error
        value, error = measure(shifted, generator, centre, quantile)
        shifted_total += value
        shifted_se += error
    return {
        "alpha": alpha,
        "target": 1.0 - alpha,
        "quantile": quantile,
        "predictor": centre,
        "exchangeable_coverage": exchange_total / repetitions,
        "exchangeable_se": exchange_se / repetitions,
        "shifted_coverage": shifted_total / repetitions,
        "shifted_se": shifted_se / repetitions,
        "repetitions": float(repetitions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("measure certificate coverage under an exchangeable and a shifted split")
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    data = prepare_data(config, config.seed)
    shifted_patients = episode_patients(data.cohort, "development", 32, config.cohort.seed + 91)
    shifted = rollout(
        make_environment(shifted_patients, config, config.cohort.seed + 91),
        RuleBasedAnaesthetist(config.theatre, propofol_base=3.0, noise_sd=0.05),
    )
    report = coverage_study(
        data.calibration.episode_return,
        data.calibration.episode_return,
        shifted.episode_return,
        config.rounds.alpha,
        args.repetitions,
        config.seed,
    )
    print(emit([dict(report)], ["alpha", "target", "exchangeable_coverage", "shifted_coverage"]))
    directory = output_directory(config, "certificate")
    write_json(directory / "coverage.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
